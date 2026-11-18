"""Play many LLM games at once and save each game as soon as it ends.

LLM games spend almost all their time waiting for the model's reply, so we
run several in threads (Python's GIL does not matter while waiting). Every
game uses its own environment and solver, the same secret list and the same
per-game seed as `evaluation.evaluate`, so results line up with the other
solvers.

Each finished game is appended as one JSON line to the setting's .jsonl file.
Running the same setting again skips the games already in the file, so an
interrupted run (or a run that hit an API limit) picks up where it stopped.
"""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .env import FibbleEnv
from .feedback import COLOR_CHARS
from .filter import CandidateFilter
from .llm_solver import LLMSolver
from .words import load_allowed, load_answers


def colors_str(colors):
    return "".join(COLOR_CHARS[int(c)] for c in colors)


def play_game(index, secret, make_llm, n_lies, max_guesses, seed, opener, retries,
              invalid_retries, candidate_filter, answers, allowed, keep_replies=True):
    """One full game. Returns a JSON-ready record."""
    env = FibbleEnv(n_lies=n_lies, max_guesses=max_guesses, answers=answers, allowed=allowed,
                    initial_random_guess=opener)
    solver = LLMSolver(make_llm(), retries=retries, invalid_retries=invalid_retries,
                       candidate_filter=candidate_filter, allowed=allowed, keep_replies=keep_replies)
    obs = env.reset(secret=secret, seed=seed + index)
    solver.reset()
    start = time.perf_counter()
    while not env.done:
        obs, _, _ = env.step(solver.act(obs))
    turns = solver.turns
    return {
        "index": index,
        "secret": secret,
        "won": bool(env.won),
        "n_guesses": len(env.guesses),
        "guesses": list(env.guesses),
        "shown": [colors_str(c) for c in env.shown_history],
        "truth": [colors_str(c) for c in env.true_history],
        "llm_calls": sum(t.llm_calls for t in turns),
        "retries": sum(t.retries for t in turns),
        "fallbacks": sum(t.fallback for t in turns),
        "inconsistent_played": sum(t.inconsistent_played for t in turns),
        "input_tokens": sum(t.input_tokens for t in turns),
        "output_tokens": sum(t.output_tokens for t in turns),
        "seconds": round(time.perf_counter() - start, 2),
        "turns": [{"guess": t.guess, "llm_calls": t.llm_calls, "problems": t.problems,
                   "fallback": t.fallback, "inconsistent_played": t.inconsistent_played,
                   "input_tokens": t.input_tokens, "output_tokens": t.output_tokens,
                   "reasoning_chars": t.reasoning_chars, "seconds": t.seconds,
                   "replies": t.replies} for t in turns],
    }


def load_records(path):
    path = Path(path)
    if not path.exists():
        return {}
    records = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            records[r["index"]] = r
    return records


def summarize(records, game, solver_name, max_guesses, seconds):
    """Same first columns as evaluation.evaluate, plus LLM usage columns."""
    rs = [records[i] for i in sorted(records)]
    wins_by_round = np.zeros(max_guesses + 1, dtype=int)
    for r in rs:
        if r["won"]:
            wins_by_round[r["n_guesses"]] += 1
    n = len(rs)
    return {
        "game": game,
        "solver": solver_name,
        "games": n,
        "max_guesses": max_guesses,
        "win_rate": sum(r["won"] for r in rs) / n if n else 0.0,
        "mean_guesses": float(np.mean([r["n_guesses"] for r in rs])) if n else 0.0,
        "wins_by_round": wins_by_round[1:].tolist(),
        "seconds": round(seconds, 1),
        "llm_calls": sum(r["llm_calls"] for r in rs),
        "retries": sum(r["retries"] for r in rs),
        "fallbacks": sum(r["fallbacks"] for r in rs),
        "inconsistent_played": sum(r["inconsistent_played"] for r in rs),
        "input_tokens": sum(r["input_tokens"] for r in rs),
        "output_tokens": sum(r["output_tokens"] for r in rs),
    }


def run_setting(make_llm, solver_name, secrets, n_lies, max_guesses, out_path, workers=4,
                seed=0, opener=False, retries=5, invalid_retries=3, candidate_filter=None,
                keep_replies=True, log=print):
    """Play every secret once (skipping games already in out_path) and return the summary row."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    answers, allowed = load_answers(), load_allowed()
    candidate_filter = candidate_filter or CandidateFilter(answers)
    records = load_records(out_path)
    todo = [i for i in range(len(secrets)) if i not in records]
    game = "Wordle" if n_lies == 0 else f"Fibble{n_lies}"
    if records:
        log(f"[{solver_name} {game}] {len(records)} games already done, {len(todo)} to play")
    lock = threading.Lock()
    start = time.perf_counter()

    def task(i):
        return play_game(i, secrets[i], make_llm, n_lies, max_guesses, seed, opener, retries,
                         invalid_retries, candidate_filter, answers, allowed, keep_replies)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(task, i): i for i in todo}
        for fut in as_completed(futures):
            r = fut.result()
            with lock:
                with open(out_path, "a") as f:
                    f.write(json.dumps(r) + "\n")
                records[r["index"]] = r
                done = len(records)
            log(f"[{solver_name} {game}] game {done}/{len(secrets)}: {r['secret']} "
                f"{'WON' if r['won'] else 'lost'} in {r['n_guesses']}  "
                f"(calls {r['llm_calls']}, retries {r['retries']}, {r['seconds']}s)")
    return summarize({i: records[i] for i in range(len(secrets)) if i in records},
                     game, solver_name, max_guesses, time.perf_counter() - start)
