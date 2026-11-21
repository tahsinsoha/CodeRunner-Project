"""Check the LLM game logs and build one summary table from them.

    python summarize_llm.py                       # results/llm_runs/*.jsonl -> results/llm_results.csv
    python summarize_llm.py --runs results/llm_runs --csv results/llm_results.csv

Checks every saved game before counting it:
  - one record per game index 0..N-1, and the secret is the one secret_list() gives
    for that index (the same words every other solver was tested on);
  - every non-winning feedback line has exactly n_lies wrong tiles, and a win is
    shown truthfully (so the environment really played Wordle / FibbleN);
  - no guess is repeated, and `won` matches the last line being all green.

Writes one row per setting with the same first columns as evaluate.py, plus a
95% Wilson interval for the win rate and LLM usage per game.

Fallbacks: when the model fails 3 times in a row to give a playable word, the
solver plays the first word that fits the feedback, a guess made by our filter,
not the model. `win_rate_llm_only` counts only games won without any fallback,
which is the model's own result; `win_rate` is what the harness achieved.

Some words were played twice (a run stopped for lack of API credit let its games
in progress finish while the restarted run played them again). Both are complete
games; which one counts is chosen by a hash of the word, never by the result, so
the choice cannot favour wins or losses. `duplicates` says how many words this
applied to, and `win_rate_both_plays` averages both plays as a check.
"""

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

from wordle_fibble.evaluation import secret_list

NAME = re.compile(r"^(?P<solver>.+)_(?P<game>Wordle|Fibble[1-5])_g(?P<guesses>\d+)$")
GAMES = ["Wordle", "Fibble1", "Fibble2", "Fibble3", "Fibble4", "Fibble5"]


def wilson(wins, n, z=1.96):
    """95% Wilson score interval for a win rate (as fractions)."""
    if n == 0:
        return 0.0, 0.0
    p = wins / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def check_game(r, n_lies, max_guesses, secret):
    """List of problems with one game record (empty if it is fine)."""
    problems = []
    if r["secret"] != secret:
        problems.append(f"secret {r['secret']} != expected {secret}")
    if len(r["guesses"]) != r["n_guesses"] or r["n_guesses"] > max_guesses:
        problems.append("guess count")
    if len(set(r["guesses"])) != len(r["guesses"]):
        problems.append("repeated guess")
    for i, (shown, truth) in enumerate(zip(r["shown"], r["truth"])):
        last_win = r["won"] and i == len(r["truth"]) - 1
        wrong = sum(a != b for a, b in zip(shown, truth))
        if last_win and (truth != "GGGGG" or wrong):
            problems.append(f"line {i + 1}: win not shown truthfully")
        elif not last_win and wrong != n_lies:
            problems.append(f"line {i + 1}: {wrong} lies, expected {n_lies}")
    if r["won"] != (r["truth"][-1] == "GGGGG"):
        problems.append("won flag does not match the last line")
    if not r["won"] and r["n_guesses"] != max_guesses:
        problems.append("lost before using all guesses")
    return problems


def pick_play(plays):
    """One record per word, chosen by a hash of the word (blind to the outcome)."""
    if len(plays) == 1:
        return plays[0]
    h = int(hashlib.sha256(plays[0]["secret"].encode()).hexdigest(), 16)
    return plays[h % len(plays)]


def load_setting(path, n_games, seed):
    m = NAME.match(path.stem)
    if not m:
        raise ValueError(f"unexpected file name {path.name}")
    game, max_guesses = m["game"], int(m["guesses"])
    n_lies = 0 if game == "Wordle" else int(game[-1])
    plays = defaultdict(list)
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            plays[r["index"]].append(r)
    records = {i: pick_play(v) for i, v in plays.items()}
    lines = sum(len(v) for v in plays.values())
    secrets = secret_list(n_games, seed=seed)
    problems = []
    if sorted(records) != list(range(n_games)):
        problems.append(f"has games {len(records)}/{n_games}")
    for i, r in records.items():
        if i < n_games:
            problems += [f"game {i} ({r['secret']}): {p}"
                         for p in check_game(r, n_lies, max_guesses, secrets[i])]
    for i, v in plays.items():
        for r in v[1:]:
            problems += [f"game {i} ({r['secret']}, 2nd play): {p}"
                         for p in check_game(r, n_lies, max_guesses, secrets[i]) if i < n_games]
    return m["solver"], game, max_guesses, records, plays, lines, problems


def summarize(solver, game, max_guesses, records, plays):
    rs = [records[i] for i in sorted(records)]
    n = len(rs)
    wins = sum(r["won"] for r in rs)
    lo, hi = wilson(wins, n)
    by_round = [0] * max_guesses
    for r in rs:
        if r["won"]:
            by_round[r["n_guesses"] - 1] += 1
    calls = sum(r["llm_calls"] for r in rs)
    return {
        "game": game,
        "solver": solver,
        "games": n,
        "max_guesses": max_guesses,
        "win_rate": round(wins / n, 4),
        "mean_guesses": round(sum(r["n_guesses"] for r in rs) / n, 3),
        "wins_by_round": by_round,
        "win_rate_ci95_low": round(lo, 4),
        "win_rate_ci95_high": round(hi, 4),
        "calls_per_game": round(calls / n, 2),
        "retry_share": round(sum(r["retries"] for r in rs) / calls, 3) if calls else 0.0,
        "fallbacks": sum(r["fallbacks"] for r in rs),
        "inconsistent_played": sum(r["inconsistent_played"] for r in rs),
        "input_tokens": sum(r["input_tokens"] for r in rs),
        "output_tokens": sum(r["output_tokens"] for r in rs),
        "mean_seconds_per_game": round(sum(r["seconds"] for r in rs) / n, 1),
        "games_with_fallback": sum(any(t["fallback"] for t in r["turns"]) for r in rs),
        "wins_by_fallback_guess": sum(r["won"] and r["turns"][-1]["fallback"] for r in rs),
        "win_rate_llm_only": round(sum(r["won"] and not any(t["fallback"] for t in r["turns"])
                                       for r in rs) / n, 4),
        "duplicates": sum(len(v) > 1 for v in plays.values()),
        "win_rate_both_plays": round(sum(sum(r["won"] for r in v) / len(v) for v in plays.values()) / n, 4),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="results/llm_runs")
    p.add_argument("--csv", default="results/llm_results.csv")
    p.add_argument("--games", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rows, bad = [], 0
    for path in sorted(Path(args.runs).glob("*.jsonl")):
        solver, game, g, records, plays, lines, problems = load_setting(path, args.games, args.seed)
        extra = f", {lines - len(records)} words played twice" if lines > len(records) else ""
        status = "ok" if not problems else f"{len(problems)} PROBLEMS"
        print(f"{path.name:<46} {len(records):>3} games{extra}  {status}")
        for pr in problems[:5]:
            print("      ", pr)
        bad += bool(problems)
        rows.append(summarize(solver, game, g, records, plays))

    rows.sort(key=lambda r: (r["solver"], GAMES.index(r["game"])))
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    table = defaultdict(dict)
    for r in rows:
        table[r["solver"]][r["game"]] = r
    print(f"\nWin rate % with fallback / LLM only (no fallback in the game), {args.games} games per cell")
    print(f"{'solver':<28}" + "".join(f"{g:>12}" for g in GAMES))
    for solver in sorted(table):
        cells = [table[solver].get(g) for g in GAMES]
        print(f"{solver:<28}" + "".join(
            f"{100 * c['win_rate']:>6.0f} /{100 * c['win_rate_llm_only']:>3.0f}" + " " if c else f"{'-':>12}"
            for c in cells))
    print(f"\nwrote {args.csv}; {len(rows)} settings, {bad} with problems")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
