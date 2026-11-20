"""Offline tests for the LLM solver and runner, with a scripted fake LLM."""

import json
import threading

import pytest

from wordle_fibble import FibbleEnv, score
from wordle_fibble.evaluation import evaluate, secret_list
from wordle_fibble.feedback import GRAY, GREEN, YELLOW
from wordle_fibble.filter import CandidateFilter
from wordle_fibble.llm_client import LLMReply
from wordle_fibble.llm_runner import RunStopped, load_records, run_setting
from wordle_fibble.llm_solver import (
    LLMSolver,
    feedback_block,
    parse_guess,
    system_prompt,
    turn_prompt,
    wordle_hint,
)
from wordle_fibble.words import load_allowed

G, Y, X = GREEN, YELLOW, GRAY
ALLOWED = set(load_allowed())


class ScriptedLLM:
    """Returns the given replies in order, then repeats the last one.

    A reply can be a (text, finish_reason) pair to fake a cut-off answer.
    """

    label = "fake"

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, messages):
        self.calls.append([dict(m) for m in messages])
        item = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        text, finish = item if isinstance(item, tuple) else (item, "stop")
        return LLMReply(text=text, input_tokens=10, output_tokens=5, finish_reason=finish)


@pytest.fixture(scope="module")
def cf():
    return CandidateFilter()


def obs_after(secret, guesses, n_lies=0, max_guesses=6, seed=0):
    env = FibbleEnv(n_lies=n_lies, max_guesses=max_guesses, seed=seed)
    obs = env.reset(secret=secret)
    for g in guesses:
        obs, _, _ = env.step(g)
    return obs


# --- prompts -------------------------------------------------------------------

def test_feedback_block_matches_listing_2():
    assert feedback_block("arise", [X, X, G, Y, Y]) == (
        "Guess: ARISE\nFeedback:\nA: incorrect\nR: incorrect\nI: correct\nS: present\nE: present")


def test_turn_prompt_lists_history_and_tries():
    obs = obs_after("early", ["slate", "coral"])
    text = turn_prompt(obs)
    assert text.startswith("Guess: SLATE\nFeedback:\n")
    assert "Guess: CORAL" in text and text.endswith("You have 4 tries remaining.")
    assert turn_prompt(obs_after("early", [])) == "You have 6 tries remaining."


def test_system_prompt_lies_line():
    assert "lie present" not in system_prompt(0)
    assert system_prompt(1).endswith("There is one lie present in the game")
    assert system_prompt(3).endswith("There are 3 lies present in the game")


# --- parsing -------------------------------------------------------------------

@pytest.mark.parametrize("reply, expected", [
    ("CRANE - common letters", "crane"),
    ("crane\n\nReason: common letters", "crane"),
    ("Guess: CRANE\nbecause ...", "crane"),
    ("**slate** because it has common letters", "slate"),
    ("My guess is 'adieu' to test vowels", "adieu"),
    ("SOARE\n\nReason: Starting with common letters.", "soare"),
    ("", ""),
    ("no idea", ""),
    ("I think the answer is CRANE", "crane"),
    ("Guess: crane because it is common", "crane"),
    ("Sure, crane is a good start", "crane"),
    ("Based on the feedback, early fits", "early"),
    ("Hello there, my guess is 'CRANE'", "crane"),
])
def test_parse_guess(reply, expected):
    assert parse_guess(reply, ALLOWED) == expected


def test_parse_guess_not_in_list_still_returned():
    # The model's first word is its answer, even when it is not a real word.
    assert parse_guess("ZZZZZ is my guess", ALLOWED) == "zzzzz"
    assert parse_guess("Xyzzy, because it sounds nice", ALLOWED) == "xyzzy"


# --- hints (paper Listing 4 style) ---------------------------------------------

def test_wordle_hints():
    guesses, fb = ["coral"], [score("coral", "early")]          # - - G Y Y
    assert wordle_hint("rrrrr", guesses, fb) == "'A' must be in the word"
    assert "the valid letter is 'R'" in wordle_hint("amber", guesses, fb)
    hint = wordle_hint("cxrxx", guesses, fb)
    assert hint.startswith("'C' is not a possible letter for this spot, valid letters are: ")
    assert "C" not in hint.split(": ")[1] and "O" not in hint.split(": ")[1]
    assert wordle_hint("bxrxx", guesses, fb) == "'A' must be in the word"
    assert wordle_hint("early", guesses, fb) is None


# --- solver --------------------------------------------------------------------

def test_valid_consistent_guess_is_played_first_try(cf):
    llm = ScriptedLLM("EARLY\nReason: fits")
    solver = LLMSolver(llm, candidate_filter=cf)
    obs = obs_after("early", ["coral"])
    assert solver.act(obs) == "early"
    t = solver.turns[0]
    assert t.llm_calls == 1 and t.retries == 0 and not t.problems and not t.fallback
    assert llm.calls[0][0]["role"] == "system" and "Guess: CORAL" in llm.calls[0][1]["content"]


def test_retry_on_word_not_in_list(cf):
    llm = ScriptedLLM("ZZZZZ", "CRANE")
    solver = LLMSolver(llm, candidate_filter=cf)
    assert solver.act(obs_after("early", [])) == "crane"
    t = solver.turns[0]
    assert t.problems == ["not_in_list"] and t.llm_calls == 2
    assert "'ZZZZZ' is not a valid word" in llm.calls[1][-1]["content"]
    assert llm.calls[1][-2] == {"role": "assistant", "content": "ZZZZZ"}


def test_retry_on_repeat_and_missing_word(cf):
    llm = ScriptedLLM("I cannot decide", "SLATE", "EARLY")
    solver = LLMSolver(llm, candidate_filter=cf)
    assert solver.act(obs_after("early", ["slate"])) == "early"
    assert solver.turns[0].problems == ["no_word", "repeat"]


def test_inconsistent_guess_gets_hint_then_retried(cf):
    llm = ScriptedLLM("AMBER", "EARLY")
    solver = LLMSolver(llm, candidate_filter=cf)
    assert solver.act(obs_after("early", ["coral"])) == "early"
    assert solver.turns[0].problems == ["inconsistent"]
    assert "the valid letter is 'R'" in llm.calls[1][-1]["content"]


def test_fibble_inconsistent_message_mentions_lies(cf):
    obs = obs_after("early", ["coral"], n_lies=2, max_guesses=9)
    bad = next(w for w in ["crane", "amber", "zebra", "pious", "mound"]
               if w not in cf.from_obs(obs))
    llm = ScriptedLLM(bad.upper(), cf.from_obs(obs)[0].upper())
    solver = LLMSolver(llm, candidate_filter=cf)
    solver.act(obs)
    assert "there are 2 lies in every feedback line" in llm.calls[1][-1]["content"]


def test_empty_reply_is_resent_without_adding_a_turn(cf):
    llm = ScriptedLLM(("", "length"), ("", "stop"), "EARLY")
    solver = LLMSolver(llm, candidate_filter=cf)
    assert solver.act(obs_after("early", ["coral"])) == "early"
    t = solver.turns[0]
    assert t.problems == ["truncated", "empty"] and t.llm_calls == 3 and not t.fallback
    assert all(len(c) == 2 for c in llm.calls)        # same question each time, no empty turns


def test_only_empty_replies_end_in_fallback(cf):
    llm = ScriptedLLM(("", "length"))
    solver = LLMSolver(llm, invalid_retries=2, candidate_filter=cf)
    obs = obs_after("early", ["coral"])
    assert solver.act(obs) in cf.from_obs(obs)
    assert solver.turns[0].fallback and solver.turns[0].llm_calls == 3


def test_no_retries_plays_inconsistent_guess(cf):
    llm = ScriptedLLM("AMBER")
    solver = LLMSolver(llm, retries=0, candidate_filter=cf)
    assert solver.act(obs_after("early", ["coral"])) == "amber"
    t = solver.turns[0]
    assert t.llm_calls == 1 and t.inconsistent_played


def test_out_of_retries_plays_last_valid_guess(cf):
    llm = ScriptedLLM("AMBER")                       # never fits
    solver = LLMSolver(llm, retries=2, candidate_filter=cf)
    assert solver.act(obs_after("early", ["coral"])) == "amber"
    t = solver.turns[0]
    assert t.llm_calls == 3 and t.inconsistent_played and not t.fallback


def test_fallback_when_no_playable_word(cf):
    llm = ScriptedLLM("no idea")
    solver = LLMSolver(llm, invalid_retries=2, candidate_filter=cf)
    obs = obs_after("early", ["coral"])
    word = solver.act(obs)
    t = solver.turns[0]
    assert t.fallback and t.llm_calls == 3 and word in cf.from_obs(obs)


def test_turn_prompts_do_not_carry_retries_into_next_turn(cf):
    llm = ScriptedLLM("ZZZZZ", "SLATE", "EARLY")
    solver = LLMSolver(llm, candidate_filter=cf)
    env = FibbleEnv(n_lies=0, max_guesses=6, seed=0)
    obs = env.reset(secret="early")
    obs, _, _ = env.step(solver.act(obs))
    solver.act(obs)
    assert len(llm.calls[-1]) == 2            # fresh system + user message each turn


class _PickFirstCandidate:
    """Fake model that reads the history from the prompt and answers a fitting word."""

    label = "pick"

    def __init__(self, cf, n_lies):
        self.cf, self.n_lies = cf, n_lies

    def chat(self, messages):
        text = messages[1]["content"]
        guesses, fb = [], []
        for block in text.split("\n\n")[:-1]:
            lines = block.splitlines()
            guesses.append(lines[0].split(": ")[1].lower())
            names = {"correct": G, "present": Y, "incorrect": X}
            fb.append([names[l.split(": ")[1]] for l in lines[2:]])
        words = self.cf.candidates(guesses, fb, self.n_lies)
        return LLMReply(text=f"{words[0].upper()}\nReason: fits", input_tokens=3, output_tokens=2)


def test_full_games_through_evaluate(cf):
    solver = LLMSolver(_PickFirstCandidate(cf, 0), candidate_filter=cf, keep_replies=False)
    r = evaluate(solver, secret_list(10, seed=4), 0, 6)
    assert r["games"] == 10 and r["win_rate"] >= 0.9


# --- runner --------------------------------------------------------------------

def test_runner_saves_games_and_resumes(tmp_path, cf):
    secrets = secret_list(6, seed=1)
    calls = []
    lock = threading.Lock()

    def make_llm():
        with lock:
            calls.append(1)
        return _PickFirstCandidate(cf, 1)

    out = tmp_path / "fake_Fibble1_g9.jsonl"
    row = run_setting(make_llm, "fake", secrets, 1, 9, out, workers=3, candidate_filter=cf, log=lambda m: None)
    lines = out.read_text().splitlines()
    assert len(lines) == 6 and row["games"] == 6
    rec = json.loads(lines[0])
    assert set(rec) >= {"index", "secret", "won", "guesses", "shown", "truth", "llm_calls", "turns"}
    assert sum(row["wins_by_round"]) == round(row["win_rate"] * 6)
    assert list(row)[:8] == ["game", "solver", "games", "max_guesses", "win_rate",
                             "mean_guesses", "wins_by_round", "seconds"]   # same as evaluate.py

    n_before = len(calls)
    row2 = run_setting(make_llm, "fake", secrets, 1, 9, out, workers=3, candidate_filter=cf, log=lambda m: None)
    assert len(calls) == n_before                   # nothing replayed
    assert len(out.read_text().splitlines()) == 6
    assert {k: row2[k] for k in ("win_rate", "mean_guesses", "wins_by_round")} == \
           {k: row[k] for k in ("win_rate", "mean_guesses", "wins_by_round")}


def test_runner_partial_file_continues(tmp_path, cf):
    secrets = secret_list(5, seed=2)
    out = tmp_path / "part.jsonl"
    full = run_setting(lambda: _PickFirstCandidate(cf, 0), "fake", secrets, 0, 6, out,
                       workers=1, candidate_filter=cf, log=lambda m: None)
    lines = out.read_text().splitlines()
    out.write_text("\n".join(lines[:2]) + "\n")      # pretend the run stopped after 2 games
    again = run_setting(lambda: _PickFirstCandidate(cf, 0), "fake", secrets, 0, 6, out,
                        workers=2, candidate_filter=cf, log=lambda m: None)
    assert len(load_records(out)) == 5
    assert again["win_rate"] == full["win_rate"] and again["wins_by_round"] == full["wins_by_round"]


def test_parallel_matches_sequential(tmp_path, cf):
    secrets = secret_list(8, seed=3)
    seq = run_setting(lambda: _PickFirstCandidate(cf, 2), "fake", secrets, 2, 9, tmp_path / "seq.jsonl",
                      workers=1, candidate_filter=cf, log=lambda m: None)
    par = run_setting(lambda: _PickFirstCandidate(cf, 2), "fake", secrets, 2, 9, tmp_path / "par.jsonl",
                      workers=4, candidate_filter=cf, log=lambda m: None)
    by_index = lambda p: {i: (r["guesses"], r["shown"]) for i, r in load_records(p).items()}
    assert by_index(tmp_path / "seq.jsonl") == by_index(tmp_path / "par.jsonl")
    for k in ("win_rate", "mean_guesses", "wins_by_round", "llm_calls"):
        assert seq[k] == par[k]


def test_runner_matches_evaluate_on_same_secrets(tmp_path, cf):
    """Same secrets and seeds as evaluation.evaluate -> same games for a deterministic solver."""
    secrets = secret_list(6, seed=5)
    row = run_setting(lambda: _PickFirstCandidate(cf, 1), "fake", secrets, 1, 9, tmp_path / "x.jsonl",
                      workers=3, candidate_filter=cf, log=lambda m: None)
    ref = evaluate(LLMSolver(_PickFirstCandidate(cf, 1), candidate_filter=cf), secrets, 1, 9)
    for k in ("win_rate", "mean_guesses", "wins_by_round"):
        assert row[k] == ref[k]


class _APIError(Exception):
    def __init__(self, status_code):
        super().__init__(f"Error code: {status_code}")
        self.status_code = status_code


class _FailingLLM(_PickFirstCandidate):
    """Plays normally, but raises for chosen secrets (seen through the prompt count)."""

    def __init__(self, cf, n_lies, error):
        super().__init__(cf, n_lies)
        self.error = error

    def chat(self, messages):
        if self.error is not None:
            raise self.error
        return super().chat(messages)


def test_out_of_credit_stops_run_and_keeps_saved_games(tmp_path, cf):
    secrets = secret_list(8, seed=6)
    out = tmp_path / "credit.jsonl"
    count = iter(range(100))
    lock = threading.Lock()

    def make_llm():
        with lock:
            k = next(count)
        return _FailingLLM(cf, 0, _APIError(402) if k >= 3 else None)   # credit runs out after 3 games

    with pytest.raises(RunStopped, match="out of credit"):
        run_setting(make_llm, "fake", secrets, 0, 6, out, workers=1, candidate_filter=cf, log=lambda m: None)
    saved = load_records(out)
    assert len(saved) == 3
    # Credit is back: the next run plays only the missing games.
    row = run_setting(lambda: _PickFirstCandidate(cf, 0), "fake", secrets, 0, 6, out,
                      workers=2, candidate_filter=cf, log=lambda m: None)
    assert row["games"] == 8 and len(load_records(out)) == 8


def test_other_errors_skip_only_that_game(tmp_path, cf):
    secrets = secret_list(6, seed=7)
    out = tmp_path / "flaky.jsonl"
    count = iter(range(100))
    lock = threading.Lock()
    messages = []

    def make_llm():
        with lock:
            k = next(count)
        return _FailingLLM(cf, 0, ValueError("boom") if k == 2 else None)

    row = run_setting(make_llm, "fake", secrets, 0, 6, out, workers=1, candidate_filter=cf, log=messages.append)
    assert row["games"] == 5 and any("will be played on the next run" in m for m in messages)
    row = run_setting(lambda: _PickFirstCandidate(cf, 0), "fake", secrets, 0, 6, out,
                      workers=1, candidate_filter=cf, log=lambda m: None)
    assert row["games"] == 6
