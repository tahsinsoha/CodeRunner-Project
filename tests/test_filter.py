import numpy as np
import pytest

from wordle_fibble import (
    GRAY,
    GREEN,
    YELLOW,
    CandidateFilter,
    FibbleEnv,
    RandomSolver,
    RuleBasedSolver,
    encode,
    encode_many,
    is_consistent,
    load_allowed,
    load_answers,
    score,
    score_many,
)
from wordle_fibble.evaluation import evaluate, secret_list

G, Y, X = GREEN, YELLOW, GRAY


@pytest.fixture(scope="module")
def cf():
    return CandidateFilter()


def test_score_many_matches_score():
    rng = np.random.default_rng(0)
    allowed, answers = load_allowed(), load_answers()
    secrets = [answers[i] for i in rng.choice(len(answers), 300)]
    tricky = ["aabbb", "bbaaa", "eerie", "there", "llama", "hello", "speed", "abide"]
    for guess in [allowed[i] for i in rng.choice(len(allowed), 50)] + tricky:
        batch = score_many(encode(guess), encode_many(secrets + tricky))
        expected = np.stack([score(guess, s) for s in secrets + tricky])
        assert (batch == expected).all(), guess


def test_wordle_filter_example(cf):
    words = cf.candidates(["coral"], [[X, X, G, Y, Y]], n_lies=0)
    assert "early" in words
    assert all(w[2] == "r" and "a" in w and "l" in w and "c" not in w for w in words)


def test_guessed_words_are_excluded(cf):
    # With 5 lies, all-gray shown for 'crane' is consistent with 'crane' by
    # mismatch count, but the game would have ended had it been the secret.
    assert "crane" not in cf.candidates(["crane"], [[X, X, X, X, X]], n_lies=5)
    assert not is_consistent("crane", ["crane"], [[X, X, X, X, X]], 5)


@pytest.mark.parametrize("n_lies", range(6))
def test_secret_always_survives_filter(cf, n_lies):
    rng = np.random.default_rng(n_lies)
    allowed = load_allowed()
    env = FibbleEnv(n_lies=n_lies, max_guesses=6, seed=n_lies)
    for _ in range(40):
        obs = env.reset()
        while not env.done:
            obs, _, _ = env.step(allowed[rng.integers(len(allowed))])
            if not env.won:
                assert env.secret in cf.from_obs(obs)


@pytest.mark.parametrize("n_lies", [0, 1, 3])
def test_mask_agrees_with_is_consistent(cf, n_lies):
    env = FibbleEnv(n_lies=n_lies, max_guesses=6, seed=10 + n_lies)
    obs = env.reset(secret="early")
    for g in ["slate", "pound"]:
        obs, _, _ = env.step(g)
    mask = cf.mask(obs["guesses"], obs["feedback"], n_lies)
    rng = np.random.default_rng(0)
    for i in rng.choice(len(cf.words), 300, replace=False):
        assert mask[i] == is_consistent(cf.words[i], obs["guesses"], obs["feedback"], n_lies)


def test_filter_narrows_wordle_fast(cf):
    env = FibbleEnv(n_lies=0, seed=0)
    obs = env.reset(secret="early")
    obs, _, _ = env.step("slate")
    assert len(cf.from_obs(obs)) < 150


@pytest.mark.parametrize("solver_cls", [RandomSolver, RuleBasedSolver])
@pytest.mark.parametrize("n_lies", [0, 1, 5])
def test_solvers_play_valid_games(cf, solver_cls, n_lies):
    r = evaluate(solver_cls(cf, seed=0), secret_list(20, seed=1), n_lies, max_guesses=9)
    assert r["games"] == 20
    assert sum(r["wins_by_round"]) == round(r["win_rate"] * 20)


def test_baselines_solve_wordle(cf):
    secrets = secret_list(100, seed=2)
    assert evaluate(RuleBasedSolver(cf), secrets, 0, 6)["win_rate"] > 0.9
    assert evaluate(RandomSolver(cf, seed=0), secrets, 0, 6)["win_rate"] > 0.8


def test_rule_based_is_deterministic(cf):
    secrets = secret_list(10, seed=3)
    a = evaluate(RuleBasedSolver(cf), secrets, 1, 9, seed=5)
    b = evaluate(RuleBasedSolver(cf), secrets, 1, 9, seed=5)
    assert a == b
