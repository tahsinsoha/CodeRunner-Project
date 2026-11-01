import numpy as np
import pytest

from wordle_fibble import GRAY, GREEN, YELLOW, FibbleEnv, apply_lies, load_allowed, load_answers, score

G, Y, X = GREEN, YELLOW, GRAY


def test_word_lists_match_paper():
    answers, allowed = load_answers(), load_allowed()
    assert len(answers) == 2315
    assert len(allowed) == 12972
    assert set(answers) <= set(allowed)


@pytest.mark.parametrize(
    "guess, secret, expected",
    [
        ("crane", "crane", [G, G, G, G, G]),
        ("slate", "crane", [X, X, G, X, G]),
        ("coral", "early", [X, X, G, Y, Y]),
        ("slate", "early", [X, Y, Y, X, Y]),
        # Repeated letters: only as many marks as the secret has copies.
        ("speed", "abide", [X, X, Y, X, Y]),
        ("eerie", "there", [Y, X, Y, X, G]),
        ("llama", "hello", [Y, Y, X, X, X]),
        ("aabbb", "bbaaa", [Y, Y, Y, Y, X]),
    ],
)
def test_score(guess, secret, expected):
    assert score(guess, secret).tolist() == expected


@pytest.mark.parametrize("n_lies", range(6))
def test_apply_lies_changes_exactly_n_tiles(n_lies):
    rng = np.random.default_rng(0)
    for _ in range(200):
        truth = rng.integers(0, 3, size=5).astype(np.int8)
        shown, lies = apply_lies(truth, n_lies, rng)
        assert len(lies) == n_lies
        assert (shown != truth).sum() == n_lies
        assert set(np.flatnonzero(shown != truth)) == set(lies)


def test_wordle_has_no_lies():
    env = FibbleEnv(n_lies=0, seed=1)
    env.reset(secret="crane")
    _, _, info = env.step("slate")
    assert (env.shown_history[0] == info["true_feedback"]).all()


@pytest.mark.parametrize("n_lies", range(1, 6))
def test_fibble_lies_every_line(n_lies):
    env = FibbleEnv(n_lies=n_lies, max_guesses=9, seed=2)
    env.reset(secret="early")
    for guess in ["slate", "coral", "pious", "mound"]:
        obs, _, info = env.step(guess)
        assert (obs["feedback"][-1] != info["true_feedback"]).sum() == n_lies


def test_win_ends_game_and_is_shown_truthfully():
    env = FibbleEnv(n_lies=3, max_guesses=9, seed=3)
    env.reset(secret="early")
    _, done, info = env.step("early")
    assert done and env.won and info["won"]
    assert env.shown_history[-1].tolist() == [G] * 5


@pytest.mark.parametrize("limit", [6, 8, 9])
def test_guess_limit(limit):
    env = FibbleEnv(n_lies=1, max_guesses=limit, seed=4)
    env.reset(secret="early")
    for i in range(limit):
        obs, done, _ = env.step("slate")
        assert obs["guesses_left"] == limit - i - 1
    assert done and not env.won
    with pytest.raises(RuntimeError):
        env.step("crane")


def test_invalid_guess_does_not_use_turn():
    env = FibbleEnv(seed=5)
    env.reset(secret="early")
    with pytest.raises(ValueError):
        env.step("zzzzz")
    with pytest.raises(ValueError):
        env.step("toolong")
    assert env.guesses_left == 6


def test_initial_random_guess_uses_one_turn():
    env = FibbleEnv(n_lies=1, max_guesses=9, initial_random_guess=True, seed=6)
    obs = env.reset(secret="early")
    assert len(obs["guesses"]) == 1 and obs["guesses"][0] != "early"
    assert obs["guesses_left"] == 8


def test_observation_arrays():
    env = FibbleEnv(n_lies=0, max_guesses=9, seed=7)
    env.reset(secret="early")
    obs, _, _ = env.step("coral")
    obs, _, _ = env.step("slate")
    assert obs["board"].shape == (9, 5)
    assert obs["board"][0].tolist() == [X, X, G, Y, Y]
    assert (obs["board"][2:] == -1).all()
    assert obs["letters"][1].tolist() == [18, 11, 0, 19, 4]  # s l a t e
    assert obs["alphabet"][ord("c") - 97] == X
    assert obs["alphabet"][ord("e") - 97] == Y
    assert obs["alphabet"][ord("z") - 97] == -1


def test_same_seed_same_game():
    def run(seed):
        env = FibbleEnv(n_lies=2, max_guesses=9, seed=seed)
        env.reset()
        for g in ["slate", "coral", "pious"]:
            env.step(g)
        return env.secret, [c.tolist() for c in env.shown_history]

    assert run(11) == run(11)
