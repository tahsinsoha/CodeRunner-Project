import numpy as np
import pytest

torch = pytest.importorskip("torch")

from wordle_fibble import FibbleEnv, is_consistent
from wordle_fibble.dqn import (
    OBS_DIM,
    ActionSpace,
    DQNSolver,
    QNet,
    encode_obs,
    reward,
    save,
    train,
    word_values,
)
from wordle_fibble.evaluation import evaluate

WORDS = ["crane", "slate", "pious", "fjord", "crate", "trace", "react", "caret"]


def test_encode_obs_counts_colors():
    env = FibbleEnv(answers=WORDS, allowed=WORDS, seed=0)
    s = encode_obs(env.reset(secret="crate"))
    assert s.shape == (OBS_DIM,) and s[0] == pytest.approx(0.6) and s[2:].sum() == 0
    s = encode_obs(env.step("crane")[0])
    per_pos = s[2 + 52:].reshape(26, 5, 2) * 10
    # C, R, A, E green in place; N gray at position 3.
    assert per_pos[2, 0, 0] == 1 and per_pos[17, 1, 0] == 1 and per_pos[0, 2, 0] == 1
    assert per_pos[4, 4, 0] == 1 and per_pos[13, 3, 1] == 1 and per_pos[:, :, 1].sum() == 1
    per_letter = s[2:2 + 52].reshape(26, 2) * 10
    assert per_letter[13].tolist() == [0, 1] and per_letter[4].tolist() == [1, 0]


def test_word_value_is_sum_of_letter_outputs():
    space = ActionSpace(WORDS)
    out = torch.randn(3, 130)
    q = word_values(out, space.flat_codes)
    w = "crane"
    expected = sum(out[:, i * 26 + ord(c) - 97] for i, c in enumerate(w))
    assert torch.allclose(q[:, space.index[w]], expected)


def test_masks():
    env = FibbleEnv(answers=WORDS, allowed=WORDS, seed=0)
    obs = env.reset(secret="crate")
    obs, _, _ = env.step("crane")
    cands = ActionSpace(WORDS, mask="candidates").mask(obs)
    assert [w for w, m in zip(WORDS, cands) if m] == ["crate"]
    unguessed = ActionSpace(WORDS, mask="unguessed").mask(obs)
    assert unguessed.sum() == len(WORDS) - 1 and not unguessed[0]


def test_reward():
    assert reward(won=False, done=False) == -1
    assert reward(won=True, done=True) == 9
    assert reward(won=False, done=True) == -11


def test_train_save_load_and_play(tmp_path):
    net, history = train(episodes=60, learn_start=50, batch=32, log_every=30, log=lambda _: None)
    assert len(history) == 2
    path = tmp_path / "dqn.pt"
    save(net, path)
    solver = DQNSolver.load(path)
    assert solver.config["mask"] == "candidates"
    env = FibbleEnv(n_lies=1, max_guesses=9, seed=0)
    obs = env.reset(secret="fjord")
    while not env.done:
        guess = solver.act(obs)
        assert is_consistent(guess, obs["guesses"], obs["feedback"], obs["n_lies"])
        obs, _, _ = env.step(guess)
    r = evaluate(solver, ["crate", "pious", "fjord"], n_lies=0, max_guesses=6)
    assert r["games"] == 3 and r["solver"] == "dqn"


def test_untrained_net_shapes():
    out = QNet()(torch.zeros(4, OBS_DIM))
    assert out.shape == (4, 130) and np.isfinite(out.detach().numpy()).all()
