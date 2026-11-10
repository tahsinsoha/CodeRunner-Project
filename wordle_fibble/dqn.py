"""Wordle RL solver (task 4): a DQN written in PyTorch.

The network outputs 5 x 26 = 130 values, one per (position, letter). The
value of a word is the sum of its five (position, letter) outputs, so the
agent can choose among thousands of words with a 130-unit output layer.

Observation vector (see `encode_obs`):
    1      guesses left / 10
    1      lies per line / 5
    26 x 2 per letter: times shown present (yellow or green), times shown gray
    26 x 5 x 2 coloring array: times shown green at this position, times shown
           yellow or gray at this position (letter not here)
Counts are divided by 10, so repeated evidence adds up, which matters when
some of the feedback is lies.

`mask` controls which words the agent may pick:
    "candidates"  only words that still fit the feedback (uses CandidateFilter)
    "unguessed"   any word not guessed yet; the network must learn the rules
"""

import time

import numpy as np
import torch
from torch import nn

from .env import FibbleEnv
from .feedback import GRAY, GREEN
from .filter import CandidateFilter
from .solvers import Solver
from .words import ALPHABET, WORD_LEN, encode_many, load_allowed, load_answers

N_LETTERS = len(ALPHABET)
N_OUT = WORD_LEN * N_LETTERS
OBS_DIM = 2 + N_LETTERS * 2 + N_LETTERS * WORD_LEN * 2
COUNT_SCALE = 10.0
MASKS = ("candidates", "unguessed")
WORD_LISTS = {"answers": load_answers, "allowed": load_allowed}


def encode_obs(obs):
    """Environment observation -> float32 vector of length OBS_DIM."""
    letters, board = obs["letters"], obs["board"]
    rows, cols = np.nonzero(letters >= 0)
    lets, colors = letters[rows, cols].astype(np.int64), board[rows, cols]
    per_letter = np.zeros((N_LETTERS, 2), dtype=np.float32)
    np.add.at(per_letter, (lets, (colors == GRAY).astype(np.int64)), 1)
    per_pos = np.zeros((N_LETTERS, WORD_LEN, 2), dtype=np.float32)
    np.add.at(per_pos, (lets, cols, (colors != GREEN).astype(np.int64)), 1)
    head = np.array([obs["guesses_left"] / COUNT_SCALE, obs["n_lies"] / WORD_LEN], dtype=np.float32)
    return np.concatenate([head, per_letter.ravel() / COUNT_SCALE, per_pos.ravel() / COUNT_SCALE])


class QNet(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(OBS_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, N_OUT),
        )

    def forward(self, x):
        """(B, OBS_DIM) -> (B, 130) letter-position values."""
        return self.body(x)


def word_values(out, flat_codes):
    """(B, 130) outputs and (N, 5) flat indices -> (B, N) word values."""
    return out[:, flat_codes].sum(dim=-1)


class ActionSpace:
    """The words the agent can guess, plus the masking rule."""

    def __init__(self, words, mask="candidates"):
        if mask not in MASKS:
            raise ValueError(f"mask must be one of {MASKS}, got {mask!r}")
        self.words = list(words)
        self.mask_kind = mask
        self.filter = CandidateFilter(self.words)
        self.index = self.filter.index
        codes = encode_many(self.words).astype(np.int64)
        self.flat_codes = torch.from_numpy(codes + np.arange(WORD_LEN) * N_LETTERS)

    def __len__(self):
        return len(self.words)

    def mask(self, obs):
        if self.mask_kind == "candidates":
            m = self.filter.mask(obs["guesses"], obs["feedback"], obs["n_lies"])
            if m.any():
                return m
        m = np.ones(len(self.words), dtype=bool)
        for g in obs["guesses"]:
            if g in self.index:
                m[self.index[g]] = False
        return m


class ReplayBuffer:
    def __init__(self, capacity, n_actions, rng):
        self.capacity, self.n_actions, self.rng = capacity, n_actions, rng
        self.s = np.zeros((capacity, OBS_DIM), dtype=np.float32)
        self.s2 = np.zeros((capacity, OBS_DIM), dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int64)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.mask2 = np.zeros((capacity, (n_actions + 7) // 8), dtype=np.uint8)
        self.size = self.pos = 0

    def add(self, s, a, r, s2, done, mask2):
        i = self.pos
        self.s[i], self.a[i], self.r[i], self.s2[i], self.done[i] = s, a, r, s2, done
        self.mask2[i] = np.packbits(mask2)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch):
        i = self.rng.integers(self.size, size=batch)
        mask2 = np.unpackbits(self.mask2[i], axis=1, count=self.n_actions).astype(bool)
        return (torch.from_numpy(self.s[i]), torch.from_numpy(self.a[i]),
                torch.from_numpy(self.r[i]), torch.from_numpy(self.s2[i]),
                torch.from_numpy(self.done[i]), torch.from_numpy(mask2))


def reward(won, done):
    """-1 per guess, +10 for a win, -10 for a loss."""
    return -1.0 + (10.0 if won else -10.0 if done else 0.0)


def greedy(net, space, s, mask):
    with torch.no_grad():
        q = word_values(net(torch.from_numpy(s)[None]), space.flat_codes)[0].numpy()
    q[~mask] = -np.inf
    return int(np.argmax(q))


def train(
    n_lies=0,
    max_guesses=6,
    words="answers",
    secrets=None,
    mask="candidates",
    episodes=20000,
    hidden=256,
    lr=5e-4,
    gamma=0.99,
    batch=256,
    buffer_size=100_000,
    learn_start=2000,
    target_every=1000,
    eps_start=1.0,
    eps_end=0.05,
    eps_episodes=None,
    seed=0,
    log_every=1000,
    log=print,
):
    """Train a DQN; returns (net, history). `secrets` defaults to all answers."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    space = ActionSpace(WORD_LISTS[words](), mask=mask)
    secrets = list(secrets) if secrets is not None else load_answers()
    env = FibbleEnv(n_lies=n_lies, max_guesses=max_guesses, allowed=space.words, seed=seed)
    net, target = QNet(hidden), QNet(hidden)
    target.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    buf = ReplayBuffer(buffer_size, len(space), rng)
    eps_episodes = eps_episodes or episodes // 2

    steps, history, recent, start = 0, [], [], time.perf_counter()
    for ep in range(episodes):
        eps = max(eps_end, eps_start - (eps_start - eps_end) * ep / eps_episodes)
        obs = env.reset(secret=secrets[rng.integers(len(secrets))])
        s, m = encode_obs(obs), space.mask(obs)
        while not env.done:
            if rng.random() < eps:
                a = int(rng.choice(np.flatnonzero(m)))
            else:
                a = greedy(net, space, s, m)
            obs, done, info = env.step(space.words[a])
            s2 = encode_obs(obs)
            m2 = space.mask(obs) if not done else np.zeros(len(space), dtype=bool)
            buf.add(s, a, reward(info["won"], done), s2, float(done), m2)
            s, m = s2, m2
            steps += 1

            if buf.size >= learn_start:
                bs, ba, br, bs2, bd, bm2 = buf.sample(batch)
                q_sa = net(bs).gather(1, space.flat_codes[ba]).sum(dim=1)
                with torch.no_grad():
                    # Double DQN: the online net picks the next word, the target net values it.
                    q_next = word_values(net(bs2), space.flat_codes).masked_fill(~bm2, -1e9)
                    a_next = q_next.argmax(dim=1)
                    v_next = target(bs2).gather(1, space.flat_codes[a_next]).sum(dim=1)
                    y = br + gamma * (1 - bd) * v_next
                loss = nn.functional.smooth_l1_loss(q_sa, y)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 10.0)
                opt.step()
                if steps % target_every == 0:
                    target.load_state_dict(net.state_dict())

        recent.append((env.won, len(env.guesses)))
        if (ep + 1) % log_every == 0:
            win = np.mean([w for w, _ in recent])
            mean_g = np.mean([g for _, g in recent])
            history.append({"episode": ep + 1, "epsilon": round(eps, 3), "win_rate": win,
                            "mean_guesses": mean_g, "steps": steps,
                            "seconds": round(time.perf_counter() - start, 1)})
            log(f"ep {ep + 1:>7}  eps {eps:.2f}  train win {win:6.1%}  mean {mean_g:.2f}  "
                f"{time.perf_counter() - start:6.0f}s")
            recent = []

    net.config = {"hidden": hidden, "words": words, "mask": mask, "n_lies": n_lies,
                  "max_guesses": max_guesses, "episodes": episodes, "seed": seed}
    return net, history


def save(net, path):
    torch.save({"state_dict": net.state_dict(), "config": net.config}, path)


class DQNSolver(Solver):
    """Plays greedily with a trained network."""

    name = "dqn"

    def __init__(self, net, words="answers", mask="candidates"):
        self.net = net.eval()
        self.space = ActionSpace(WORD_LISTS[words](), mask=mask)

    @classmethod
    def load(cls, path, mask=None):
        ckpt = torch.load(path, weights_only=False)
        cfg = ckpt["config"]
        net = QNet(cfg["hidden"])
        net.load_state_dict(ckpt["state_dict"])
        solver = cls(net, words=cfg["words"], mask=mask or cfg["mask"])
        solver.config = cfg
        return solver

    def act(self, obs):
        m = self.space.mask(obs)
        return self.space.words[greedy(self.net, self.space, encode_obs(obs), m)]
