import numpy as np

from .env import FibbleEnv
from .words import load_answers


def secret_list(n_games, seed=0, answers=None):
    """A fixed list of secrets, so every solver is tested on the same words."""
    answers = answers if answers is not None else load_answers()
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(answers), size=n_games, replace=n_games > len(answers))
    return [answers[i] for i in picks]


def evaluate(solver, secrets, n_lies, max_guesses, initial_random_guess=False, seed=0):
    """Play one game per secret. Mean guesses counts a loss as max_guesses,
    matching the paper's tables."""
    env = FibbleEnv(n_lies=n_lies, max_guesses=max_guesses,
                    initial_random_guess=initial_random_guess)
    wins, used = 0, []
    win_rounds = np.zeros(max_guesses + 1, dtype=int)
    for game, secret in enumerate(secrets):
        obs = env.reset(secret=secret, seed=seed + game)
        solver.reset()
        while not env.done:
            obs, _, _ = env.step(solver.act(obs))
        wins += env.won
        used.append(len(env.guesses))
        if env.won:
            win_rounds[len(env.guesses)] += 1
    return {
        "game": env.name,
        "solver": solver.name,
        "games": len(secrets),
        "max_guesses": max_guesses,
        "win_rate": wins / len(secrets),
        "mean_guesses": float(np.mean(used)),
        "wins_by_round": win_rounds[1:].tolist(),
    }
