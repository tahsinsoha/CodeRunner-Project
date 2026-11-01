# RL and LLM Solvers for Wordle/Fibble

CMPM 244 project reimplementing *RL- and LLM-based AI Solvers for the Game of Wordle/Fibble* (Chusap et al., IEEE CoG 2025).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
```

## Game environment

`wordle_fibble.FibbleEnv` plays Wordle (`n_lies=0`) and Fibble1-5 (`n_lies=1..5`). Every feedback line has exactly `n_lies` tiles switched to a wrong color; a winning guess ends the game and is shown truthfully.

```python
from wordle_fibble import FibbleEnv

env = FibbleEnv(n_lies=1, max_guesses=9, seed=0)
obs = env.reset(secret="early")       # omit secret for a random answer
obs, done, info = env.step("crane")
print(env.render(show_truth=True))
```

- Colors: `0` gray, `1` yellow, `2` green.
- `obs` holds what a solver may see: `guesses`, `feedback` (shown colors), `guesses_left`, `n_lies`, and arrays `board` / `letters` (`max_guesses x 5`, `-1` if unplayed) and `alphabet` (26, last color shown per letter).
- `info` holds `won`, `true_feedback` and `lie_positions` for analysis only; solvers must not use them.
- `step` raises `ValueError` for words outside the allowed list without using up a turn.
- The environment gives no reward; each solver computes its own.
- `initial_random_guess=True` starts the game with a random opener, as in original Fibble.

Word lists in `data/`: `answers.txt` (2,315 secrets) and `allowed.txt` (12,972 guesses, includes the answers).

## Playing in the terminal

```bash
python play.py                           # Wordle, you type guesses
python play.py --lies 1 --guesses 9      # Fibble
python play.py --lies 3 --auto --games 1000   # random-guess bot win rate
```
