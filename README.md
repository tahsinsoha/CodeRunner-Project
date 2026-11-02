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

## Candidate filter

Keeps the words that could still be the secret. A word fits a feedback line when its true feedback differs from the shown feedback in exactly `n_lies` tiles; already-guessed words are excluded.

```python
from wordle_fibble import CandidateFilter, is_consistent, load_allowed

cf = CandidateFilter()                    # possible secrets = 2,315 answers
cf = CandidateFilter(load_allowed())      # or all 12,972 allowed words
words = cf.from_obs(obs)                  # list of remaining candidates
mask = cf.mask(obs["guesses"], obs["feedback"], obs["n_lies"])  # bool array over cf.words

is_consistent("relay", obs["guesses"], obs["feedback"], obs["n_lies"])  # one word, e.g. for LLM retries
```

## Baseline solvers

Solvers implement `act(obs) -> word` (see `wordle_fibble.solvers.Solver`).

- `RandomSolver`: random word among the remaining candidates.
- `RuleBasedSolver`: remaining candidate with the most common letters (by position and by presence among candidates); deterministic.

```bash
python evaluate.py                                        # both baselines, Wordle + Fibble1-5, 500 games, 6 guesses
python evaluate.py --guesses 9 --candidates allowed --csv results/baselines.csv
```

`wordle_fibble.evaluation.secret_list(n, seed)` gives the fixed list of secrets so every solver is tested on the same words; `evaluate(...)` reports win rate, mean guesses (a loss counts as the guess limit, as in the paper) and wins by round.

## Playing in the terminal

```bash
python play.py                           # Wordle, you type guesses
python play.py --lies 1 --guesses 9      # Fibble
python play.py --lies 3 --auto --games 1000   # random-guess bot win rate
```
