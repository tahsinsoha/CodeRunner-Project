# RL and LLM Solvers for Wordle/Fibble

CMPM 244 project reimplementing the paper *RL- and LLM-based AI Solvers for the Game of Wordle/Fibble* (Chusap et al., IEEE CoG 2025).

## Setup

```bash
pip install -r requirements.txt
python -m pytest -q        # checks that everything works
python play.py             # play Wordle in the terminal
```

## What each file does

### `data/`
- **`answers.txt`**: the 2,315 words that can be the secret word.
- **`allowed.txt`**: the 12,972 words the player is allowed to guess. It includes all the answers.

### `wordle_fibble/` (the main code)
- **`words.py`**: reads the word lists from `data/`. It also turns words into numbers (a=0, b=1, ... z=25) so the code can work with them quickly.
- **`feedback.py`**: works out the colors for a guess (green, yellow or gray). It also adds the lies for Fibble by switching some tiles to a wrong color.
- **`env.py`**: the game itself. It picks a secret word, takes guesses, shows the colors (with lies if it's Fibble), and tracks how many guesses are left and whether you won.
- **`filter.py`**: finds the words that could still be the secret, based on the guesses and colors so far and the number of lies.
- **`solvers.py`**: two simple players to compare our AI against. One guesses a random word that still fits the colors. The other guesses the fitting word that uses the most common letters.
- **`letter_probs.py`**: builds the 5 x 26 table of how often each letter appears at each position (the paper's Table I), from the 12,972 allowed words. Saved in `data/letter_probs.csv`.
- **`llm_client.py`**: connects to LLMs through an OpenAI-compatible API: the hosted DeepSeek API (`deepseek-flash`, `deepseek-v4-pro`), our own Ollama server on NRP (an open-weight model such as `gpt-oss:20b`), or the NRP managed LLMs as a backup. Can switch reasoning on or off, and retries when the service is busy.
- **`evaluation.py`**: plays many games with a player and reports how often it wins and how many guesses it needs on average.
- **`__init__.py`**: lets you import everything with `from wordle_fibble import ...`.

### Scripts
- **`play.py`**: play the game yourself in the terminal, or watch a random bot play.
- **`check_llm.py`**: checks the LLM setup: the key works, the model is served, and one Wordle-style reply from each model.
- **`evaluate.py`**: runs the two simple players on Wordle and Fibble1–5 and prints their win rates. It can also save the results to a CSV file.

### Other
- **`tests/`**: automatic checks that the game, colors, lies and filter work correctly. Run them with `python -m pytest -q`.
- **`results/`**: saved win rates of the two simple players.
- **`requirements.txt`**: the Python packages needed.
- **`nrp/`**: files for running on the NRP cluster (storage, a CPU dev pod, an Ollama GPU server for the open-weight model).
- **`.env`** (not in git): API keys, e.g. `DEEPSEEK_API_KEY=sk-...`. On NRP it lives in `/work/.env`.
