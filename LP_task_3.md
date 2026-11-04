# Task 3: Data preparation and LLM setup (Lakshmi)

This note covers what I did for task 3 in our proposal, where the code is, what we found, and what is left for the next tasks. Everything here is on `main` as of commit `3374986`.

Task 3 in the proposal has three parts:

1. Answer and guess word lists
2. The 5 x 26 letter-by-position probability table (the paper's Table I)
3. LLM API setup for the LLM solver (task 6), using current models instead of the paper's GPT-4, GPT-4o and o1-preview

All three are done, except one check on the open-weight model (see section 3.4).

---

## 1. Word lists

The lists were already in the repo from task 1, so no new files were needed here.

| File | Words | Use |
|---|---|---|
| `data/answers.txt` | 2,315 | Possible secret words |
| `data/allowed.txt` | 12,972 | Allowed guesses (includes all the answers) |

Both match the counts in the paper (Section III), and `tests/test_env.py` checks this.

The paper's targeted Fibble solver (Section IV-A) builds its table from a different list of 5,757 words, which we do not have. Section 2 explains how we handled that.

---

## 2. Letter-by-position probability table

**Code:** `wordle_fibble/letter_probs.py`
**Saved table:** `data/letter_probs.csv`
**Tests:** `tests/test_letter_probs.py`

### What it is

For each of the 5 positions, the share of words that have each letter there. Each position sums to 1. This is the table the PPO solver (task 5) rescales with its green / yellow / gray weights.

```python
from wordle_fibble.letter_probs import letter_position_probs, load_csv

probs = letter_position_probs()   # numpy array, shape (5, 26): probs[position, letter]
probs = load_csv()                # same table, read from data/letter_probs.csv
probs[4, ord("s") - 97]           # chance of "s" in position 5 -> 0.3051
```

Note the shape: rows are positions and columns are letters (a = 0 ... z = 25). The CSV uses the paper's layout instead (one row per letter, one column per position), so it can go straight into the report.

To rebuild the CSV:

```bash
python -m wordle_fibble.letter_probs
```

### Which word list we used, and why

The paper's 5,757-word list is not available, so we built the table from both of our lists and compared each to the paper's Table I:

| Word list | Words | Mean difference per cell | Largest difference | "S" in position 5 |
|---|---|---|---|---|
| Paper's Table I | 5,757 | - | - | 0.3064 |
| `allowed.txt` | 12,972 | **0.0042** | **0.0339** | 0.3051 |
| `answers.txt` | 2,315 | 0.0120 | 0.2908 | 0.0156 |

The allowed list is very close to the paper. The answer list is not, mainly because it has almost no plurals (S at the end is 30% in the paper and only 1.6% in the answers). So the table is built from `allowed.txt`. This is also the list the solver guesses from, so it makes sense.

The most likely letter per position is S, A, A, E, S. The tests check a few cells against Table I (within 0.035), check that each position sums to 1, and check the CSV round trip.

**For the report:** we should say that we used the 12,972 allowed words because the paper's 5,757-word list is not available, and give the difference numbers above.

---

## 3. LLM setup

**Code:** `wordle_fibble/llm_client.py`
**Setup check script:** `check_llm.py`
**Tests:** `tests/test_llm_client.py` (offline; no key or network needed)

### 3.1 Models

The proposal says to test current models instead of the paper's, with one hosted model through an API and one open-weight model.

| Type | Provider in the code | Models | Notes |
|---|---|---|---|
| Hosted | `deepseek` | `deepseek-flash` (DeepSeek V4.1 Flash), `deepseek-v4-pro` | Paid API. Thinking (reasoning) can be turned on or off per request |
| Open-weight | `ollama` | `gpt-oss:20b` (default) | OpenAI's open-weight reasoning model, run through an Ollama server |
| Other | `custom` | any | Any other OpenAI-compatible server |

None of these were in the paper. The thinking switch is useful for our comparison: the paper's main LLM result is that the reasoning model (o1-preview, 95.74% on Wordle) did far better than GPT-4o (21.27%). With DeepSeek we can run the same model with and without reasoning.

### 3.2 How to use the client

```python
from wordle_fibble.llm_client import LLMClient

llm = LLMClient("deepseek", "deepseek-flash", thinking=False)
reply = llm.chat([
    {"role": "system", "content": "...rules..."},
    {"role": "user", "content": "No feedback yet. You have 6 tries remaining."},
])
reply.text            # the answer
reply.reasoning       # the model's reasoning text, if thinking is on
reply.input_tokens, reply.output_tokens, reply.seconds, reply.attempts
```

Options:

- `thinking=None / True / False`: provider default, reasoning on, reasoning off (DeepSeek and `custom`; not available for Ollama)
- `reasoning_effort="low" / "high" / "max"`: DeepSeek thinking mode only
- `temperature`: has no effect in DeepSeek thinking mode
- `max_tokens` (default 4096), `timeout` (default 300 s), `max_retries` (default 8)

When the service is busy (HTTP 429), has a server error (5xx), or times out, the client waits and tries again: 2, 4, 8, ... seconds, up to 120 s each time. Errors like a wrong key (401) or a bad request (400) fail at once.

### 3.3 Keys and settings

Keys are never in git. The client reads them from environment variables or from a `.env` file:

```
DEEPSEEK_API_KEY=sk-...
```

It looks in the repo's `.env` first (already in `.gitignore`), then in the file named by `LLM_ENV_FILE` if that is set.

| Setting | Used for | Default |
|---|---|---|
| `DEEPSEEK_API_KEY` | DeepSeek | none (required) |
| `OLLAMA_BASE_URL` | Address of the Ollama server | `http://localhost:11434/v1` |
| `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` | `custom` provider | none |
| `LLM_ENV_FILE` | A `.env` file outside the repo | none |

`requirements.txt` already lists `openai` (the client library) and `ollama`.

### 3.4 Setup check and results

`check_llm.py` lists the models a provider serves, sends each one the paper's rules prompt (Appendix A, Listing 1) with "No feedback yet. You have 6 tries remaining.", and prints the reply, the first 5-letter word in it, the time and the tokens. It also adds a row to `results/llm_check.csv`.

```bash
python check_llm.py --provider deepseek
python check_llm.py --provider deepseek --thinking off
python check_llm.py --provider ollama --models gpt-oss:20b
```

**DeepSeek results (Nov 3):**

| Model | Thinking | First guess | Time | Output tokens | Reasoning text |
|---|---|---|---|---|---|
| deepseek-flash | default (on) | CRANE | 2.5 s | 76 | 238 chars |
| deepseek-v4-pro | default (on) | CRANE | 4.5 s | 386 | 1,619 chars |
| deepseek-flash | off | CRANE | 1.6 s | 33 | none |
| deepseek-v4-pro | off | ADIEU | 1.4 s | 3 | none |

Both models work, and the thinking switch works (no reasoning text when it is off).

One thing to keep in mind for the solver: with thinking off, `deepseek-v4-pro` answered with only the word and skipped the reason the prompt asks for. So the solver should not depend on the reply format. It should take the first 5-letter word, which is what `check_llm.py` does (`first_word()`).

**Ollama / gpt-oss:20b:** the model is downloaded and the server answers, but the full check has not passed yet. We could not get a GPU large enough to load the model in time (it needs about 11 GB of GPU memory, ideally 16 GB or more). On smaller cards, loading timed out. The code side is done. Once an Ollama server with the model is running, the check is just the third command above. If loading keeps being a problem, a smaller open-weight model such as `qwen3:8b` (about 5 GB) is an easy swap: only the model name changes.

---

## 4. Tests

```bash
python -m pytest -q      # 69 passed
```

This includes Soha's tests plus `test_letter_probs.py` and `test_llm_client.py`. The LLM tests use a fake client, so they need no key and no network.

---

## 5. Notes for the next tasks

**Task 6 (LLM solver):**

- The paper's prompts (Appendix A) are copied in `check_llm.py` as `RULES`. The retry prompts (Listing 4) still need to be added.
- `is_consistent` and `CandidateFilter.mask` in Soha's `filter.py` can be the "does this guess fit the feedback" check for the retry loop. The paper allowed up to 5 retries.
- Parse the guess with the first-5-letter-word rule above, and check it is in `allowed.txt` (the environment raises `ValueError` for an invalid word without using up a turn).
- Run several games at once (for example 4 to 8 threads, since the work is waiting on the API, not on the CPU) and use the same secret list as the other solvers (`evaluation.secret_list`).
- Log tokens and time per game, since DeepSeek is paid.

**Task 5 (Fibble PPO solver):**

- The probability table from section 2 is the input.
- I have a working draft (word scorer, PPO in PyTorch, training script) that is not in the repo yet. Its word scorer reproduces the paper's worked example (Table III) to within 0.5%. In a short test it already beat the paper's PPO numbers, but our rule-based baseline still beat it on Fibble4. I can share it when we start task 5.

**Things in the paper worth mentioning in the report:**

- Section IV-E's worked example is not consistent: the board colors in Fig. 4 are not the real feedback for EARLY, and the seven-word candidate list does not follow from those colors either. Our scorer reproduces Table III only when using the Fig. 4 colors as given.
- The targeted solver is described with 8 guesses plus a random first guess in Section IV, but Table IV says 9 guesses.
- The o1-preview Fibble win rate is given as both 29.62% and 29.63%.

---

## 6. Commits for task 3

| Commit | What |
|---|---|
| `eef3556` | Letter-position probability table, its tests, saved CSV, full `requirements.txt` |
| `ccc8ad1` | LLM client (DeepSeek and Ollama), `check_llm.py`, offline tests |
| `1fdb182`, `3374986` | Made the LLM settings generic (addresses and key file come from environment variables) and kept deployment files out of the repo |
