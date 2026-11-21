# Task 6: LLM solver, and its experiments (Lakshmi)

This note covers the LLM solver (task 6 in the proposal), the LLM side of the experiments (task 7), what we found, and how it compares with the paper and with our other solvers. All numbers come from `results/llm_results.csv` and `results/same50_baselines_dqn_*.csv`, which `summarize_llm.py` and `evaluate.py` produce from the saved games.

Short version: with reasoning turned on, DeepSeek plays Wordle perfectly and stays strong on Fibble (80% on its own with two lies per line). Without reasoning, both models lose track of the feedback, and Qwen 2.5 7B almost never wins on its own. A lot of the "wins" in the weaker settings were made by our fallback, not by the model, so we report both numbers.

---

## 1. What was built

| File | What it does |
|---|---|
| `wordle_fibble/llm_solver.py` | The solver: builds the prompts, reads the guess from the reply, runs the retry loop |
| `wordle_fibble/llm_runner.py` | Plays many games at the same time and saves each game as soon as it ends; a stopped run continues where it left off |
| `run_llm.py` | Runs one model on Wordle and Fibble1–5 |
| `summarize_llm.py` | Checks every saved game and builds the summary table |
| `tests/test_llm_solver.py` | Offline tests with a fake model (prompt format, parsing, every retry path, resume, parallel = sequential) |

### Prompts (paper Appendix A)

Every turn the model gets a fresh conversation with:

1. the rules (Listing 1), plus the number of lies for Fibble (Listing 3: "There is one lie present in the game", and "There are N lies present in the game" for 2–5);
2. every earlier guess with its feedback, one line per letter (Listing 2: `A: incorrect`, `I: correct`, `S: present`);
3. how many tries are left.

### Reading the reply

The guess is the first ALL-CAPS 5-letter word in the word list, otherwise the reply's first word, otherwise a **bold** or quoted word, otherwise the first listed word anywhere. Opening words like "Guess:", "Sure," or "Based" are skipped. The model's first word counts as its answer even when it is not a real word, so the retry can tell it so.

### Retry loop

- **The guess does not fit the feedback** (given the number of lies): the model is asked again, up to 5 times as in the paper, with a hint in the style of Listing 4 for Wordle ("'E' must be in the word", "'E' is not a possible letter for this spot, the valid letter is 'A'"). For Fibble the hint says the word does not fit given N lies per line. After 5 tries the last valid guess is played. With `--retries 0` the guess is played as it is (the paper's first phase).
- **The reply is empty, has no word, has a word not in the list, or repeats a guess**: the model is asked again, up to 3 times, because the game cannot play such a guess.
- **Fallback**: if those 3 tries also fail, the solver plays the first word from the 2,315-word answer list that fits the feedback. This keeps every game going, but that guess comes from our filter (which also knows the answer list), not from the model. See section 3.

"No retries" below means no retries for guesses that break the feedback. Unplayable replies were still asked again, and fallbacks could still happen.

---

## 2. Setup

| | |
|---|---|
| Models | **DeepSeek flash** (DeepSeek V4.1 Flash, hosted API) with thinking on and off; **Qwen 2.5 7B** (open-weight, run with Ollama on a GPU, Ollama's default 4-bit build) |
| Settings | DeepSeek: thinking on + 5 retries, thinking off + 5 retries, thinking off + no retries. Qwen: 5 retries, no retries |
| Games | 50 per setting (the paper used 47, and 27 for o1-preview on Fibble), 1,500 games in total |
| Guess limit | 6 for Wordle, 9 for Fibble (as in the paper's Section V) |
| Secret words | The same 50 words for every solver (`secret_list(50, seed=0)`), and the same per-game random seed for the lies as `evaluate.py` |
| Other | Provider default temperature and reasoning effort; no cap on output tokens |

### Data checks

`summarize_llm.py` checks every game before counting it: the secret is the expected word for that game number, every non-winning feedback line has exactly N wrong tiles, a win is shown truthfully, no guess repeats, and `won` matches the last line being all green. **All 1,500 games pass.**

---

## 3. Results

Each cell is **win % with the fallback / win % by the model alone** (games won without any fallback guess). The second number is the model's own result. It is a lower bound, because a game that needed even one fallback counts as a loss.

| Model | Wordle | Fibble1 | Fibble2 | Fibble3 | Fibble4 | Fibble5 |
|---|---|---|---|---|---|---|
| DeepSeek flash, thinking on, 5 retries | 100 / **100** | 100 / **96** | 92 / **80** | 62 / **50** | 54 / **46** | 88 / **70** |
| DeepSeek flash, thinking off, 5 retries | 90 / **86** | 78 / **40** | 30 / **10** | 22 / **2** | 16 / **0** | 34 / **4** |
| DeepSeek flash, thinking off, no retries | 52 / **52** | 18 / **10** | 12 / **2** | 8 / **4** | 2 / **0** | 4 / **0** |
| Qwen 2.5 7B, 5 retries | 86 / **2** | 80 / **4** | 64 / **6** | 44 / **2** | 32 / **4** | 56 / **0** |
| Qwen 2.5 7B, no retries | 6 / **0** | 28 / **6** | 22 / **2** | 8 / **2** | 10 / **0** | 18 / **2** |

Mean guesses (a loss counts as the full 6 or 9, as in the paper):

| Model | Wordle | Fibble1 | Fibble2 | Fibble3 | Fibble4 | Fibble5 |
|---|---|---|---|---|---|---|
| DeepSeek flash, thinking on, 5 retries | 3.62 | 4.82 | 6.62 | 7.90 | 7.90 | 6.50 |
| DeepSeek flash, thinking off, 5 retries | 4.24 | 6.66 | 8.68 | 8.78 | 8.78 | 8.46 |
| DeepSeek flash, thinking off, no retries | 5.42 | 8.70 | 8.74 | 8.92 | 9.00 | 9.00 |
| Qwen 2.5 7B, 5 retries | 4.72 | 6.84 | 7.50 | 8.38 | 8.26 | 7.86 |
| Qwen 2.5 7B, no retries | 5.88 | 8.48 | 8.62 | 8.84 | 8.70 | 8.60 |

With 50 games, the 95% interval on a win rate is about ±10–14 points (Wilson intervals for every cell are in the CSV). For example, DeepSeek with thinking on Fibble3 is 62% with an interval of 48–74%. Differences of less than about 20 points between two settings should not be read as real.

### Why the fallback matters

| Model | Game | Games with a fallback | Won by a fallback guess |
|---|---|---|---|
| Qwen 2.5 7B, 5 retries | Wordle | 48 of 50 | 38 |
| Qwen 2.5 7B, 5 retries | Fibble1 | 45 | 37 |
| DeepSeek flash, thinking off, 5 retries | Fibble1 | 24 | 17 |
| DeepSeek flash, thinking on, 5 retries | Fibble3 | 14 | 4 |

The weaker models mostly **repeat an earlier guess** or name a word that is not in the list, even though the prompt lists every earlier guess. After three such replies the fallback plays a word that fits the feedback, which is about as good as our random baseline (96–100% on Wordle). So "Qwen with retries wins 86% of Wordle games" mostly measures our filter. With thinking on, DeepSeek rarely needed it, and most of its fallbacks there came from replies cut off by very long reasoning.

---

## 4. What we found

1. **Reasoning is what makes an LLM a working solver.** On its own, DeepSeek with thinking wins 100 / 96 / 80 / 50 / 46 / 70% on Wordle and Fibble1–5. The same model without thinking wins 86 / 40 / 10 / 2 / 0 / 4% even with retries. This is the same gap the paper saw between o1-preview and GPT-4o, and it is larger for us on Fibble.
2. **Retries help a lot on Wordle, much less on Fibble.** Without thinking, DeepSeek goes from 52% to 86% on Wordle with retries (model alone), close to the paper's GPT-4o going from 21% to 60%. On Fibble1 it goes from 10% to 40%, and above that it stays near zero. The retry hints for Fibble are weaker (they cannot say which tile is the lie).
3. **A 7B open model cannot play this on its own.** Qwen 2.5 7B wins at most 6% without help in every setting. It struggles even with the basics: repeating words and naming non-words.
4. **Fibble5 is easier than Fibble4.** With thinking, 70% on Fibble5 vs 46% on Fibble4 (model alone). With all five tiles lying, every colour is known to be wrong, which tells the player more than one hidden truth among five lies; the reasoning model seems to use that. The paper's targeted RL solver shows the same pattern (61.1% vs 38.6%).
5. **The reasoning costs a lot.** With thinking, a Fibble game used about 116,000–582,000 output tokens and 12–61 minutes on average per setting; without thinking, about 5,000–25,000 tokens and under 3 minutes. Tokens for the counted games: thinking on 1.0M in / 100.8M out; thinking off with retries 19.6M / 4.6M; thinking off without retries 2.7M / 1.7M. (Actual spend was higher because of interrupted runs, see section 6.)

---

## 5. Comparison

### With the paper (Tables IV and V)

| | Wordle | Fibble1 | Fibble2 | Fibble3 | Fibble4 | Fibble5 |
|---|---|---|---|---|---|---|
| Paper: GPT-4o, no retries | 21.27 | 14.89 | 2.12 | 0.00 | 0.00 | 0.00 |
| **Ours: DeepSeek thinking off, no retries** (model alone) | 52 | 10 | 2 | 4 | 0 | 0 |
| Paper: GPT-4o, 5 retries | 59.57 | – | – | – | – | – |
| **Ours: DeepSeek thinking off, 5 retries** (model alone) | 86 | 40 | 10 | 2 | 0 | 4 |
| Paper: o1-preview (reasoning) | 95.74 | 29.62 | – | – | – | – |
| **Ours: DeepSeek thinking on, 5 retries** (model alone) | 100 | 96 | 80 | 50 | 46 | 70 |
| Paper: targeted Fibble RL solver | 99.40 | 96.90 | 84.50 | 49.0 | 38.60 | 61.1 |

Our o1-preview analogue (thinking on) also had retries, which the paper's o1-preview did not, so that comparison favours us somewhat on Fibble. Even so, DeepSeek with thinking on Fibble1 (96% alone) is far above o1-preview (29.62%), and it is in the same range as the paper's targeted RL solver on Fibble1–5.

### With our other solvers, on the same 50 words

The baselines and Soha's DQN were run again on exactly the words the LLMs played, with 6 guesses for Wordle and 9 for Fibble. Win % (mean guesses):

| Solver | Wordle | Fibble1 | Fibble2 | Fibble3 | Fibble4 | Fibble5 |
|---|---|---|---|---|---|---|
| Rule-based baseline (filter + common letters) | 100 (3.66) | 100 (4.56) | 100 (6.02) | 88 (7.62) | 80 (7.76) | 96 (6.44) |
| Random baseline (random fitting word) | 96 (4.08) | 100 (4.74) | 98 (6.26) | 82 (7.64) | 74 (8.18) | 84 (6.86) |
| DQN trained on Wordle (Soha) | 96 (4.20) | 98 (5.36) | 96 (6.38) | 78 (7.64) | 72 (8.06) | 88 (6.72) |
| DQN trained on Fibble1 (Soha) | 100 (4.30) | 98 (4.98) | 96 (6.74) | 78 (8.08) | 52 (8.46) | 86 (7.40) |
| Best LLM: DeepSeek thinking, model alone | 100 | 96 | 80 | 50 | 46 | 70 |

Every non-LLM solver here uses our lie-aware candidate filter, so it only ever guesses words that fit the feedback exactly. The LLM has to work that out from text. That is why even the random baseline beats the best LLM on Fibble2–5. Up to Fibble1, the reasoning LLM keeps up with them.

---

## 6. Problems we hit, and how they were handled

- **Cut-off replies.** In the first test, a 4,096-token cap made thinking replies end before the answer (17 empty replies in 5 games). Fixed in `32a563f`: no cap, and empty or cut-off replies are asked again. Those 5 test games are not part of the results.
- **Running out of API credit.** The runner now stops cleanly with a clear message and keeps every finished game (`0a52240`). Re-running the same command plays only the missing games.
- **Games played twice.** When the credit ran out, the runner let games already in progress finish, while the restarted run played the same words again. So 72 words in the thinking-on Fibble2–5 runs (10, 24, 21, 17) were played twice, as two complete, valid games. Both plays are kept in the logs. Which one counts is decided by a hash of the word, never by the result, because "first saved" would favour quick wins. Averaging both plays instead gives almost the same win rates (Fibble2 94 vs 92, Fibble3 62 vs 62, Fibble4 55 vs 54, Fibble5 87 vs 88). The runner is now fixed so that games in progress also stop at their next call; a test checks this.
- **Fallback guesses.** Reported as a second column instead of being hidden; see section 3.

---

## 7. Limits

- 50 games per setting: the intervals are wide (about ±10–14 points).
- One run per setting at the provider's default temperature, so the results are not exactly repeatable.
- The "model alone" number is a lower bound; the "with fallback" number is an upper bound.
- Only one reasoning setting (DeepSeek's default reasoning effort) and one small open model.

---

## 8. How to reproduce

```bash
python run_llm.py --provider deepseek --model deepseek-flash --thinking on               # thinking on, 5 retries
python run_llm.py --provider deepseek --model deepseek-flash --thinking off              # thinking off, 5 retries
python run_llm.py --provider deepseek --model deepseek-flash --thinking off --retries 0  # no retries
python run_llm.py --provider ollama --model qwen2.5:7b                                   # needs OLLAMA_BASE_URL
python run_llm.py --provider ollama --model qwen2.5:7b --retries 0
python summarize_llm.py                                                                  # checks + results/llm_results.csv
python evaluate.py --solvers random rule_based --dqn checkpoints/dqn_wordle.pt checkpoints/dqn_fibble1.pt --games 50 --guesses 6 --lies 0
python evaluate.py --solvers random rule_based --dqn checkpoints/dqn_wordle.pt checkpoints/dqn_fibble1.pt --games 50 --guesses 9 --lies 1 2 3 4 5
```

Each game is saved in `results/llm_runs/<setting>.jsonl` with every guess, the shown and true feedback, every model reply, and the tokens used, so any game can be replayed for the demo.

---

## 9. What is left

- Task 5 (PPO solver): the draft code is done; it needs updating to the current code, training, and evaluating on the same 50 words.
- Report sections for the LLM solver and the PPO solver, and the demo video (replaying saved games, so no new API calls).
