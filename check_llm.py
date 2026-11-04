"""Check the LLM setup: key works, model is served, one Wordle-style reply.

    python check_llm.py --provider deepseek                      # deepseek-flash and deepseek-v4-pro
    python check_llm.py --provider deepseek --thinking off
    python check_llm.py --provider ollama --models gpt-oss:20b   # needs an Ollama server (OLLAMA_BASE_URL)

Prints, per model: the reply, its first 5-letter word, time and tokens used.
Also appends to results/llm_check.csv.
"""

import argparse
import csv
import re
from pathlib import Path

from wordle_fibble.llm_client import PROVIDERS, LLMClient, list_models

# The paper's rules prompt (Appendix A, Listing 1) and a first-turn message.
RULES = (
    "You will guess a 5 letter word based off previous guesses and feedback in the form of "
    "correct: correct place, present: letter is present but not in correct spot, and "
    "incorrect: letter is not present in the word.\n"
    "There may be more than one lies, meaning one or more feedbacks will be incorrect.\n"
    "Assume there are no lies unless otherwise stated.\n"
    "Respond with the 5 letter word and then the reason why you chose that word, nothing else!\n"
    "If no feedback is provided, you must guess the word without feedback.\n"
    "You only have a certain amount of tries to get the word."
)
FIRST_TURN = "No feedback yet. You have 6 tries remaining."
FIELDS = ["provider", "model", "thinking", "ok", "word", "seconds", "input_tokens",
          "output_tokens", "reasoning_chars", "error"]


def first_word(text):
    m = re.search(r"\b[A-Za-z]{5}\b", text)
    return m.group(0).lower() if m else ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--provider", choices=list(PROVIDERS), default="deepseek")
    p.add_argument("--models", nargs="+", default=None, help="default: the provider's models")
    p.add_argument("--thinking", choices=["default", "on", "off"], default="default")
    p.add_argument("--csv", default="results/llm_check.csv")
    args = p.parse_args()
    models = args.models or PROVIDERS[args.provider]["models"]
    thinking = {"default": None, "on": True, "off": False}[args.thinking]

    try:
        served = list_models(args.provider)
        print(f"{args.provider} serves:", ", ".join(served))
    except Exception as e:
        served = None
        print(f"could not list models ({type(e).__name__}: {e})")

    rows = []
    for model in models:
        row = {"provider": args.provider, "model": model, "thinking": args.thinking}
        print(f"\n--- {args.provider}/{model} (thinking={args.thinking})")
        if served is not None and model not in served:
            print("not in the served list; trying anyway")
        try:
            llm = LLMClient(args.provider, model, thinking=thinking)
            r = llm.chat([{"role": "system", "content": RULES}, {"role": "user", "content": FIRST_TURN}])
        except Exception as e:
            print(f"FAILED: {type(e).__name__}: {e}")
            rows.append({**row, "ok": False, "error": str(e)[:200]})
            continue
        print(r.text[:300])
        print(f"word={first_word(r.text)!r}  {r.seconds}s  in={r.input_tokens} out={r.output_tokens} "
              f"reasoning_chars={len(r.reasoning)}")
        rows.append({**row, "ok": True, "word": first_word(r.text), "seconds": r.seconds,
                     "input_tokens": r.input_tokens, "output_tokens": r.output_tokens,
                     "reasoning_chars": len(r.reasoning)})

    if rows and args.csv:
        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        new = not path.exists()
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                writer.writeheader()
            writer.writerows(rows)
        print(f"\nappended to {path}")


if __name__ == "__main__":
    main()
