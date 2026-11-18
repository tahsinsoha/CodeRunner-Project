"""Run the LLM solver (task 6) on Wordle and Fibble1-5.

    python run_llm.py --provider deepseek --model deepseek-flash --lies 0 --games 5    # quick try
    python run_llm.py --provider deepseek --model deepseek-flash --thinking off --lies 0 1 2 3 4 5
    python run_llm.py --provider deepseek --model deepseek-flash --retries 0           # no retries (paper phase 1)
    python run_llm.py --provider ollama --model qwen2.5:7b --workers 4

Defaults follow the paper: 50 games per setting (the paper used 47), 6 guesses
for Wordle and 9 for Fibble, up to 5 retries. Secrets come from the same fixed
list as evaluate.py, so every solver sees the same words.

Each game is saved as it ends in results/llm/<solver>_<game>_g<guesses>.jsonl,
and running the same command again continues an interrupted run. The summary
rows (same first columns as evaluate.py) go to results/llm_results.csv.
"""

import argparse
import csv
from pathlib import Path

from wordle_fibble.evaluation import secret_list
from wordle_fibble.llm_client import LLMClient
from wordle_fibble.llm_runner import run_setting


def solver_label(args):
    think = {"default": "", "on": "-think", "off": "-nothink"}[args.thinking]
    effort = f"-{args.reasoning_effort}" if args.reasoning_effort else ""
    model = args.model.replace(":", "-").replace("/", "-")
    return f"{model}{think}{effort}-r{args.retries}"


def write_summary(path, row):
    """Add `row` to the CSV, replacing an older row for the same solver, game and guess limit."""
    path = Path(path)
    rows = []
    if path.exists():
        with open(path, newline="") as f:
            rows = [r for r in csv.DictReader(f)
                    if (r["solver"], r["game"], r["max_guesses"]) !=
                       (row["solver"], row["game"], str(row["max_guesses"]))]
    rows.append({k: v for k, v in row.items()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--provider", choices=["deepseek", "ollama", "custom"], default="deepseek")
    p.add_argument("--model", default=None, help="default: the provider's first model")
    p.add_argument("--thinking", choices=["default", "on", "off"], default="default")
    p.add_argument("--reasoning-effort", default=None, help="DeepSeek thinking: low / high / max")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--lies", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5])
    p.add_argument("--guesses", type=int, default=None, help="default: 6 for Wordle, 9 for Fibble")
    p.add_argument("--games", type=int, default=50)
    p.add_argument("--retries", type=int, default=5, help="re-asks for guesses that break the feedback")
    p.add_argument("--invalid-retries", type=int, default=3, help="re-asks for replies with no valid word")
    p.add_argument("--workers", type=int, default=4, help="games played at the same time")
    p.add_argument("--opener", action="store_true", help="random first guess, as in original Fibble")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-replies", action="store_true", help="don't store the model's replies")
    p.add_argument("--out-dir", default="results/llm")
    p.add_argument("--csv", default="results/llm_results.csv")
    args = p.parse_args()

    thinking = {"default": None, "on": True, "off": False}[args.thinking]
    probe = LLMClient(args.provider, args.model, thinking=thinking)  # fails early on a missing key
    args.model = probe.model
    label = solver_label(args)

    def make_llm():
        return LLMClient(args.provider, args.model, thinking=thinking,
                         reasoning_effort=args.reasoning_effort, temperature=args.temperature,
                         max_tokens=args.max_tokens)

    secrets = secret_list(args.games, seed=args.seed)
    print(f"{'solver':<34}{'game':<9}{'win %':>8}{'mean':>7}  calls  retries  tokens in/out")
    for n_lies in args.lies:
        guesses = args.guesses or (6 if n_lies == 0 else 9)
        game = "Wordle" if n_lies == 0 else f"Fibble{n_lies}"
        out = Path(args.out_dir) / f"{label}_{game}_g{guesses}.jsonl"
        row = run_setting(make_llm, label, secrets, n_lies, guesses, out, workers=args.workers,
                          seed=args.seed, opener=args.opener, retries=args.retries,
                          invalid_retries=args.invalid_retries, keep_replies=not args.no_replies,
                          log=lambda msg: print("   ", msg, flush=True))
        write_summary(args.csv, row)
        print(f"{label:<34}{game:<9}{row['win_rate']:>8.2%}{row['mean_guesses']:>7.2f}  "
              f"{row['llm_calls']:>5}  {row['retries']:>7}  {row['input_tokens']}/{row['output_tokens']}",
              flush=True)
    print(f"summary rows in {args.csv}; per-game logs in {args.out_dir}/")


if __name__ == "__main__":
    main()
