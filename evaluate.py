"""Win rate and mean guesses for the baseline solvers on Wordle and Fibble1-5.

    python evaluate.py                                   # both baselines, 500 games, 6 guesses
    python evaluate.py --solvers rule_based --games 2315 --guesses 9
    python evaluate.py --lies 0 1 --csv results/baselines.csv
    python evaluate.py --candidates allowed   # don't assume secrets come from the answer list
    python evaluate.py --solvers --dqn checkpoints/dqn_wordle.pt   # a trained DQN only

Secrets are drawn from a fixed seed, so every solver sees the same words.
"""

import argparse
import csv
import time
from pathlib import Path

from wordle_fibble.evaluation import evaluate, secret_list
from wordle_fibble.filter import CandidateFilter
from wordle_fibble.solvers import SOLVERS
from wordle_fibble.words import load_allowed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--solvers", nargs="*", default=list(SOLVERS), choices=list(SOLVERS))
    p.add_argument("--dqn", nargs="+", default=[], help="trained DQN checkpoints to evaluate")
    p.add_argument("--lies", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5])
    p.add_argument("--guesses", type=int, default=6, help="guess limit (paper uses 6, 8 or 9)")
    p.add_argument("--games", type=int, default=500)
    p.add_argument("--opener", action="store_true", help="random first guess, as in original Fibble")
    p.add_argument("--candidates", choices=["answers", "allowed"], default="answers",
                   help="word list the solvers treat as possible secrets")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--csv", default=None, help="also write results to this CSV file")
    args = p.parse_args()

    secrets = secret_list(args.games, seed=args.seed)
    candidate_filter = CandidateFilter(load_allowed() if args.candidates == "allowed" else None)
    makers = [(name, lambda name=name: SOLVERS[name](candidate_filter, seed=args.seed))
              for name in args.solvers]
    if args.dqn:
        from wordle_fibble.dqn import DQNSolver
        makers += [(Path(path).stem, lambda path=path: DQNSolver.load(path)) for path in args.dqn]

    rows = []
    print(f"{'solver':<20}{'game':<9}{'win %':>8}{'mean':>7}   wins by round")
    for name, make in makers:
        for n_lies in args.lies:
            solver = make()
            solver.name = name
            start = time.perf_counter()
            r = evaluate(solver, secrets, n_lies, args.guesses,
                         initial_random_guess=args.opener, seed=args.seed)
            r["seconds"] = round(time.perf_counter() - start, 1)
            rows.append(r)
            print(f"{name:<20}{r['game']:<9}{r['win_rate']:>8.2%}{r['mean_guesses']:>7.2f}   "
                  f"{r['wins_by_round']}")

    if args.csv:
        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
