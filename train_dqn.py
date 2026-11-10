"""Train the Wordle DQN solver (task 4).

    python train_dqn.py                                    # Wordle, 6 guesses, answer list, 20k games
    python train_dqn.py --n-secrets 100 --episodes 5000    # small word list first
    python train_dqn.py --mask unguessed --episodes 100000 # no candidate filter: learn the rules
    python train_dqn.py --lies 1 --guesses 9 --out checkpoints/dqn_fibble1.pt

Writes the network to --out and the training curve to the same name with .csv.
"""

import argparse
import csv
from pathlib import Path

from wordle_fibble.dqn import MASKS, WORD_LISTS, save, train
from wordle_fibble.evaluation import secret_list
from wordle_fibble.words import load_answers


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lies", type=int, default=0)
    p.add_argument("--guesses", type=int, default=6)
    p.add_argument("--words", choices=list(WORD_LISTS), default="answers",
                   help="words the agent can guess")
    p.add_argument("--mask", choices=MASKS, default="candidates")
    p.add_argument("--n-secrets", type=int, default=None,
                   help="train on this many answers only (small word list)")
    p.add_argument("--episodes", type=int, default=20000)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="checkpoints/dqn_wordle.pt")
    args = p.parse_args()

    secrets = load_answers()
    if args.n_secrets:
        secrets = sorted(set(secret_list(args.n_secrets, seed=args.seed + 1000)))
    net, history = train(n_lies=args.lies, max_guesses=args.guesses, words=args.words,
                         secrets=secrets, mask=args.mask, episodes=args.episodes,
                         hidden=args.hidden, lr=args.lr, seed=args.seed,
                         log_every=max(1, args.episodes // 40))
    net.config["n_secrets"] = len(secrets)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save(net, out)
    with open(out.with_suffix(".csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    print(f"wrote {out} and {out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
