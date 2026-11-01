"""Play Wordle/Fibble in the terminal.

    python play.py                      # Wordle, 6 guesses, you type the guesses
    python play.py --lies 1 --guesses 9 # original Fibble
    python play.py --lies 3 --auto      # random-guess bot plays one game
    python play.py --lies 2 --auto --games 1000   # quick win-rate check

Feedback legend: G = green (correct spot), Y = yellow (in word), - = gray.
"""

import argparse

from wordle_fibble import FibbleEnv


def play_human(env, args):
    env.reset(secret=args.secret)
    print(f"{env.name}: {env.n_lies} lie(s) per line, {env.max_guesses} guesses.")
    print(env.render())
    while not env.done:
        guess = input(f"guess ({env.guesses_left} left)> ")
        try:
            env.step(guess)
        except ValueError as e:
            print(e)
            continue
        print(env.render(show_truth=env.done))


def play_random(env, args):
    wins, turns = 0, 0
    for game in range(args.games):
        env.reset(secret=args.secret)
        while not env.done:
            guess = env.allowed[env.rng.integers(len(env.allowed))]
            env.step(guess)
        wins += env.won
        turns += len(env.guesses)
        if args.games == 1:
            print(env.render(show_truth=True))
    if args.games > 1:
        print(f"{env.name}, random guesser: win {wins / args.games:.2%}, "
              f"mean guesses {turns / args.games:.2f} over {args.games} games")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lies", type=int, default=0, help="lies per feedback line (0-5)")
    p.add_argument("--guesses", type=int, default=None, help="guess limit (default 6 for Wordle, 9 for Fibble)")
    p.add_argument("--secret", default=None, help="fix the secret word")
    p.add_argument("--opener", action="store_true", help="start with a random first guess, as in original Fibble")
    p.add_argument("--auto", action="store_true", help="let a random-guess bot play")
    p.add_argument("--games", type=int, default=1)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    max_guesses = args.guesses or (6 if args.lies == 0 else 9)
    env = FibbleEnv(n_lies=args.lies, max_guesses=max_guesses,
                    initial_random_guess=args.opener, seed=args.seed)
    (play_random if args.auto else play_human)(env, args)


if __name__ == "__main__":
    main()
