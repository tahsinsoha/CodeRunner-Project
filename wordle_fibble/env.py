import numpy as np

from .feedback import COLOR_CHARS, GREEN, apply_lies, format_line, score
from .words import ALPHABET, WORD_LEN, encode, load_allowed, load_answers

UNSEEN = -1


class FibbleEnv:
    """Wordle / Fibble game with `n_lies` wrong tiles in every feedback line.

    n_lies=0 is Wordle, 1 is the original Fibble, 2-5 are Fibble2-Fibble5.

        env = FibbleEnv(n_lies=1, max_guesses=9, seed=0)
        obs = env.reset()
        obs, done, info = env.step("crane")

    Feedback colors are 0 = gray, 1 = yellow, 2 = green. Observations only
    contain the feedback the player is shown; the true feedback and the lie
    positions are in `info` and `true_history` for analysis, not for solvers.
    """

    def __init__(
        self,
        n_lies=0,
        max_guesses=6,
        answers=None,
        allowed=None,
        initial_random_guess=False,
        seed=None,
    ):
        if not 0 <= n_lies <= WORD_LEN:
            raise ValueError(f"n_lies must be between 0 and {WORD_LEN}, got {n_lies}")
        if max_guesses < 1:
            raise ValueError("max_guesses must be at least 1")
        self.n_lies = n_lies
        self.max_guesses = max_guesses
        self.answers = list(answers) if answers is not None else load_answers()
        self.allowed = list(allowed) if allowed is not None else load_allowed()
        self.allowed_set = set(self.allowed) | set(self.answers)
        self.initial_random_guess = initial_random_guess
        self.rng = np.random.default_rng(seed)
        self.secret = None
        self.done = True

    @property
    def name(self):
        return "Wordle" if self.n_lies == 0 else f"Fibble{self.n_lies}"

    @property
    def guesses_left(self):
        return self.max_guesses - len(self.guesses)

    def reset(self, secret=None, seed=None):
        """Start a new game. Pass `secret` to fix the answer across solvers."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if secret is None:
            secret = self.answers[self.rng.integers(len(self.answers))]
        secret = secret.lower()
        if secret not in self.allowed_set:
            raise ValueError(f"secret {secret!r} is not in the word list")
        self.secret = secret
        self.guesses = []
        self.shown_history = []
        self.true_history = []
        self.lie_history = []
        self.won = False
        self.done = False
        # Original Fibble hands the player a first guess; it uses up one turn.
        if self.initial_random_guess:
            opener = secret
            while opener == secret:
                opener = self.allowed[self.rng.integers(len(self.allowed))]
            self._play(opener)
        return self.observation()

    def step(self, guess):
        """Play one guess. Returns (observation, done, info).

        Raises ValueError for a word not in the allowed list; the turn is not
        used up, so a solver can retry.
        """
        if self.done:
            raise RuntimeError("game is over; call reset() first")
        guess = guess.strip().lower()
        if len(guess) != WORD_LEN or guess not in self.allowed_set:
            raise ValueError(f"{guess!r} is not a valid guess")
        info = self._play(guess)
        return self.observation(), self.done, info

    def _play(self, guess):
        true_colors = score(guess, self.secret)
        won = bool((true_colors == GREEN).all())
        if won:
            # The game ends on a correct guess, so that line is shown truthfully.
            shown, lies = true_colors.copy(), np.array([], dtype=np.int64)
        else:
            shown, lies = apply_lies(true_colors, self.n_lies, self.rng)
        self.guesses.append(guess)
        self.true_history.append(true_colors)
        self.shown_history.append(shown)
        self.lie_history.append(lies)
        self.won = won
        self.done = won or self.guesses_left == 0
        return {
            "won": won,
            "true_feedback": true_colors,
            "lie_positions": lies,
            "secret": self.secret if self.done else None,
        }

    def observation(self):
        """What a solver may see.

        board / letters: (max_guesses, 5) colors and letter indices, -1 if unplayed.
        alphabet: (26,) last color shown for each letter, -1 if never guessed.
        """
        board = np.full((self.max_guesses, WORD_LEN), UNSEEN, dtype=np.int8)
        letters = np.full((self.max_guesses, WORD_LEN), UNSEEN, dtype=np.int8)
        alphabet = np.full(len(ALPHABET), UNSEEN, dtype=np.int8)
        for row, (guess, colors) in enumerate(zip(self.guesses, self.shown_history)):
            board[row] = colors
            letters[row] = encode(guess)
            for idx, c in zip(letters[row], colors):
                alphabet[idx] = c
        return {
            "guesses": list(self.guesses),
            "feedback": [c.copy() for c in self.shown_history],
            "guesses_left": self.guesses_left,
            "n_lies": self.n_lies,
            "board": board,
            "letters": letters,
            "alphabet": alphabet,
        }

    def render(self, show_truth=False):
        lines = [f"{self.name}  ({len(self.guesses)}/{self.max_guesses} guesses)"]
        for i, guess in enumerate(self.guesses):
            line = format_line(guess, self.shown_history[i])
            if show_truth and self.n_lies:
                truth = " ".join(COLOR_CHARS[int(c)] for c in self.true_history[i])
                line += f"   truth: {truth}"
            lines.append(line)
        if self.done:
            lines.append(f"{'WON' if self.won else 'LOST'} - secret was {self.secret.upper()}")
        return "\n".join(lines)


def make_wordle(max_guesses=6, **kwargs):
    return FibbleEnv(n_lies=0, max_guesses=max_guesses, **kwargs)


def make_fibble(n_lies=1, max_guesses=9, **kwargs):
    return FibbleEnv(n_lies=n_lies, max_guesses=max_guesses, **kwargs)
