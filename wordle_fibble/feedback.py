import numpy as np

from .words import WORD_LEN

GRAY, YELLOW, GREEN = 0, 1, 2
COLOR_NAMES = {GRAY: "incorrect", YELLOW: "present", GREEN: "correct"}
COLOR_CHARS = {GRAY: "-", YELLOW: "Y", GREEN: "G"}


def score(guess, secret):
    """True Wordle feedback for `guess` against `secret`.

    Returns a length-5 int8 array of GRAY/YELLOW/GREEN. Repeated letters follow
    the official rule: greens are assigned first, then yellows left to right,
    and a letter is never marked more times than it appears in the secret.
    """
    colors = np.zeros(WORD_LEN, dtype=np.int8)
    remaining = {}
    for i in range(WORD_LEN):
        if guess[i] == secret[i]:
            colors[i] = GREEN
        else:
            remaining[secret[i]] = remaining.get(secret[i], 0) + 1
    for i in range(WORD_LEN):
        if colors[i] != GREEN and remaining.get(guess[i], 0) > 0:
            colors[i] = YELLOW
            remaining[guess[i]] -= 1
    return colors


def apply_lies(colors, n_lies, rng):
    """Corrupt `n_lies` distinct tiles of a feedback line.

    Each chosen tile is switched to one of the two other colors, picked
    uniformly, so a lie is always a wrong color. Returns (shown, lie_positions).
    """
    if not 0 <= n_lies <= WORD_LEN:
        raise ValueError(f"n_lies must be between 0 and {WORD_LEN}, got {n_lies}")
    shown = colors.copy()
    positions = np.sort(rng.choice(WORD_LEN, size=n_lies, replace=False))
    for p in positions:
        shown[p] = (colors[p] + rng.integers(1, 3)) % 3
    return shown, positions


def format_line(guess, colors):
    """'CRANE  - G Y - -' style one-line rendering."""
    return f"{guess.upper()}  " + " ".join(COLOR_CHARS[int(c)] for c in colors)
