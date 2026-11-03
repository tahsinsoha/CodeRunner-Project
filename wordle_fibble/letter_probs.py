"""Letter-by-position probability table (the paper's Table I).

The paper builds Table I from a 5,757-word list that we do not have. We use
the 12,972 allowed guesses instead: its table is within 0.034 of Table I in
every cell (mean difference 0.004), while the 2,315 answer list is far off
(e.g. S in position 5: 0.016 vs the paper's 0.306).

    python -m wordle_fibble.letter_probs     # writes data/letter_probs.csv
"""

import numpy as np

from .words import ALPHABET, DATA_DIR, WORD_LEN, encode_many, load_allowed


def letter_position_probs(words=None):
    """(5, 26) array: probs[pos, letter] = share of words with that letter at pos.

    Each row sums to 1. Defaults to the allowed-guess list.
    """
    codes = encode_many(words if words is not None else load_allowed())
    counts = np.stack(
        [np.bincount(codes[:, pos], minlength=len(ALPHABET)) for pos in range(WORD_LEN)]
    )
    return counts / len(codes)


def save_csv(probs, path):
    """Write the table in the paper's layout: one row per letter, one column per position."""
    with open(path, "w") as f:
        f.write("letter," + ",".join(str(p + 1) for p in range(WORD_LEN)) + "\n")
        for i, letter in enumerate(ALPHABET.upper()):
            f.write(letter + "," + ",".join(f"{probs[p, i]:.4f}" for p in range(WORD_LEN)) + "\n")


def load_csv(path=None):
    """Read a table written by `save_csv` back into a (5, 26) array."""
    path = path or DATA_DIR / "letter_probs.csv"
    rows = np.loadtxt(path, delimiter=",", skiprows=1, usecols=range(1, WORD_LEN + 1))
    return rows.T


if __name__ == "__main__":
    probs = letter_position_probs()
    out = DATA_DIR / "letter_probs.csv"
    save_csv(probs, out)
    print(f"wrote {out}")
    top = [ALPHABET.upper()[i] for i in probs.argmax(axis=1)]
    print("most likely letter per position:", " ".join(top))
