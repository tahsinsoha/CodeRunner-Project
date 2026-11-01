from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
WORD_LEN = 5
ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def load_words(path):
    with open(path) as f:
        words = [w.strip().lower() for w in f if w.strip()]
    bad = [w for w in words if len(w) != WORD_LEN or not w.isalpha()]
    if bad:
        raise ValueError(f"{path} has invalid entries, e.g. {bad[:3]}")
    return words


def load_answers():
    """The 2,315 possible secret words."""
    return load_words(DATA_DIR / "answers.txt")


def load_allowed():
    """The 12,972 accepted guesses (a superset of the answers)."""
    return load_words(DATA_DIR / "allowed.txt")


def encode(word):
    """'crane' -> array([2, 17, 0, 13, 4]) as uint8 letter indices."""
    return np.frombuffer(word.encode("ascii"), dtype=np.uint8) - ord("a")


def encode_many(words):
    """List of N words -> (N, 5) uint8 array of letter indices."""
    joined = "".join(words).encode("ascii")
    return (np.frombuffer(joined, dtype=np.uint8) - ord("a")).reshape(-1, WORD_LEN)
