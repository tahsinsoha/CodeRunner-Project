import numpy as np

from .feedback import score, score_many
from .words import encode, encode_many, load_answers


def is_consistent(word, guesses, feedback, n_lies):
    """Could `word` be the secret, given the feedback shown so far?

    A word fits a line when its true feedback differs from the shown feedback
    in exactly `n_lies` tiles. Already-guessed words are ruled out, since
    guessing the secret ends the game.
    """
    word = word.lower()
    if word in guesses:
        return False
    for guess, shown in zip(guesses, feedback):
        if (score(guess, word) != np.asarray(shown)).sum() != n_lies:
            return False
    return True


class CandidateFilter:
    """Keeps the words that are still possible secrets.

        f = CandidateFilter()                 # defaults to the 2,315 answers
        words = f.candidates(obs["guesses"], obs["feedback"], obs["n_lies"])
        words = f.from_obs(obs)               # same thing
    """

    def __init__(self, words=None):
        self.words = list(words) if words is not None else load_answers()
        self.codes = encode_many(self.words)
        self.index = {w: i for i, w in enumerate(self.words)}

    def mask(self, guesses, feedback, n_lies):
        """Boolean array over `self.words`, True where the word still fits."""
        keep = np.ones(len(self.words), dtype=bool)
        for guess, shown in zip(guesses, feedback):
            if guess in self.index:
                keep[self.index[guess]] = False
            sub = np.flatnonzero(keep)
            if sub.size == 0:
                break
            true = score_many(encode(guess), self.codes[sub])
            mismatches = (true != np.asarray(shown)).sum(axis=1)
            keep[sub[mismatches != n_lies]] = False
        return keep

    def candidates(self, guesses, feedback, n_lies):
        return [self.words[i] for i in np.flatnonzero(self.mask(guesses, feedback, n_lies))]

    def from_obs(self, obs):
        return self.candidates(obs["guesses"], obs["feedback"], obs["n_lies"])
