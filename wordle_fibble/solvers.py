import numpy as np

from .filter import CandidateFilter
from .words import ALPHABET, WORD_LEN


class Solver:
    """A solver picks the next guess from an environment observation."""

    name = "solver"

    def reset(self):
        pass

    def act(self, obs):
        raise NotImplementedError


class _FilteredSolver(Solver):
    def __init__(self, candidate_filter=None, seed=None):
        self.filter = candidate_filter or CandidateFilter()
        self.rng = np.random.default_rng(seed)

    def _candidate_indices(self, obs):
        idx = np.flatnonzero(self.filter.mask(obs["guesses"], obs["feedback"], obs["n_lies"]))
        if idx.size == 0:
            # Only reachable if the secret is outside the filter's word list.
            idx = np.array([i for i, w in enumerate(self.filter.words) if w not in obs["guesses"]])
        return idx


class RandomSolver(_FilteredSolver):
    """Guesses a random word that still fits the feedback."""

    name = "random"

    def act(self, obs):
        idx = self._candidate_indices(obs)
        return self.filter.words[self.rng.choice(idx)]


class RuleBasedSolver(_FilteredSolver):
    """Guesses the remaining candidate made of the most common letters.

    Each candidate scores the frequency of its letter at each position among
    the candidates, plus how many candidates contain each of its distinct
    letters. Ties go to the alphabetically first word, so play is deterministic.
    """

    name = "rule_based"

    def act(self, obs):
        idx = self._candidate_indices(obs)
        codes = self.filter.codes[idx]
        n = len(idx)
        pos_freq = np.stack(
            [np.bincount(codes[:, i], minlength=len(ALPHABET)) for i in range(WORD_LEN)]
        ) / n
        present = np.zeros((n, len(ALPHABET)), dtype=bool)
        present[np.arange(n)[:, None], codes] = True
        letter_freq = present.mean(axis=0)
        scores = pos_freq[np.arange(WORD_LEN), codes].sum(axis=1) + present @ letter_freq
        return self.filter.words[idx[np.argmax(scores)]]


SOLVERS = {cls.name: cls for cls in (RandomSolver, RuleBasedSolver)}
