"""LLM solver for Wordle/Fibble (task 6, paper Section V).

Each turn the model gets the paper's prompts (Appendix A):

1. the rules (Listing 1), plus the number of lies for Fibble (Listing 3);
2. every earlier guess with its feedback, one line per letter (Listing 2);
3. how many tries are left.

The reply should start with the guess. We take the first 5-letter word from
it (see `parse_guess`). Then the retry loop (the paper's second phase):

- not a word / not in the word list / already guessed: always asked again,
  up to `invalid_retries` times, since the game cannot play such a guess;
- does not fit the feedback (given the number of lies): asked again up to
  `retries` times, with a hint in the style of the paper's Listing 4.
  With `retries=0` the guess is played as it is (the paper's first phase).

If the model never gives a playable word, we play the first word that still
fits the feedback and count it as a fallback, so a game always finishes.
"""

import re
import time
from dataclasses import dataclass, field

import numpy as np

from .feedback import GRAY, GREEN, YELLOW
from .filter import CandidateFilter, is_consistent
from .solvers import Solver
from .words import WORD_LEN, load_allowed

# Listing 1 of the paper.
RULES = (
    "You will guess a 5 letter word based off previous guesses and feedback in the form of "
    "correct: correct place, present: letter is present but not in correct spot, and "
    "incorrect: letter is not present in the word.\n"
    "There may be more than one lies, meaning one or more feedbacks will be incorrect.\n"
    "Assume there are no lies unless otherwise stated.\n"
    "Respond with the 5 letter word and then the reason why you chose that word, nothing else!\n"
    "If no feedback is provided, you must guess the word without feedback.\n"
    "You only have a certain amount of tries to get the word."
)
COLOR_WORDS = {GREEN: "correct", YELLOW: "present", GRAY: "incorrect"}
KEYBOARD = "QWERTYUIOPASDFGHJKLZXCVBNM"  # letter order used in the paper's Listing 4


def lies_message(n_lies):
    """Listing 3 ("There is one lie present in the game"), extended to 2-5 lies."""
    if n_lies == 0:
        return ""
    if n_lies == 1:
        return "There is one lie present in the game"
    return f"There are {n_lies} lies present in the game"


def system_prompt(n_lies):
    lies = lies_message(n_lies)
    return RULES + ("\n" + lies if lies else "")


def feedback_block(guess, colors):
    """Listing 2: 'Guess: ARISE' then one 'letter: color' line per letter."""
    lines = [f"Guess: {guess.upper()}", "Feedback:"]
    lines += [f"{ch.upper()}: {COLOR_WORDS[int(c)]}" for ch, c in zip(guess, colors)]
    return "\n".join(lines)


def turn_prompt(obs):
    """Earlier guesses with feedback, then the tries left."""
    blocks = [feedback_block(g, c) for g, c in zip(obs["guesses"], obs["feedback"])]
    blocks.append(f"You have {obs['guesses_left']} tries remaining.")
    return "\n\n".join(blocks)


# Opening words that are not the answer ("Sure, ...", "Based on ...", "Guess: crane").
LEAD_INS = {"guess", "based", "given", "sure", "since", "hello", "maybe", "let's", "first",
            "after", "using", "there", "think", "start", "okay", "word"}
_WORD5 = re.compile(r"(?<![A-Za-z])[A-Za-z]{5}(?![A-Za-z])")


def parse_guess(text, allowed):
    """The guess in a reply, or '' if there is none.

    In order: the first ALL-CAPS 5-letter word in the list; the reply's
    first word if it has 5 letters, even if it is not in the list (that is
    the model's answer, so the retry can say it is not a valid word); a
    **bold** or quoted word in the list; the first 5-letter word in the list
    anywhere; any 5-letter word.
    """
    if not text:
        return ""
    words = _WORD5.findall(text)
    for w in words:
        if w.isupper() and w.lower() in allowed:
            return w.lower()
    first = re.sub(r"^[\s*_`'\"#>:.-]+", "", text)
    m = re.match(r"([A-Za-z]+)\s*(:?)", first)
    if m and len(m.group(1)) == 5 and not m.group(2) and m.group(1).lower() not in LEAD_INS:
        return m.group(1).lower()
    for m in re.finditer(r"(\*\*|[\"'`])([A-Za-z]{5})\1", text):
        if m.group(2).lower() in allowed:
            return m.group(2).lower()
    words = [w for w in words if w.lower() not in LEAD_INS]
    for w in words:
        if w.lower() in allowed:
            return w.lower()
    return words[0].lower() if words else ""


def wordle_hint(word, guesses, feedback):
    """A Listing 4 style hint for a Wordle guess that breaks the feedback, or None.

    Assumes the feedback is true, so it is only used when there are no lies.
    """
    green, present, not_at, seen = {}, set(), [set() for _ in range(WORD_LEN)], set()
    gray = set()
    for g, colors in zip(guesses, feedback):
        for pos, (ch, c) in enumerate(zip(g, colors)):
            if c == GREEN:
                green[pos] = ch
                seen.add(ch)
            elif c == YELLOW:
                not_at[pos].add(ch)
                present.add(ch)
                seen.add(ch)
            else:
                gray.add(ch)
                not_at[pos].add(ch)
    absent = gray - seen
    for pos, ch in enumerate(word):
        if pos in green and ch != green[pos]:
            return (f"'{ch.upper()}' is not a possible letter for this spot, "
                    f"the valid letter is '{green[pos].upper()}'")
        if ch in absent or ch in not_at[pos]:
            valid = "".join(k for k in KEYBOARD if k.lower() not in absent | not_at[pos])
            return f"'{ch.upper()}' is not a possible letter for this spot, valid letters are: {valid}"
    for ch in sorted(present | set(green.values())):
        if ch not in word:
            return f"'{ch.upper()}' must be in the word"
    return None


@dataclass
class TurnLog:
    """What happened while choosing one guess."""
    guess: str = ""
    llm_calls: int = 0
    problems: list = field(default_factory=list)   # e.g. ["not_in_list", "inconsistent"]
    fallback: bool = False
    inconsistent_played: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_chars: int = 0
    seconds: float = 0.0
    replies: list = field(default_factory=list)

    @property
    def retries(self):
        return max(0, self.llm_calls - 1)


class LLMSolver(Solver):
    """Plays with an LLM. `llm` needs `.chat(messages) -> LLMReply` (see llm_client).

    retries          max re-asks for guesses that break the feedback (paper: 5; 0 = none)
    invalid_retries  max re-asks for replies with no playable word
    """

    def __init__(self, llm, retries=5, invalid_retries=3, candidate_filter=None,
                 allowed=None, name=None, keep_replies=True):
        self.llm = llm
        self.retries = retries
        self.invalid_retries = invalid_retries
        self.filter = candidate_filter or CandidateFilter()
        self.allowed = set(allowed) if allowed is not None else set(load_allowed())
        self.allowed |= set(self.filter.words)
        self.keep_replies = keep_replies
        self.name = name or f"llm_{getattr(llm, 'label', 'model')}_r{retries}"
        self.turns = []

    def reset(self):
        self.turns = []

    def _problem(self, word, obs):
        """(kind, message to send back) for an unplayable or inconsistent guess, else None."""
        guesses, n = obs["guesses"], obs["n_lies"]
        if not word:
            return "no_word", ("Your answer did not contain a 5 letter word. "
                               "Respond with the 5 letter word and then the reason, nothing else!")
        if word not in self.allowed:
            return "not_in_list", f"'{word.upper()}' is not a valid word. Choose a real 5 letter English word."
        if word in guesses:
            return "repeat", f"'{word.upper()}' was already guessed. Choose a different word."
        if self.retries > 0 and not is_consistent(word, guesses, obs["feedback"], n):
            if n == 0:
                hint = wordle_hint(word, guesses, obs["feedback"])
                return "inconsistent", hint or f"'{word.upper()}' does not fit the feedback. Choose a different word."
            lie = "lie" if n == 1 else "lies"
            return "inconsistent", (f"'{word.upper()}' does not fit the feedback, given that there "
                                    f"{'is' if n == 1 else 'are'} {n} {lie} in every feedback line. "
                                    "Choose a different word.")
        return None

    def _fallback(self, obs):
        idx = np.flatnonzero(self.filter.mask(obs["guesses"], obs["feedback"], obs["n_lies"]))
        if idx.size:
            return self.filter.words[idx[0]]
        return next(w for w in sorted(self.allowed) if w not in obs["guesses"])

    def act(self, obs):
        log = TurnLog()
        start = time.perf_counter()
        messages = [{"role": "system", "content": system_prompt(obs["n_lies"])},
                    {"role": "user", "content": turn_prompt(obs)}]
        fit_retries = bad_retries = 0
        word = None
        while True:
            reply = self.llm.chat(messages)
            log.llm_calls += 1
            log.input_tokens += reply.input_tokens
            log.output_tokens += reply.output_tokens
            log.reasoning_chars += len(reply.reasoning or "")
            if self.keep_replies:
                log.replies.append(reply.text)
            guess = parse_guess(reply.text, self.allowed)
            problem = self._problem(guess, obs)
            if problem is None:
                word = guess
                break
            kind, message = problem
            log.problems.append(kind)
            if kind == "inconsistent":
                if fit_retries >= self.retries:
                    word, log.inconsistent_played = guess, True   # out of retries: play it
                    break
                fit_retries += 1
            else:
                if bad_retries >= self.invalid_retries:
                    word, log.fallback = self._fallback(obs), True
                    break
                bad_retries += 1
            messages += [{"role": "assistant", "content": reply.text},
                         {"role": "user", "content": message}]
        if self.retries == 0 and not is_consistent(word, obs["guesses"], obs["feedback"], obs["n_lies"]):
            log.inconsistent_played = True
        log.guess = word
        log.seconds = round(time.perf_counter() - start, 2)
        self.turns.append(log)
        return word
