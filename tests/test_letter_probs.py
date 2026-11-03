import numpy as np

from wordle_fibble.letter_probs import letter_position_probs, load_csv, save_csv
from wordle_fibble.words import load_answers

# A few cells of the paper's Table I, as (letter, position 1-5, value).
PAPER_CELLS = [
    ("a", 1, 0.0514), ("a", 2, 0.1615), ("e", 4, 0.2133), ("o", 2, 0.1582),
    ("s", 1, 0.1258), ("s", 5, 0.3064), ("y", 5, 0.1155), ("c", 1, 0.0764),
]


def test_shape_and_rows_sum_to_one():
    probs = letter_position_probs()
    assert probs.shape == (5, 26)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_close_to_paper_table():
    probs = letter_position_probs()
    for letter, pos, value in PAPER_CELLS:
        assert abs(probs[pos - 1, ord(letter) - 97] - value) < 0.035, (letter, pos)


def test_counts_small_list():
    probs = letter_position_probs(["crane", "crate", "slate"])
    assert probs[0, ord("c") - 97] == 2 / 3
    assert probs[4, ord("e") - 97] == 1.0


def test_answers_list_differs_from_paper():
    # Why we use the allowed list: the answer list has almost no plurals.
    assert letter_position_probs(load_answers())[4, ord("s") - 97] < 0.05


def test_csv_round_trip(tmp_path):
    probs = letter_position_probs()
    save_csv(probs, tmp_path / "t.csv")
    assert np.allclose(load_csv(tmp_path / "t.csv"), probs, atol=5e-5)
