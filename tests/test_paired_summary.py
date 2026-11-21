import warnings

import numpy as np
import pytest

from src.poly import paired_summary


DIFFS = [
    ("A", "B", -1.0),
    ("A", "B", -2.0),
    ("A", "B", -0.5),
    ("A", "B", -1.5),
    ("A", "B", 12.0),
    ("A", "B", 0.0),
]


def test_paired_summary_counts_negative_wins_and_excludes_ties():
    mean, _, a_wins = paired_summary(DIFFS)

    assert mean > 0
    assert a_wins == 4


def test_paired_summary_returns_mean_and_standard_error():
    mean, standard_error, _ = paired_summary(DIFFS)

    np.testing.assert_allclose(mean, 7.0 / 6.0)
    np.testing.assert_allclose(standard_error, np.sqrt(43.0) / 3.0)


def test_paired_summary_rejects_empty_differences():
    with pytest.raises(ValueError, match="No paired differences provided"):
        paired_summary([])


def test_paired_summary_rejects_single_difference_without_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="At least two paired differences"):
            paired_summary([("A", "B", -1.0)])

    assert caught == []


def test_paired_summary_rejects_mixed_pipeline_pairs():
    diffs = [
        ("A", "B", -1.0),
        ("C", "D", 2.5),
    ]

    with pytest.raises(ValueError, match="same pipeline pair"):
        paired_summary(diffs)


def test_paired_summary_rejects_mismatched_pair_that_sorts_before_first():
    diffs = [
        ("A", "B", -1.0),
        ("A", "A", 2.5),
    ]

    with pytest.raises(ValueError, match="same pipeline pair"):
        paired_summary(diffs)
