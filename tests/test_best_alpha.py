import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.model_selection import KFold

from src.poly import best_alpha, build_pipelines, curve_for, make_poly_data


def test_best_alpha_uses_mean_not_max_min_or_median_across_folds():
    scores = np.array(
        [
            [0.6, 0.8, 0.8, 0.8, 0.8],
            [1.0, 0.5, 0.5, 0.5, 0.5],
            [0.7, 0.7, 0.7, 0.7, 0.7],
            [0.0, 0.0, 0.85, 0.9, 0.9],
        ]
    )
    alphas = np.array([0.01, 0.1, 1.0, 10.0])

    # Mean, max, min, and median choose rows 0, 1, 2, and 3, respectively.
    assert best_alpha((alphas, None, scores)) == 0.01


def test_best_alpha_returns_first_of_tied_best_rows():
    scores = np.array(
        [
            [0.9, 0.3, 0.6],
            [0.3, 0.9, 0.6],
            [0.0, 0.0, 0.9],
            [0.0, 0.0, 0.9],
        ]
    )
    alphas = np.array([0.01, 0.1, 1.0, 10.0])

    # The first two row means tie at 0.6; averaging columns instead picks index 2.
    assert best_alpha((alphas, None, scores)) == 0.01


def test_best_alpha_selects_smallest_alpha_from_tied_rows_on_unsorted_grid():
    alphas = np.array([10.0, 1.0, 0.1, 0.01])
    scores = np.array(
        [
            [0.9, 0.9],
            [0.9, 0.9],
            [0.5, 0.5],
            [0.1, 0.1],
        ]
    )

    # The top mean score is 0.9 for alpha 10.0 and 1.0, so the smaller
    # matching alpha, 1.0, should be returned even though 10.0 appears first.
    assert best_alpha((alphas, None, scores)) == 1.0


def test_best_alpha_selects_alpha_from_real_validation_curve():
    X, y = make_poly_data(
        n=40,
        seed=73,
        x_low=-2.0,
        x_high=2.0,
        true_coefs=np.array([1.0, 2.0, 3.0]),
        noise=5.0,
    )
    alphas = np.array([1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0])
    model = build_pipelines(degree=2)["scale_first"]
    splitter = KFold(n_splits=4, shuffle=True, random_state=73)
    result = curve_for(
        model, X, y, alphas, cv=splitter, scoring="r2", n_jobs=1
    )

    selected_alpha = best_alpha(result=result)
    mean_scores = result[2].mean(axis=1)
    sorted_mean_scores = np.sort(mean_scores)

    assert_allclose(result[0], alphas)
    # At this fixed seed, alpha 10.0 beats runner-up 100.0 by about 0.0634.
    # Require at least a 0.01 gap so floating-point noise cannot choose the winner.
    assert sorted_mean_scores[-1] - sorted_mean_scores[-2] > 0.01
    assert selected_alpha == 10.0


def test_best_alpha_chooses_first_alpha_after_sorting_tied_best_rows():
    alphas = np.array([10.0, 0.01, 1.0, 0.1])
    scores = np.array(
        [
            [0.9, 0.9, 0.9],
            [0.9, 0.9, 0.9],
            [0.9, 0.9, 0.9],
            [0.9, 0.9, 0.9],
        ]
    )

    # Every validation row ties at 0.9, so the correct policy is to take the
    # first alpha in the sorted grid: 0.01 < 0.1 < 1.0 < 10.0.
    assert best_alpha((alphas, None, scores)) == 0.01


@pytest.mark.parametrize(
    "invalid_alpha",
    [
        pytest.param(np.nan, id="nan"),
        pytest.param(np.inf, id="positive-infinity"),
        pytest.param(-np.inf, id="negative-infinity"),
        pytest.param("0.5", id="string"),
        pytest.param(1 + 2j, id="complex"),
    ],
)

def test_best_alpha_rejects_non_scalar_alphas(invalid_alpha):
    alphas = np.array([0.1, invalid_alpha])
    scores = np.array([[0.5, 0.5], [0.9, 0.9]])

    with pytest.raises(TypeError, match="Each alpha must be a scalar float or int."):
        best_alpha((alphas, None, scores))


def test_best_alpha_rejects_negative_alphas():
    alphas = np.array([0.1, -0.5])
    scores = np.array([[0.5, 0.5], [0.9, 0.9]])

    with pytest.raises(ValueError, match="All Alphas must be non-negative."):
        best_alpha((alphas, None, scores))
def test_best_alpha_accepts_zero_alpha():
    alphas = np.array([0.0, 0.1])
    scores = np.array([[0.9, 0.9], [0.5, 0.5]])

    assert best_alpha((alphas, None, scores)) == 0.0


def test_best_alpha_selects_exactly_the_best_mean_score():
    alphas = np.array([0.1, 1.0, 10.0])
    scores = np.array([[0.1, 0.2], [0.8, 0.8], [0.4, 0.5]])

    assert best_alpha((alphas, None, scores)) == 1.0
@pytest.mark.parametrize(
    "boolean_alphas",
    [
        pytest.param(True, id="true"),
        pytest.param(False, id="false")
    ])

def test_best_alpha_rejects_boolean_alphas(boolean_alphas):
    alphas = [0.1, boolean_alphas]
    scores = np.array([[0.5, 0.5], [0.9, 0.9]])

    with pytest.raises(TypeError, match="Alphas can't be boolean."):
        best_alpha((alphas, None, scores))
