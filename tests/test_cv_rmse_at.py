import numpy as np
import pytest

from src.poly import cv_rmse_at


def test_cv_rmse_at_returns_hand_computed_mean_and_standard_deviation():
    alphas = np.array([0.1, 1.0])
    scores = np.array([[-2.0, -4.0, -6.0], [-10.0, -12.0, -14.0]])
    mean_rmse, std_rmse = cv_rmse_at((alphas, None, scores), alpha=0.1, scoring='neg_root_mean_squared_error')
    mean_rmse2, std_rmse2 = cv_rmse_at((alphas, None, scores), alpha=1.0, scoring='neg_root_mean_squared_error')
    np.testing.assert_allclose(mean_rmse, 4.0)
    np.testing.assert_allclose(std_rmse, np.sqrt(8.0 / 3.0))
    # The selected second-row RMSEs are [10, 12, 14].
    np.testing.assert_allclose(mean_rmse2, 12.0)
    np.testing.assert_allclose(std_rmse2, np.sqrt(8.0 / 3.0))


def test_cv_rmse_at_returns_positive_values():
    alphas = np.array([0.1])
    scores = np.array([[-2.0, -4.0, -6.0]])

    mean_rmse, std_rmse = cv_rmse_at((alphas, None, scores), alpha=0.1, scoring='neg_root_mean_squared_error')

    assert mean_rmse > 0
    assert std_rmse > 0


def test_cv_rmse_at_raises_when_alpha_is_not_in_grid():
    alphas = np.array([0.1, 1.0])
    scores = np.array([[-2.0, -4.0], [-6.0, -8.0]])

    with pytest.raises(ValueError):
        cv_rmse_at((alphas, None, scores), alpha=0.5, scoring='neg_root_mean_squared_error')


def test_cv_rmse_at_returns_correct_mean_and_std():
    alphas = np.array([0.1, 0.5])
    results = np.array([[-2,-6,-8,-4,-3],[-7,-3,-8,-2,-4]])
    mean, std = cv_rmse_at((alphas,None,results), alpha=0.5, ddof=1, scoring='neg_root_mean_squared_error')
    mean_0, std_0 = cv_rmse_at((alphas,None,results), alpha=0.5, ddof=0, scoring='neg_root_mean_squared_error')
    assert np.isclose(mean, 4.8)
    assert np.isclose(std, np.sqrt(6.7))
    assert np.isclose(mean_0, 4.8)
    assert np.isclose(std_0, np.sqrt(5.36))


def test_cv_rmse_at_returns_std_0_when_folds_identical():
    results = np.array([[-2,-2,-2,-2,-2],[-4,-4,-4,-4,-4]])
    alphas = np.array([0.1,0.5])
    assert cv_rmse_at((alphas,None,results), 0.1, ddof=1, scoring='neg_root_mean_squared_error')[1] == 0
    assert cv_rmse_at((alphas,None,results), 0.5, ddof=1, scoring='neg_root_mean_squared_error')[1] == 0


def test_cv_rmse_at_raise_error_when_fold_less_than_or_equal_to_ddof():
    results = np.array([[-1],[-2]])
    alphas=np.array([0.1,0.5])
    with pytest.raises(ValueError, match="Number of folds must be greater than ddof."):
        cv_rmse_at((alphas, None, results), 0.1, ddof=1, scoring='neg_root_mean_squared_error')

@pytest.mark.parametrize("scoring", ["r2", "neg_mean_squared_error"])
def test_cv_rmse_at_raises_error_when_scoring_not_rmse(scoring):
    results=np.array([[-1,-3,-56,-34],[-3,-5,-34,-4]])
    alphas=[0.1,0.5]
    with pytest.raises(ValueError, match='scoring must be rmse.'):
        cv_rmse_at((alphas,None,results), 0.1,scoring=scoring)


@pytest.mark.parametrize("score", [0.5, 2.0])
def test_cv_rmse_at_rejects_nonnegative_fold_scores(score):
    with pytest.raises(ValueError, match="Scores need to be negative."):
        cv_rmse_at(
            (np.array([0.1]), None, np.array([[score, -1.0]])),
            0.1,
            scoring="neg_root_mean_squared_error",
        )


def test_cv_rmse_at_allows_zero_fold_scores():
    mean, _ = cv_rmse_at(
        (np.array([0.1]), None, np.array([[0.0, -1.0]])),
        0.1,
        scoring="neg_root_mean_squared_error",
    )

    assert mean == pytest.approx(0.5)


def test_cv_rmse_at_accepts_negative_scores_between_minus_one_and_zero():
    mean, _ = cv_rmse_at(
        (np.array([0.1]), None, np.array([[-0.5, -0.75]])),
        0.1,
        scoring="neg_root_mean_squared_error",
    )

    assert mean == pytest.approx(0.625)


def test_cv_rmse_at_accepts_equal_runtime_scoring_string():
    scoring = "".join(("neg_root_mean_", "squared_error"))
    expected_scoring = "neg_root_mean_squared_error"

    assert scoring == expected_scoring
    assert scoring is not expected_scoring
    mean, _ = cv_rmse_at(
        (np.array([0.1]), None, np.array([[-1.0, -3.0]])),
        0.1,
        scoring=scoring,
    )

    assert mean == pytest.approx(2.0)


def test_cv_rmse_at_rejects_fold_count_below_ddof():
    with pytest.raises(ValueError, match="Number of folds must be greater than ddof."):
        cv_rmse_at(
            (np.array([0.1]), None, np.array([[-1.0, -2.0]])),
            0.1,
            ddof=3,
            scoring="neg_root_mean_squared_error",
        )
