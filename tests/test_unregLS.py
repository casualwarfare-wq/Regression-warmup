import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from src.unregLS import Train_CV_Ratio_Predictions


def _prediction_model():
    return Pipeline(
        [
            ("poly", PolynomialFeatures()),
            ("scale", StandardScaler()),
            ("linreg", LinearRegression()),
        ]
    )


def test_cubic_degree_has_no_bias_and_matches_fixed_ratios():
    X = np.arange(48, dtype=float).reshape(-1, 1)
    y = np.zeros(48)
    sigma = 2.5
    degree = 3

    train_ratios, cv_ratios, cv_train_ratios = Train_CV_Ratio_Predictions(
        [(X, y, degree, sigma)],
        _prediction_model(),
        np.array([8.0, 6.0, 4.0, 0.3]),
        KFold(n_splits=5),
    )

    np.testing.assert_allclose(train_ratios, [np.sqrt(34 / 38)], rtol=1e-12)
    np.testing.assert_allclose(cv_ratios, [np.sqrt(42 / 38)], rtol=1e-12)
    np.testing.assert_allclose(cv_train_ratios, [np.sqrt(42 / 38) - np.sqrt(34 / 38)], rtol=1e-12)


def test_degree_one_approximation_has_fixed_bias_rms():
    X = np.arange(48, dtype=float).reshape(-1, 1)
    y = np.zeros(48)
    sigma = 2.5
    degree = 1
    true_coefs = np.array([8.0, 6.0, 4.0, 0.3])

    train_ratios, cv_ratios, cv_train_ratios = Train_CV_Ratio_Predictions(
        [(X, y, degree, 144.98200701200935)],
        _prediction_model(),
        true_coefs,
        KFold(n_splits=5),
    )

    np.testing.assert_allclose(train_ratios, [2.862666540432592], rtol=1e-12)
    np.testing.assert_allclose(cv_ratios, [2.880993384165788], rtol=1e-12)
    np.testing.assert_allclose(cv_train_ratios, [0.018326843733196], rtol=1e-12)


def test_degree_two_approximation_has_fixed_bias_rms():
    X = np.arange(48, dtype=float).reshape(-1, 1)
    y = np.zeros(48)

    train_ratios, cv_ratios, cv_train_ratios = Train_CV_Ratio_Predictions(
        [(X, y, 2, 2.5)],
        _prediction_model(),
        np.array([8.0, 6.0, 4.0, 0.3]),
        KFold(n_splits=5),
    )

    np.testing.assert_allclose(train_ratios, [np.sqrt(1 - 3 / 38 + (45.369341667634 / 2.5) ** 2)])
    np.testing.assert_allclose(cv_ratios, [np.sqrt(1 + 3 / 38 + (45.369341667634 / 2.5) ** 2)])
    np.testing.assert_allclose(cv_train_ratios, [cv_ratios[0] - train_ratios[0]])


def test_quartic_degree_has_zero_bias():
    X = np.arange(48, dtype=float).reshape(-1, 1)
    y = np.zeros(48)

    train_ratios, cv_ratios, cv_train_ratios = Train_CV_Ratio_Predictions(
        [(X, y, 4, 2.5)],
        _prediction_model(),
        np.array([8.0, 6.0, 4.0, 0.3]),
        KFold(n_splits=5),
    )

    np.testing.assert_allclose(train_ratios, [np.sqrt(1 - 5 / 38)])
    np.testing.assert_allclose(cv_ratios, [np.sqrt(1 + 5 / 38)])
    np.testing.assert_allclose(cv_train_ratios, [cv_ratios[0] - train_ratios[0]])


def test_negative_sigma_is_treated_as_an_absolute_noise_floor():
    X = np.arange(48, dtype=float).reshape(-1, 1)
    y = np.zeros(48)

    positive = Train_CV_Ratio_Predictions(
        [(X, y, 3, 2.5)], _prediction_model(),
        np.array([8.0, 6.0, 4.0, 0.3]), KFold(n_splits=5),
    )
    negative = Train_CV_Ratio_Predictions(
        [(X, y, 3, -2.5)], _prediction_model(),
        np.array([8.0, 6.0, 4.0, 0.3]), KFold(n_splits=5),
    )

    for positive_list, negative_list in zip(positive, negative):
        np.testing.assert_allclose(negative_list, positive_list)


def test_zero_sigma_is_rejected():
    X = np.arange(48, dtype=float).reshape(-1, 1)
    y = np.zeros(48)

    with pytest.raises(ZeroDivisionError, match="Sigma can't be zero"):
        Train_CV_Ratio_Predictions(
            [(X, y, 3, 0)], _prediction_model(),
            np.array([8.0, 6.0, 4.0, 0.3]), KFold(n_splits=5),
        )


class _FixedTrainSizeSplitter:
    def __init__(self, n_fold_train):
        self.n_fold_train = n_fold_train

    def split(self, X, y=None):
        yield np.arange(self.n_fold_train), np.array([], dtype=int)


@pytest.mark.parametrize("degree,n_fold_train", [(3, 4), (12, 12)])
def test_parameter_count_at_or_above_fold_training_size_is_rejected(degree, n_fold_train):
    X = np.arange(max(n_fold_train, 48), dtype=float).reshape(-1, 1)
    y = np.zeros(len(X))

    with pytest.raises(ValueError, match="smaller than"):
        Train_CV_Ratio_Predictions(
            [(X, y, degree, 2.5)], _prediction_model(),
            np.array([8.0, 6.0, 4.0, 0.3]),
            _FixedTrainSizeSplitter(n_fold_train),
        )


def test_large_n_zero_bias_ratios_approach_one():
    X = np.zeros((1, 1))
    y = np.zeros(1)

    train_ratios, cv_ratios, cv_train_ratios = Train_CV_Ratio_Predictions(
        [(X, y, 3, 200.0)], _prediction_model(),
        np.array([8.0, 6.0, 4.0, 0.3]),
        _FixedTrainSizeSplitter(1_000_000),
    )

    np.testing.assert_allclose(train_ratios, [np.sqrt(1 - 4 / 1_000_000)], rtol=1e-10)
    np.testing.assert_allclose(cv_ratios, [np.sqrt(1 + 4 / 1_000_000)], rtol=1e-10)
    np.testing.assert_allclose(cv_train_ratios, [cv_ratios[0] - train_ratios[0]], rtol=1e-10)
