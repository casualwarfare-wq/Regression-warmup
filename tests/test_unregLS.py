import numpy as np
from numpy.polynomial import Polynomial
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from src.unregLS import Train_CV_ratio_predictions


def _prediction_model():
    return Pipeline(
        [
            ("poly", PolynomialFeatures()),
            ("scale", StandardScaler()),
            ("linreg", LinearRegression()),
        ]
    )


def test_train_cv_ratios_match_noise_only_expectations_for_cubic_degree():
    X = np.arange(48, dtype=float).reshape(-1, 1)
    y = np.zeros(48)
    sigma = 2.5
    degree = 3
    n_fold_train = 38
    p = degree + 1

    train_ratios, cv_ratios, cv_train_ratios = Train_CV_ratio_predictions(
        [("small", X, y, degree, sigma)],
        _prediction_model(),
        np.array([8.0, 6.0, 4.0, 0.3]),
        KFold(n_splits=5),
    )

    expected_train = np.sqrt(1 - p / n_fold_train)
    expected_cv = np.sqrt(1 + p / n_fold_train)
    np.testing.assert_allclose(train_ratios, [expected_train])
    np.testing.assert_allclose(cv_ratios, [expected_cv])
    np.testing.assert_allclose(cv_train_ratios, [expected_cv - expected_train])


def test_train_cv_ratios_include_degree_one_approximation_bias():
    X = np.arange(48, dtype=float).reshape(-1, 1)
    y = np.zeros(48)
    sigma = 2.5
    degree = 1
    n_fold_train = 38
    p = degree + 1
    true_coefs = np.array([8.0, 6.0, 4.0, 0.3])
    x_dense = np.linspace(0, 20, 10_001)
    y_dense = np.polynomial.polynomial.polyval(x_dense, true_coefs)
    best_line = Polynomial.fit(x_dense, y_dense, deg=degree)
    bias = np.sqrt(np.mean((y_dense - best_line(x_dense)) ** 2))
    bias_ratio_squared = (bias / sigma) ** 2

    train_ratios, cv_ratios, cv_train_ratios = Train_CV_ratio_predictions(
        [("small", X, y, degree, sigma)],
        _prediction_model(),
        true_coefs,
        KFold(n_splits=5),
    )

    expected_train = np.sqrt(1 - p / n_fold_train + bias_ratio_squared)
    expected_cv = np.sqrt(1 + p / n_fold_train + bias_ratio_squared)
    np.testing.assert_allclose(train_ratios, [expected_train])
    np.testing.assert_allclose(cv_ratios, [expected_cv])
    np.testing.assert_allclose(cv_train_ratios, [expected_cv - expected_train])
