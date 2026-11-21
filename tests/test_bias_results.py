import numpy as np
import pytest
import inspect
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from src.poly import bias_results


def test_bias_results_matches_hand_calculated_linear_data():
    X_train = np.array([[0.0], [1.0], [2.0]])
    y_train = np.array([1.0, 3.0, 5.0])

    model = Pipeline([
        ("polynomial", PolynomialFeatures(degree=1)),
        ("ridge", Ridge(fit_intercept=False)),
    ])

    bias, max_difference = bias_results(
        model,
        "polynomial",
        "ridge",
        X_train,
        y_train,
        X_train,
        alphas=[1.0],
    )

    # At alpha=1, the full model's bias coefficient is 1.
    # The full-minus-no-bias prediction differences are [1, 0.5, 0].
    np.testing.assert_allclose(bias, 1.0)
    np.testing.assert_allclose(max_difference, 1.0)


def test_bias_results_uses_absolute_negative_prediction_difference():
    X_train = np.array([[0.0], [1.0], [2.0]])
    y_train = np.array([-1.0, 1.0, 3.0])
    X_test = np.array([[-3.0]])
    model = Pipeline(
        [
            ("polynomial", PolynomialFeatures(degree=1)),
            ("ridge", Ridge(fit_intercept=False)),
        ]
    )

    bias, max_difference = bias_results(
        model,
        "polynomial",
        "ridge",
        X_train,
        y_train,
        X_test,
        alphas=[1.0],
    )

    np.testing.assert_allclose(bias, 0.2)
    np.testing.assert_allclose(max_difference, 0.5)


def test_bias_results_returns_maxima_across_alphas():
    X_train = np.array([[0.0], [1.0], [2.0]])
    y_train = np.array([-1.0, 1.0, 3.0])
    X_test = np.array([[-3.0]])
    model = Pipeline(
        [
            ("polynomial", PolynomialFeatures(degree=1)),
            ("ridge", Ridge(fit_intercept=False)),
        ]
    )

    bias, max_difference = bias_results(
        model,
        "polynomial",
        "ridge",
        X_train,
        y_train,
        X_test,
        alphas=[1.0, 10.0],
    )

    np.testing.assert_allclose(bias, 1 / 5)
    np.testing.assert_allclose(max_difference, 1 / 2)


def test_bias_results_raises_Type_error_for_unsupported_model_type():
    with pytest.raises(TypeError, match="Unsupported model type"):
        bias_results(
            Ridge(),
            "polynomial",
            "ridge",
            np.array([[0.0], [1.0]]),
            np.array([0.0, 1.0]),
            np.array([[0.5]]),
        )


def test_bias_results_default_alpha_grid():
    assert inspect.signature(bias_results).parameters["alphas"].default == [
        0,
        0.001,
        0.01,
        0.1,
        1,
        5,
        10,
    ]


def test_bias_results_reads_coefficients_from_final_pipeline_step():
    X_train = np.array([[0.0], [1.0], [2.0], [3.0]])
    y_train = np.array([1.0, 3.0, 5.0, 7.0])
    model = Pipeline(
        [
            ("polynomial", PolynomialFeatures(degree=1)),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(fit_intercept=False)),
        ]
    )

    bias, difference = bias_results(
        model,
        "polynomial",
        "ridge",
        X_train,
        y_train,
        X_train,
        alphas=[1.0],
    )

    assert np.isfinite(bias)
    assert np.isfinite(difference)
    