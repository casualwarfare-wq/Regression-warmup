import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures, StandardScaler

import src.poly as poly


def _hand_built_columns(_):
    return np.array(
        [
            [1, -1, 0, 0],
            [-1, 0, 1, 0],
            [0, 0, -1, 1],
        ],
        dtype=float,
    )


def test_placement_returns_variances_condition_and_max_absolute_correlation():
    pipeline = Pipeline(
        [
            ("known_columns", FunctionTransformer(_hand_built_columns)),
            ("ridge", Ridge()),
        ]
    )
    X_train = np.arange(4).reshape(-1, 1)
    y_train = np.zeros(4)

    variances, condition, off_diagonal_correlation = poly.placement(
        pipeline, X_train, y_train
    )

    assert np.allclose(variances, [2 / 3, 2 / 9, 2 / 3, 2 / 9])
    assert np.isclose(condition, 1 + np.sqrt(2))
    assert np.isclose(off_diagonal_correlation, np.sqrt(3) / 2)


def test_placement_correlates_columns_not_rows():
    transformed = np.array(
        [
            [3, 0, -3],
            [3, 0, 0],
            [-1, -3, 4],
            [3, 1, -1],
            [3, 0, -1],
        ],
        dtype=float,
    )
    pipeline = Pipeline(
        [
            ("known_columns", FunctionTransformer(lambda _: transformed)),
            ("ridge", Ridge()),
        ]
    )
    X_train = np.arange(5).reshape(-1, 1)
    y_train = np.zeros(5)

    _, _, off_diagonal_correlation = poly.placement(pipeline, X_train, y_train)

    assert np.isclose(off_diagonal_correlation, 0.9583727150068312)


def test_placement_does_not_mutate_training_data():
    pipeline = Pipeline(
        [
            ("identity", FunctionTransformer()),
            ("ridge", Ridge()),
        ]
    )
    X_train = np.array([[1.0, 3.0], [2.0, 5.0], [4.0, 8.0]])
    y_train = np.array([2.0, 4.0, 7.0])
    X_before = X_train.copy()
    y_before = y_train.copy()

    poly.placement(pipeline, X_train, y_train)

    np.testing.assert_array_equal(X_train, X_before)
    np.testing.assert_array_equal(y_train, y_before)


def test_placement_does_not_fit_the_callers_pipeline():
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("polynomial", PolynomialFeatures(degree=2)),
            ("ridge", Ridge()),
        ]
    )
    X_train = np.arange(4, dtype=float).reshape(-1, 1)
    y_train = np.zeros(4)

    assert not hasattr(pipeline.named_steps["scaler"], "mean_")
    poly.placement(pipeline, X_train, y_train)

    assert not hasattr(pipeline.named_steps["scaler"], "mean_")
    assert not hasattr(pipeline.named_steps["polynomial"], "powers_")


def test_placement_returns_zero_when_only_one_column_varies():
    pipeline = Pipeline(
        [
            (
                "known_columns",
                FunctionTransformer(
                    lambda X: np.column_stack(
                        [np.ones(len(X)), np.asarray(X).reshape(-1)]
                    )
                ),
            ),
            ("ridge", Ridge()),
        ]
    )
    X_train = np.arange(3, dtype=float).reshape(-1, 1)
    y_train = np.zeros(3)
    _, condition, off_diagonal_correlation = poly.placement(pipeline, X_train, y_train)

    assert condition == 0
    assert off_diagonal_correlation == 0


def test_placement_computes_condition_from_nonzero_variance_columns():
    transformed = np.array(
        [
            [0, 0, 1],
            [0, 1, 0],
            [0, 2, 1],
            [0, 3, 2],
        ],
        dtype=float,
    )
    pipeline = Pipeline(
        [
            ("known_columns", FunctionTransformer(lambda _: transformed)),
            ("ridge", Ridge()),
        ]
    )
    X_train = np.arange(4).reshape(-1, 1)
    y_train = np.zeros(4)

    variances, condition, _ = poly.placement(pipeline, X_train, y_train)

    nonzero_variance_columns = variances > 0
    expected_condition = np.linalg.cond(transformed[:, nonzero_variance_columns])
    assert np.isfinite(condition)
    assert np.isclose(condition, expected_condition)


def test_placement_transforms_all_steps_before_the_estimator():
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("polynomial", PolynomialFeatures(degree=2)),
            ("ridge", Ridge()),
        ]
    )
    X_train = np.arange(5, dtype=float).reshape(-1, 1)
    y_train = np.zeros(5)

    variances, _, _ = poly.placement(pipeline, X_train, y_train)

    assert variances.shape == (3,)
