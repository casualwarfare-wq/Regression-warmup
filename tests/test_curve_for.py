import numpy as np
from numpy.testing import assert_allclose
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
import src.poly as poly


def test_curve_for_forwards_arguments(monkeypatch):
    model = object()
    X_train = np.array([[1.0], [2.0]])
    y_train = np.array([3.0, 4.0])
    alphas = np.array([0.01, 0.1, 1.0])
    expected = (np.array([[0.8, 0.9]]), np.array([[0.7, 0.85]]))
    captured = {}

    def fake_validation_curve(**kwargs):
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(poly, "validation_curve", fake_validation_curve)

    result = poly.curve_for(
        model,
        X_train,
        y_train,
        alphas,
        cv=4,
        parameter="ridge__alpha",
        scoring="r2",
    )

    assert result[0] is alphas
    assert result[1] is expected[0]
    assert result[2] is expected[1]
    assert captured == {
        "estimator": model,
        "X": X_train,
        "y": y_train,
        "param_name": "ridge__alpha",
        "param_range": alphas,
        "cv": 4,
        "n_jobs": -1,
        "scoring": "r2",
    }


def test_curve_for_default_cross_validation_count_is_five(monkeypatch):
    captured = {}

    def fake_validation_curve(**kwargs):
        captured.update(kwargs)
        return np.zeros((1, 5)), np.zeros((1, 5))

    monkeypatch.setattr(poly, "validation_curve", fake_validation_curve)
    alphas = np.array([0.1])

    poly.curve_for(object(), np.array([[1.0]]), np.array([1.0]), alphas)

    assert captured["cv"] == 5


def test_curve_for_returns_alpha_by_fold_scores():
    X, y = poly.make_poly_data(
        n=40,
        seed=73,
        x_low=-2.0,
        x_high=2.0,
        true_coefs=np.array([1.0, 2.0, 3.0]),
        noise=0.0,
    )
    alphas = np.array([1e-8, 1e-5, 1e-2])
    model = poly.build_pipelines(degree=2)["scale_first"]

    returned_alphas, train_scores, validation_scores = poly.curve_for(
        model, X, y, alphas, cv=4, scoring="r2", n_jobs=2
    )

    assert_allclose(returned_alphas, alphas)
    assert train_scores.shape == (len(alphas),4)
    assert validation_scores.shape == (len(alphas),4)


def test_curve_for_is_deterministic_with_fixed_seed_shuffled_folds():
    X, y = poly.make_poly_data(
        n=40,
        seed=73,
        x_low=-2.0,
        x_high=2.0,
        true_coefs=np.array([1.0, 2.0, 3.0]),
        noise=0.0,
    )
    splitter = ShuffleSplit(n_splits=4, test_size=0.25, random_state=42)
    alphas = np.array([1e-8, 1e-5, 1e-2])
    model = poly.build_pipelines(degree=2)["scale_first"]

    first = poly.curve_for(model, X, y, alphas, cv=splitter, scoring="r2", n_jobs=2)
    second = poly.curve_for(model, X, y, alphas, cv=splitter, scoring="r2", n_jobs=2)

    assert_allclose(first[0], second[0])
    assert_allclose(first[1], second[1])
    assert_allclose(first[2], second[2])


def test_curve_for_noise_free_polynomial_has_near_perfect_validation_score():
    X, y = poly.make_poly_data(
        n=40,
        seed=73,
        x_low=-2.0,
        x_high=2.0,
        true_coefs=np.array([1.0, 2.0, 3.0]),
        noise=0.0,
    )
    alphas = np.array([1e-10, 1e-8])
    model = poly.build_pipelines(degree=2)["scale_first"]

    _, train, validation_scores = poly.curve_for(
        model, X, y, alphas, cv=5, scoring="r2", n_jobs=2
    )

    assert np.min(validation_scores) > 0.99


def test_curve_for_preserves_descending_alpha_order_with_shuffled_folds():
    X, y = poly.make_poly_data(
        n=60,
        seed=21,
        x_low=-1.0,
        x_high=1.0,
        true_coefs=np.array([1.0, -2.0, 0.5]),
        noise=0.5,
    )
    model = poly.build_pipelines(degree=9)["scale_first"]
    ascending_alphas = np.array([1e-8, 1e-3, 1e-1])
    descending_alphas = ascending_alphas[::-1].copy()

    _, ascending_train, ascending_validation = poly.curve_for(
        model,
        X,
        y,
        ascending_alphas,
        cv=ShuffleSplit(n_splits=5, test_size=0.25, random_state=42),
        scoring="r2",
        n_jobs=1,
    )
    returned_alphas, descending_train, descending_validation = poly.curve_for(
        model,
        X,
        y,
        descending_alphas,
        cv=ShuffleSplit(n_splits=5, test_size=0.25, random_state=42),
        scoring="r2",
        n_jobs=1,
    )

    assert_allclose(returned_alphas, descending_alphas)
    assert_allclose(descending_train, ascending_train[::-1])
    assert_allclose(descending_validation, ascending_validation[::-1])


def test_curve_for_training_scores_exceed_validation_for_overfit_low_alpha():
    X, y = poly.make_poly_data(
        n=60,
        seed=21,
        x_low=-1.0,
        x_high=1.0,
        true_coefs=np.array([1.0, -2.0, 0.5]),
        noise=0.5,
    )
    alphas = np.array([1e-8, 1e-3])
    model = poly.build_pipelines(degree=9)["scale_first"]

    _, train_scores, validation_scores = poly.curve_for(
        model,
        X,
        y,
        alphas,
        cv=ShuffleSplit(n_splits=5, test_size=0.25, random_state=42),
        scoring="r2",
        n_jobs=1,
    )

    assert train_scores[0].mean() > validation_scores[0].mean()
