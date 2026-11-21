import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.metrics import get_scorer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

import src.poly as poly


@pytest.fixture
def regression_data():
    X, y = poly.make_poly_data(
        n=48,
        seed=17,
        x_low=-2.0,
        x_high=2.0,
        true_coefs=np.array([1.0, 2.0, 3.0]),
        noise=0.2,
    )
    return train_test_split(X, y, test_size=0.25, random_state=23)


@pytest.mark.parametrize(
    "alpha",
    [
        pytest.param([0.5], id="list"),
        pytest.param((0.5,), id="tuple"),
        pytest.param(np.array([0.5]), id="array"),
    ],
)
def test_collection_alphas_raise_type_error(regression_data, alpha):
    X_train, X_test, y_train, y_test = regression_data

    with pytest.raises(TypeError):
        poly.final_test_score(
            Ridge(), X_train, y_train, X_test, y_test, alpha
        )


@pytest.mark.parametrize(
    "alpha",
    [
        pytest.param(float("nan"), id="nan"),
        pytest.param(np.inf, id="positive-infinity"),
        pytest.param(-np.inf, id="negative-infinity"),
        pytest.param(True, id="boolean"),
        pytest.param(1 + 2j, id="complex"),
        pytest.param("0.5", id="string"),
    ],
)
def test_invalid_scalar_alphas_raise_type_error(regression_data, alpha):
    X_train, X_test, y_train, y_test = regression_data

    with pytest.raises(TypeError, match="best_alpha must be a scalar float, not a list-like value, boolean, inf or nan."):
        poly.final_test_score(
            Ridge(), X_train, y_train, X_test, y_test, alpha
        )
def test_negative_alphas_raise_value_error(regression_data):
    X_train, X_test, y_train, y_test = regression_data
    with pytest.raises(ValueError, match="Alphas can't be negative."):
        poly.final_test_score(Ridge(),X_train, y_train, X_test, y_test, -0.5)


def test_zero_alpha_is_allowed(regression_data):
    X_train, X_test, y_train, y_test = regression_data

    score = poly.final_test_score(
        Ridge(), X_train, y_train, X_test, y_test, 0.0
    )

    assert np.isfinite(score)


def test_unsupported_model_raises_Type_error(regression_data):
    X_train, X_test, y_train, y_test = regression_data

    with pytest.raises(TypeError):
        poly.final_test_score(
            object(), X_train, y_train, X_test, y_test, 0.5
        )


@pytest.mark.parametrize(
    "model_type",
    [
        pytest.param("ridge", id="ridge"),
        pytest.param("pipeline", id="pipeline"),
    ],
)
def test_caller_model_is_unchanged(regression_data, model_type):
    X_train, X_test, y_train, y_test = regression_data
    if model_type == "ridge":
        model = Ridge(alpha=2.0)
        alpha_parameter = "alpha"
    else:
        model = Pipeline(
            [
                ("polynomial", PolynomialFeatures(degree=2)),
                ("scaler", StandardScaler()),
                ("rid", Ridge(alpha=2.0)),
            ]
        )
        alpha_parameter = "rid__alpha"

    poly.final_test_score(
        model, X_train, y_train, X_test, y_test, 0.25
    )

    assert model.get_params()[alpha_parameter] == 2.0


def test_score_matches_independently_built_pipeline(regression_data):
    X_train, X_test, y_train, y_test = regression_data
    alpha = 0.25
    model = Pipeline(
        [
            ("polynomial", PolynomialFeatures(degree=2)),
            ("scaler", StandardScaler()),
            ("ridger", Ridge(alpha=3.0)),
        ]
    )
    expected_model = Pipeline(
        [
            ("polynomial", PolynomialFeatures(degree=2)),
            ("scaler", StandardScaler()),
            ("ridger", Ridge(alpha=alpha)),
        ]
    )

    actual_score = poly.final_test_score(
        model, X_train, y_train, X_test, y_test, alpha
    )
    expected_model.fit(X_train, y_train)
    expected_score = expected_model.score(X_test, y_test)

    assert actual_score == pytest.approx(expected_score)


def test_score_matches_independently_built_named_scorer(regression_data):
    X_train, X_test, y_train, y_test = regression_data
    alpha = 0.25
    scoring = "neg_mean_squared_error"
    model = Ridge(alpha=2.0)
    expected_model = Ridge(alpha=alpha)

    actual_score = poly.final_test_score(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        alpha,
        scoring=scoring,
    )
    expected_model.fit(X_train, y_train)
    scorer = get_scorer(scoring)
    expected_score = abs(scorer(expected_model, X_test, y_test))

    assert abs(actual_score) == pytest.approx(expected_score)
    assert np.sign(actual_score) != np.sign(expected_score)

def test_negative_r2_test(regression_data):
    X_train, X_test, y_train, y_test = regression_data
    X2,y2=poly.make_poly_data(n=60,seed=42, x_low=-200, x_high= 4000, true_coefs=[5,4,3], noise =500)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X2,y2, test_size=0.3, random_state=42)
    model= Ridge(alpha=1)
    actual_score = poly.final_test_score(model, X_train, y_train, X_test2, y_test2, 0.25)
    assert np.sign(actual_score) == np.sign(-1)


