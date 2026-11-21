from src.poly import build_pipelines
from src.poly import make_poly_data
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
import numpy as np
from sklearn.model_selection import train_test_split
import pytest
def assert_step_order(pipeline, expected_steps):
    actual_names = [name for name, _ in pipeline.steps]
    expected_names = [name for name, _ in expected_steps]

    for (_, actual_step), (_, expected_step) in zip(pipeline.steps, expected_steps):
        assert isinstance(actual_step, type(expected_step))
    assert len(pipeline.steps) == len(expected_steps)


def test_keys():
    pipelines = build_pipelines(degree=8)
    assert pipelines.keys() == {"scale_first", "scale_afterpoly", "scale_both"}


def test_steps():
    pipelines = build_pipelines(degree=8)
    assert_step_order(
        pipelines["scale_first"],
        [("scaler", StandardScaler()), ("polynomial", PolynomialFeatures(degree=8)), ("ridge", Ridge())],
    )
    assert_step_order(
        pipelines["scale_afterpoly"],
        [("polynomial", PolynomialFeatures(degree=8)), ("scaler", StandardScaler()), ("ridge", Ridge())],
    )
    assert_step_order(
        pipelines["scale_both"], [("scaler_before", StandardScaler()), ("polynomial", PolynomialFeatures(degree=8)), ("scaler_after", StandardScaler()), ("ridge", Ridge())])

def test_degree():
    pipelines=build_pipelines(degree=5)
    for name, pipeline in pipelines.items():
        assert pipeline.get_params()["polynomial__degree"] == 5
    pipelines = build_pipelines(degree=12)
    for name, pipeline in pipelines.items():
        assert pipeline.get_params()["polynomial__degree"] == 12
    
    with pytest.raises(TypeError):
        pipelines = build_pipelines(degree=True)

def test_independence():
    steps = [
        step
        for pipelines in (build_pipelines(degree=5), build_pipelines(degree=5))
        for pipeline in pipelines.values()
        for _, step in pipeline.steps
    ]

    assert len(steps) == len({id(step) for step in steps})

def test_alpha():
    pipelines = build_pipelines(degree=8)
    for name, pipeline in pipelines.items():
        assert "ridge__alpha" in pipeline.get_params()
def test_runtime(): 
    pipelines = build_pipelines(degree=8)
    X,y = make_poly_data(n=80, seed=42, x_low=0, x_high=20, true_coefs=np.array([8,6,4,0.3]), noise=200)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    for name, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)
        assert pipeline.predict(X_test).shape == (X_test.shape[0],)
def test_ridge():
    pipelines = build_pipelines(degree=8)
    for name, pipeline in pipelines.items():
        assert isinstance(pipeline.steps[-1][1], Ridge)

def test_fit_intercept():
    pipelines = build_pipelines(degree=8)
    for pipeline in pipelines.values():
        assert pipeline.get_params()["ridge__fit_intercept"] is True

def test_typeError():
    with pytest.raises(TypeError):
        build_pipelines(degree="invalid")
def test_bias():
    pipelines = build_pipelines(degree=8)
    for name, pipeline in pipelines.items():
        assert pipeline.get_params()["polynomial__include_bias"] == True
