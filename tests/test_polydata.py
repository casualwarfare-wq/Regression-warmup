from src.poly import make_poly_data
import numpy as np
from numpy.random import default_rng
from numpy.polynomial.polynomial import polyval
import src.poly as poly


def test_shape_data():
    X, y = make_poly_data(n=100, seed=42, x_low=0, x_high=10, true_coefs=np.array([1, 2, 3]), noise=5)
    assert X.shape == (100, 1)
    assert y.shape == (100,)

    X_2, y_2 = make_poly_data(n=500, seed=42, x_low=0, x_high=10, true_coefs=np.array([1, 2, 3]), noise=5)
    assert X_2.shape == (500, 1)
    assert y_2.shape == (500,)


def test_seed_checks():
    X_1, y_1 = make_poly_data(n=100, seed=42, x_low=0, x_high=10, true_coefs=np.array([1, 2, 3]), noise=5)
    X_2, y_2 = make_poly_data(n=100, seed=42, x_low=0, x_high=10, true_coefs=np.array([1, 2, 3]), noise=5)
    assert np.array_equal(X_1, X_2)
    assert np.array_equal(y_1, y_2)

    X_3, y_3 = make_poly_data(n=100, seed=43, x_low=0, x_high=10, true_coefs=np.array([1, 2, 3]), noise=5)
    assert not np.array_equal(X_1, X_3)
    assert not np.array_equal(y_1, y_3)


def test_x_range_checks():
    X_1, y_1 = make_poly_data(n=100, seed=42, x_low=0, x_high=10, true_coefs=np.array([1, 2, 3]), noise=5)
    assert np.all(X_1 <= 10)
    assert np.all(X_1 >= 0)

    X_2, y_2 = make_poly_data(n=100, seed=42, x_low=-5, x_high=5, true_coefs=np.array([1, 2, 3]), noise=5)
    assert np.all(X_2 <= 5)
    assert np.all(X_2 >= -5)


def test_true_coefs_checks():
    X, y = make_poly_data(n=100, seed=42, x_low=5, x_high=5, true_coefs=np.array([1, 2, 3]), noise=0)
    x_flat = X.ravel()
    expected = 1 + 2 * x_flat + 3 * x_flat**2
    np.testing.assert_allclose(y, expected)

    X, y = make_poly_data(n=100, seed=42, x_low=5, x_high=5, true_coefs=np.array([0, 0, 1]), noise=0)
    x_flat = X.ravel()
    np.testing.assert_allclose(y, x_flat**2)


def test_coefficients_are_applied_in_ascending_power_order():
    coefficients = np.array([2.0, -3.0, 5.0, 7.0])
    X, y = make_poly_data(
        n=8,
        seed=17,
        x_low=-2.0,
        x_high=2.0,
        true_coefs=coefficients,
        noise=0,
    )
    x = X.ravel()
    expected = 2.0 - 3.0 * x + 5.0 * x**2 + 7.0 * x**3
    reversed_order = 7.0 + 5.0 * x - 3.0 * x**2 + 2.0 * x**3

    np.testing.assert_allclose(y, expected)
    assert not np.allclose(expected, reversed_order)


def test_noise_std():
    X, y = make_poly_data(n=500, seed=42, x_low=0, x_high=10, true_coefs=np.array([1, 2, 3]), noise=5)
    X_1, y_1 = make_poly_data(n=500, seed=42, x_low=0, x_high=10, true_coefs=np.array([1, 2, 3]), noise=0)
    noise = y - y_1
    standard_error = 5 / np.sqrt(2 * len(noise))
    assert np.isclose(noise.std(), 5, atol=standard_error)


def test_seeded_noise_uses_the_generated_normal_values_without_sign_change():
    n = 6
    seed = 12
    x_low, x_high = -1.0, 2.0
    coefficients = np.array([1.0, -2.0, 0.5])
    noise = 3.0
    rng = default_rng(seed)
    expected_x = rng.uniform(x_low, x_high, n)
    expected_y = polyval(c=coefficients, x=expected_x)
    expected_y += rng.normal(scale=noise, size=n)

    X, y = poly.make_poly_data(
        n=n,
        seed=seed,
        x_low=x_low,
        x_high=x_high,
        true_coefs=coefficients,
        noise=noise,
    )

    np.testing.assert_array_equal(X[:, 0], expected_x)
    np.testing.assert_array_equal(y, expected_y)


def test_make_poly_data_returns_a_column_vector_for_x():
    X, y = poly.make_poly_data(
        n=7,
        seed=1,
        x_low=0,
        x_high=1,
        true_coefs=np.array([1.0]),
        noise=0,
    )

    assert X.shape == (7, 1)
    assert y.shape == (7,)
