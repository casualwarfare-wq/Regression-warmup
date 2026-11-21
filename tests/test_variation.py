import src.poly as poly
import numpy as np
def test_correct_variance():
    X = np.array([[7],[9],[4], [18]])
    X_poly = poly.variation(X)
    assert np.isclose(X_poly[0], 0, atol=0.01)
    assert np.isclose(X_poly[1], 27.25, atol=0.01)
    assert np.isclose(X_poly[2], 14742.25, atol=0.01)
    assert np.isclose(X_poly[3], 5631788.5, atol=0.01)


def test_variation_default_degree_is_three():
    X = np.array([[0.0], [1.0], [2.0]])

    assert poly.variation(X).shape == (4,)
