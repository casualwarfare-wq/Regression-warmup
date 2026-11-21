import numpy as np

from src.poly import realized_noise_rms


def test_realized_noise_rms_matches_hand_computed_residuals():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    true_coefs = np.array([1.0, 2.0])
    y = np.array([4.0, 7.0, 5.0, 7.0])

    # The residuals are [3, 4, 0, 0], so RMS = sqrt((9 + 16) / 4) = 2.5.
    assert realized_noise_rms(X, y, true_coefs) == 2.5


def test_realized_noise_rms_includes_nonzero_residual_mean():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    true_coefs = np.array([1.0, 2.0])
    y = np.array([7.0, 9.0, 11.0, 13.0])

    # A constant residual of 6 has RMS 6 but standard deviation 0.
    assert realized_noise_rms(X, y, true_coefs) == 6.0


def test_realized_noise_rms_is_zero_for_noise_free_data():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    true_coefs = np.array([1.0, 2.0])
    y = np.array([1.0, 3.0, 5.0, 7.0])

    assert realized_noise_rms(X, y, true_coefs) == 0.0
