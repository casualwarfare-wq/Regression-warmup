import numpy as np
from sklearn.model_selection import KFold

from src.poly import cv_rmse_per_fold


def test_cv_rmse_per_fold_averages_unequal_fold_rms_values():
    residuals = np.array([1.0, -3.0, 2.0, -4.0, 10.0, -20.0])
    splitter = KFold(n_splits=3, shuffle=False)

    splits = list(splitter.split(residuals))
    fold_indices = [test_indices for _, test_indices in splits]
    fold_rms = [
        np.sqrt(np.mean(residuals[test_indices] ** 2))
        for test_indices in fold_indices
    ]
    train_rms = [
        np.sqrt(np.mean(residuals[train_indices] ** 2))
        for train_indices, _ in splits
    ]
    fold_mean_residuals = [
        np.mean(residuals[test_indices]) for test_indices in fold_indices
    ]
    fold_mean_absolute_residuals = [
        np.mean(np.abs(residuals[test_indices])) for test_indices in fold_indices
    ]

    # The unequal fold RMS values make the mean distinct from the median.
    assert [indices.tolist() for indices in fold_indices] == [[0, 1], [2, 3], [4, 5]]
    np.testing.assert_allclose(fold_rms, [np.sqrt(5.0), np.sqrt(10.0), np.sqrt(250.0)])

    # The mean also differs from scoring training folds, pooling residuals, or
    # averaging either signed or absolute residuals within folds.
    mean_fold_rms = np.mean(fold_rms)
    median_fold_rms = np.median(fold_rms)
    mean_train_rms = np.mean(train_rms)
    pooled_rms = np.sqrt(np.mean(residuals**2))
    mean_fold_residual = np.mean(fold_mean_residuals)
    mean_fold_absolute_residual = np.mean(fold_mean_absolute_residuals)
    assert not np.isclose(mean_fold_rms, median_fold_rms)
    assert not np.isclose(mean_fold_rms, mean_train_rms)
    assert len(
        {
            mean_fold_rms,
            median_fold_rms,
            mean_train_rms,
            pooled_rms,
            mean_fold_residual,
            mean_fold_absolute_residual,
        }
    ) == 6
    np.testing.assert_allclose(
        cv_rmse_per_fold(splitter, residuals), mean_fold_rms
    )
