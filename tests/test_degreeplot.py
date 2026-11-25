import numpy as np
import pandas as pd
import pytest

from src.degreeplot import (
    MARKER_STYLES,
    PREDICTIONS,
    compute_prediction_percentile,
    evaluate_predictions,
    make_curve,
)


def test_below_all_replicates_is_zero_percentile_and_miss():
    replicates = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    res = compute_prediction_percentile(replicates, prediction=0.5)

    assert res["Prediction Percentile"] == 0.0
    assert res["Replicate Count"] == 5
    assert res["Result"] == "Miss"


def test_above_all_replicates_is_hundred_percentile_and_miss():
    replicates = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    res = compute_prediction_percentile(replicates, prediction=6.0)

    assert res["Prediction Percentile"] == 100.0
    assert res["Result"] == "Miss"


def test_exact_median_with_odd_replicates_uses_midrank_tie():
    replicates = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    res = compute_prediction_percentile(replicates, prediction=30.0)

    assert res["Prediction Percentile"] == 50.0
    assert res["Result"] == "Not miss"


def test_multiple_tied_values_midrank_calculation():
    replicates = np.array([1.0, 2.0, 2.0, 2.0, 5.0])
    res = compute_prediction_percentile(replicates, prediction=2.0)

    assert res["Prediction Percentile"] == pytest.approx(50.0)
    assert res["Result"] == "Not miss"


def test_between_distinct_values_without_ties():
    replicates = np.array([1.0, 2.0, 3.0, 4.0])
    res = compute_prediction_percentile(replicates, prediction=3.5)

    assert res["Prediction Percentile"] == pytest.approx(75.0)
    assert res["Result"] == "Not miss"


def test_band_boundaries_inclusive_not_miss():
    replicates = np.linspace(0.0, 10.0, 11)
    low_res = compute_prediction_percentile(replicates, prediction=1.0)
    high_res = compute_prediction_percentile(replicates, prediction=9.0)
    just_below = compute_prediction_percentile(replicates, prediction=0.9999)
    just_above = compute_prediction_percentile(replicates, prediction=9.0001)

    assert low_res["Replicate Band Low (10th percentile)"] == pytest.approx(1.0)
    assert high_res["Replicate Band High (90th percentile)"] == pytest.approx(9.0)
    assert low_res["Result"] == "Not miss"
    assert high_res["Result"] == "Not miss"
    assert just_below["Result"] == "Miss"
    assert just_above["Result"] == "Miss"


def test_empty_replicates_raises_value_error():
    with pytest.raises(ValueError):
        compute_prediction_percentile(np.array([]), prediction=1.0)


def test_evaluate_predictions_end_to_end_and_missing_degree():
    raw_df = pd.DataFrame(
        {
            "Degree": [3, 3, 3],
            "Train RMSE": [9.0, 10.0, 11.0],
            "CV RMSE": [11.0, 12.0, 13.0],
            "Floor": [10.0, 10.0, 10.0],
        }
    )
    curve = make_curve(raw_df)
    actual_curves = {("Small", "LinearRegression"): curve}
    preds = {("Small", "LinearRegression"): {3: (1.0, 1.2)}}

    out_df = evaluate_predictions(actual_curves, preds)
    assert len(out_df) == 3
    subset = out_df[out_df["Ratio"].isin(["CV Ratio", "Train Ratio"])]
    assert (subset["Prediction Percentile"] == 50.0).all()

    bad_preds = {("Small", "LinearRegression"): {99: (1.0, 1.2)}}
    with pytest.raises(ValueError):
        evaluate_predictions(actual_curves, bad_preds)


def test_overlapping_predictions_have_distinct_visual_styles_from_module():
    """
    Imports MARKET_STYLES from degreeplot.py to test the live implementation,
    verifying that estimators sharing identical coordinate predictions at degree 12
    (Small LinearRegression and Small Ridge) use distinct marker shapes or colors
    so they do not occlude each other.
    """
    ols_coords = PREDICTIONS[("Small", "LinearRegression")][12]
    ridge_coords = PREDICTIONS[("Small", "Ridge")][12]

    # Confirm the duplicate coordinate exists as specified
    assert ols_coords == ridge_coords

    ls_style = MARKER_STYLES[("LinearRegression", "Small")]
    ridge_style = MARKER_STYLES[("Ridge", "Small")]

    # Assert that their styles are distinct
    assert ls_style != ridge_style
