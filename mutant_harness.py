"""Generate test-targeted mutants and merge mutations from the poly matrix."""

from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import posixpath
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Iterator
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parent
TESTS_DIR = ROOT / "tests"
TEST_TARGETS = {
    "test_best_alpha": "best_alpha",
    "test_buildpipelines": "build_pipelines",
    "test_bias_results": "bias_results",
    "test_curve_for": "curve_for",
    "test_cv_rmse_at": "cv_rmse_at",
    "test_cv_rmse_per_fold": "cv_rmse_per_fold",
    "test_final_test_score": "final_test_score",
    "test_noise_floor": "add_noise_floor",
    "test_placement": "placement",
    "test_plot": "plot_curve",
    "test_polydata": "make_poly_data",
    "test_realized_noise_rms": "realized_noise_rms",
    "test_variation": "variation",
    "test_paired_summary": "paired_summary",
}
MATRIX_HEADERS = ("Mutation", "Code change")
SUMMARY_HEADERS = (
    "Mutation",
    "Code change",
    "Tests failing",
    "Failing tests and why",
    "Outcome source",
)
MUTATION_MATRIX_PATH = ROOT / "poly mutation matrix.xlsx"
PAIRED_DIFF_TESTS = (
    "test_paired_summary_counts_negative_wins_and_excludes_ties",
    "test_paired_summary_returns_mean_and_standard_error",
    "test_seed_pipeline_pairwise_diffs_match_hand_built_table",
)
PAIRED_DIFF_MUTANTS = (
    (
        "P001",
        "paired_summary counts positive differences instead of negative differences",
        "if diff < 0:",
        "if diff > 0:",
        (PAIRED_DIFF_TESTS[0], PAIRED_DIFF_TESTS[2]),
    ),
    (
        "P002",
        "paired_summary counts entries with the smallest absolute difference",
        "return np.mean(d), standard_error, a_win_count",
        "return np.mean(d), standard_error, sum("
        "abs(diff) == min(abs(value) for value in d) for diff in d"
        ")",
        (PAIRED_DIFF_TESTS[0], PAIRED_DIFF_TESTS[2]),
    ),
    (
        "P003",
        "paired_summary uses ddof=0 instead of ddof=1 in the standard error",
        "np.std(d, ddof=1)",
        "np.std(d, ddof=0)",
        (PAIRED_DIFF_TESTS[1],),
    ),
    (
        "P004",
        "paired_summary returns the standard deviation instead of the standard error",
        "np.std(d, ddof=1) / np.sqrt(len(d))",
        "np.std(d, ddof=1)",
        (PAIRED_DIFF_TESTS[1],),
    ),
)
NOISE_FLOOR_TESTS = (
    "test_add_noise_floor_adds_labeled_line_and_closes_figure",
    "test_main_uses_training_cv_noise_floor_not_test_residuals",
    "test_main_draws_overlay_noise_floor_after_pipeline_curves",
)
NOISE_FLOOR_MUTANTS = (
    (
        "N001",
        "add_noise_floor leaves a positive noise floor above zero",
        (
            (
                "if floor > 0:\n        floor = -floor",
                "if floor > 0:\n        floor = floor",
            ),
        ),
        (NOISE_FLOOR_TESTS[0],),
    ),
    (
        "N002",
        "add_noise_floor uses test residual RMS instead of training CV RMSE",
        (
            (
                "add_noise_floor(ax, big_cv_rmse)",
                "add_noise_floor(ax, big_rnoise)",
            ),
        ),
        (NOISE_FLOOR_TESTS[1],),
    ),
    (
        "N003",
        "add_noise_floor is drawn before the overlay curves and uses red",
        (
            (
                "    plot_curve(sf,parameter=\"ridge__alpha\", ax=axe[0], score=\"Negative Root Mean Squared Error\")\n"
                "    plot_curve(sl,parameter=\"ridge__alpha\", ax=axe[0], score=\"Negative Root Mean Squared Error\")\n"
                "    plot_curve(sb, parameter=\"ridge__alpha\", ax=axe[0], score=\"Negative Root Mean Squared Error\")\n"
                "    axe[0].grid(True)\n"
                "    add_noise_floor(axe[0], big_cv_rmse)",
                "    add_noise_floor(axe[0], big_cv_rmse)\n"
                "    plot_curve(sf,parameter=\"ridge__alpha\", ax=axe[0], score=\"Negative Root Mean Squared Error\")\n"
                "    plot_curve(sl,parameter=\"ridge__alpha\", ax=axe[0], score=\"Negative Root Mean Squared Error\")\n"
                "    plot_curve(sb, parameter=\"ridge__alpha\", ax=axe[0], score=\"Negative Root Mean Squared Error\")\n"
                "    axe[0].grid(True)",
            ),
            (
                "ax.axhline(floor, label=str(floor), linestyle='--',color='grey')",
                "ax.axhline(floor, label=str(floor), linestyle='--',color='red')",
            ),
        ),
        (NOISE_FLOOR_TESTS[0], NOISE_FLOOR_TESTS[2]),
    ),
)
SEED_PIPELINE_TESTS = (
    "test_seed_pipeline_win_counts_match_hand_calculated_winners",
    "test_win_tally_can_differ_from_mean_rmse_ranking",
    "test_exact_ties_award_no_pipeline_a_win",
    "test_win_count_sum_excludes_tied_seeds",
)
SEED_PIPELINE_MUTANTS = (
    (
        "S001",
        "seed_pipeline_results uses argmax instead of argmin",
        "np.min(values)",
        "np.max(values)",
        SEED_PIPELINE_TESTS,
    ),
    (
        "S002",
        "seed_pipeline_results finds minima across seeds (axis 0) instead of per seed",
        "        values = np.asarray(list(rmse.values()))\n"
        "        winner_indices = np.flatnonzero(values == np.min(values))",
        "        values_by_seed = np.asarray(list(rmse_table.values()))\n"
        "        pipeline_minima = values_by_seed.min(axis=0)\n"
        "        winner_indices = np.flatnonzero(\n"
        "            pipeline_minima == values_by_seed.min()\n"
        "        )",
        (*SEED_PIPELINE_TESTS, "test_seed_pipeline_pairwise_diffs_match_hand_built_table"),
    ),
    (
        "S003",
        "seed_pipeline_results returns only the first pipeline's count",
        "return win_counts, diffs",
        "return {p1: win_counts[p1]}, diffs",
        SEED_PIPELINE_TESTS,
    ),
    (
        "S004",
        "seed_pipeline_results swaps two pipeline names in the output mapping",
        "return win_counts, diffs",
        "return {p2: win_counts[p1], p1: win_counts[p2], p3: win_counts[p3]}, diffs",
        SEED_PIPELINE_TESTS[:3],
    ),
    (
        "S005",
        "seed_pipeline_results awards exact ties to the first pipeline",
        "if winner_indices.size == 1:",
        "if winner_indices.size >= 1:",
        (
            SEED_PIPELINE_TESTS[0],
            SEED_PIPELINE_TESTS[2],
            SEED_PIPELINE_TESTS[3],
        ),
    ),
)
RANK_PIPELINE_TESTS = (
    "test_rank_pipelines_orders_by_lowest_error_not_closest_to_floor",
)
RANK_PIPELINE_MUTANTS = (
    (
        "R001",
        "rank_pipelines orders by absolute score descending",
        "np.argmin(list(temp.values()))",
        "np.argmax(np.abs(list(temp.values())))",
        RANK_PIPELINE_TESTS,
    ),
    (
        "R002",
        "rank_pipelines sorts negated RMSE scores ascending",
        "np.argmin(list(temp.values()))",
        "np.argmin([-value for value in temp.values()])",
        RANK_PIPELINE_TESTS,
    ),
    (
        "R003",
        "rank_pipelines orders by distance to the test noise floor",
        "np.argmin(list(temp.values()))",
        "np.argmin([abs(value - 9.0) for value in temp.values()])",
        RANK_PIPELINE_TESTS,
    ),
)
PLACEMENT_TESTS = (
    "test_placement_returns_variances_condition_and_max_absolute_correlation",
    "test_placement_correlates_columns_not_rows",
    "test_placement_does_not_mutate_training_data",
    "test_placement_does_not_fit_the_callers_pipeline",
    "test_placement_returns_zero_when_only_one_column_varies",
    "test_placement_computes_condition_from_nonzero_variance_columns",
)
PLACEMENT_MUTANTS = (
    (
        "PL001",
        "placement calculates sample variance instead of population variance",
        (("vars = np.var(result, axis=0)", "vars = np.var(result, axis=0, ddof=1)"),),
        ("test_placement_returns_variances_condition_and_max_absolute_correlation",),
    ),
    (
        "PL002",
        "placement returns the condition number of the Gram matrix",
        (
            (
                "condition = np.linalg.cond(result)",
                "condition = np.linalg.cond(result.T @ result)",
            ),
        ),
        (
            "test_placement_returns_variances_condition_and_max_absolute_correlation",
            "test_placement_computes_condition_from_nonzero_variance_columns",
        ),
    ),
    (
        "PL003",
        "placement returns the signed maximum correlation instead of absolute correlation",
        (
            (
                "c = np.abs(np.corrcoef(result, rowvar=False))",
                "c = np.corrcoef(result, rowvar=False)",
            ),
        ),
        ("test_placement_returns_variances_condition_and_max_absolute_correlation",),
    ),
    (
        "PL004",
        "placement calculates correlations across rows instead of columns",
        (
            (
                "np.corrcoef(result, rowvar=False)",
                "np.corrcoef(result, rowvar=True)",
            ),
        ),
        (
            "test_placement_returns_variances_condition_and_max_absolute_correlation",
            "test_placement_correlates_columns_not_rows",
        ),
    ),
    (
        "PL005",
        "placement calculates correlation with fewer than two varying columns",
        (
            (
                "np.count_nonzero(non_zero_vars) < 2",
                "np.count_nonzero(non_zero_vars) < 1",
            ),
        ),
        ("test_placement_returns_zero_when_only_one_column_varies",),
    ),
    (
        "PL006",
        "placement calculates the condition number before removing zero-variance columns",
        (
            (
                "result = result[:, non_zero_vars]\n"
                "        condition = np.linalg.cond(result)",
                "condition = np.linalg.cond(result)\n"
                "        result = result[:, non_zero_vars]",
            ),
        ),
        ("test_placement_computes_condition_from_nonzero_variance_columns",),
    ),
    (
        "PL007",
        "placement fits the caller's pipeline instead of a clone",
        (("scaling = clone(pipeline[:-1])", "scaling = pipeline[:-1]"),),
        ("test_placement_does_not_fit_the_callers_pipeline",),
    ),
)
BIAS_RESULTS_TESTS = (
    "test_bias_results_matches_hand_calculated_linear_data",
    "test_bias_results_uses_absolute_negative_prediction_difference",
    "test_bias_results_returns_maxima_across_alphas",
    "test_bias_results_raises_value_error_for_unsupported_model_type",
)
BIAS_RESULTS_MUTANTS = (
    (
        "BR001",
        "bias_results preserves the sign of prediction differences",
        (("residuals.append(abs(residual))", "residuals.append(residual)"),),
        (
            "test_bias_results_uses_absolute_negative_prediction_difference",
            "test_bias_results_returns_maxima_across_alphas",
        ),
    ),
    (
        "BR002",
        "bias_results raises the wrong exception for unsupported model types",
        (
            (
                'raise ValueError("Unsupported model type for setting include_bias.")',
                'raise TypeError("Unsupported model type for setting include_bias.")',
            ),
        ),
        ("test_bias_results_raises_value_error_for_unsupported_model_type",),
    ),
    (
        "BR003",
        "bias_results returns the final alpha's bias instead of the maximum",
        (("return bias.max(), residuals.max()", "return bias[-1], residuals.max()"),),
        ("test_bias_results_returns_maxima_across_alphas",),
    ),
    (
        "BR004",
        "bias_results takes the prediction difference maximum from only the final alpha",
        (("return bias.max(), residuals.max()", "return bias.max(), residuals[-1].max()"),),
        ("test_bias_results_returns_maxima_across_alphas",),
    ),
)


PIPELINE_MUTANTS = (
    (
        "PIPE001",
        "build_pipelines sets fit_intercept=False",
        (
            (
                '"scale_first": Pipeline(steps=[(\'scaler\', StandardScaler()), '
                '("polynomial", PolynomialFeatures(degree=degree)), '
                '("ridge", Ridge())]),',
                '"scale_first": Pipeline(steps=[(\'scaler\', StandardScaler()), '
                '("polynomial", PolynomialFeatures(degree=degree)), '
                '("ridge", Ridge(fit_intercept=False))]),',
            ),
            (
                '"scale_afterpoly": Pipeline([("polynomial", '
                'PolynomialFeatures(degree=degree)),(\'scaler\', StandardScaler()), '
                '("ridge", Ridge())]),',
                '"scale_afterpoly": Pipeline([("polynomial", '
                'PolynomialFeatures(degree=degree)),(\'scaler\', StandardScaler()), '
                '("ridge", Ridge(fit_intercept=False))]),',
            ),
            (
                '"scale_both": Pipeline([(\'scaler_before\', StandardScaler()),'
                '("polynomial", PolynomialFeatures(degree=degree)), '
                '(\'scaler_after\', StandardScaler()), ("ridge", Ridge())]),',
                '"scale_both": Pipeline([(\'scaler_before\', StandardScaler()),'
                '("polynomial", PolynomialFeatures(degree=degree)), '
                '(\'scaler_after\', StandardScaler()), '
                '("ridge", Ridge(fit_intercept=False))]),',
            ),
        ),
        ("test_fit_intercept",),
    ),
    (
        "PIPE002",
        "scale_afterpoly polynomial uses include_bias=False",
        (
            (
                '"scale_afterpoly": Pipeline([("polynomial", '
                'PolynomialFeatures(degree=degree)),(\'scaler\', StandardScaler()), '
                '("ridge", Ridge())]),',
                '"scale_afterpoly": Pipeline([("polynomial", '
                'PolynomialFeatures(degree=degree, include_bias=False)),'
                '(\'scaler\', StandardScaler()), ("ridge", Ridge())]),',
            ),
        ),
        ("test_bias",),
    ),
)
PIPELINE_TESTS = ("test_fit_intercept", "test_bias")
FINAL_SCORE_TESTS = (
    "test_score_matches_independently_built_named_scorer",
    "test_negative_r2_test",
)
FINAL_SCORE_MUTANTS = (
    (
        "SCORE001",
        "final_test_score applies abs() to every scorer result",
        (
            (
                "return scorer(test, X_test, y_test)",
                "return abs(scorer(test, X_test, y_test))",
            ),
        ),
        FINAL_SCORE_TESTS,
    ),
    (
        "SCORE002",
        "final_test_score makes negative R2 scores positive",
        (
            (
                "return scorer(test, X_test, y_test)",
                "return abs(scorer(test, X_test, y_test)) "
                'if scoring == "r2" else scorer(test, X_test, y_test)',
            ),
        ),
        ("test_negative_r2_test",),
    ),
)

STALE_IMPORTED_MUTANTS = {
    "final_test_score reverts to model.score (R2) while the curve scores with MSE",
    "final_test_score drops abs() on the score",
    "add_noise_floor sign not flipped",
    "add_noise_floor uses test residuals",
    "add_noise_floor drawn before plot_curve (red line)",
    "scale_afterpoly polynomial uses include_bias=False",
}

IMPORTED_MUTATION_REPLACEMENTS: dict[
    str, tuple[tuple[str, str], ...]
] = {
    "curve_for hardcodes n_jobs": (("n_jobs=n_jobs", "n_jobs=1"),),
    "curve_for drops scoring": (("scoring=scoring", "scoring=None"),),
    "curve_for returns a copy of alphas": (("return alphas, curve[0], curve[1]", "return np.array(alphas), curve[0], curve[1]"),),
    "curve_for passes a copy of X_train": (("X=X_train", "X=X_train.copy()"),),
    "validation_curve drops the last CV fold from the train scores (patched at the sklearn call)": (
        ("return alphas, curve[0], curve[1]", "return alphas, curve[0][:, :-1], curve[1]"),
    ),
    "validation_curve adds a random offset to scores when n_jobs is not 1 (patched at the sklearn call)": (
        ("return alphas, curve[0], curve[1]", "return alphas, curve[0] + np.random.random(curve[0].shape) * 1e-3, curve[1] + np.random.random(curve[1].shape) * 1e-3"),
    ),
    "validation_curve scales validation scores by 0.9 (patched at the sklearn call)": (
        ("return alphas, curve[0], curve[1]", "return alphas, curve[0], curve[1] * 0.9"),
    ),
    "validation_curve sorts param_range ascending (patched at the sklearn call)": (
        ("param_range=alphas", "param_range=np.sort(alphas)"),
    ),
    "validation_curve halves the train scores (patched at the sklearn call)": (
        ("return alphas, curve[0], curve[1]", "return alphas, curve[0] * 0.5, curve[1]"),
    ),
    "best_alpha takes the median across folds when there are 5 or more folds, otherwise the mean": (
        ("mean_scores = test_scores.mean(axis=1).reshape(-1)", "mean_scores = np.where(test_scores.shape[1] >= 5, np.median(test_scores, axis=1), test_scores.mean(axis=1)).reshape(-1)"),
    ),
    "best_alpha takes the max across folds when there are 5 or more folds, otherwise the mean": (
        ("mean_scores = test_scores.mean(axis=1).reshape(-1)", "mean_scores = np.where(test_scores.shape[1] >= 5, np.max(test_scores, axis=1), test_scores.mean(axis=1)).reshape(-1)"),
    ),
    "best_alpha breaks ties by lowest first-fold score, then smallest alpha": (
        ("return best_scores[0]", "return alphas[np.lexsort((alphas, test_scores[:, 0], -mean_scores))[0]]"),
    ),
    "validation_curve adds a bonus that grows with sqrt(alpha) to validation scores (patched at the sklearn call)": (
        ("return alphas, curve[0], curve[1]", "return alphas, curve[0], curve[1] + 0.02 * np.sqrt(alphas)[:, None]"),
    ),
    "plot_curve hardcodes the score label to r2": (("score_name=score", 'score_name="r2"'),),
    "plot_curve reverses the alpha range": (("param_range=result[0]", "param_range=result[0][::-1]"),),
    "plot_curve returns None": (
        ("return ValidationCurveDisplay(", "ValidationCurveDisplay("),
    ),
    "plot_curve swaps train and test scores": (
        ("train_scores=result[1], test_scores=result[2]", "train_scores=result[2], test_scores=result[1]"),
    ),
    "plot_curve hardcodes the parameter name": (("param_name=parameter,param_range=result[0]", 'param_name="alpha",param_range=result[0]'),),
    "plot_curve labels the y-axis with the parameter name instead of the score": (("score_name=score", "score_name=parameter"),),
    "best_alpha pairs sorted alphas with unsorted scores": (
        ("return best_scores[0]", "return np.sort(alphas)[np.argmax(mean_scores)]"),
    ),
    "best_alpha returns the lower median of tied alphas instead of the smallest": (
        ("return best_scores[0]", "return best_scores[(len(best_scores) - 1) // 2]"),
    ),
    "best_alpha accepts boolean alphas (drops the boolean check)": (
        ("elif np.any([isinstance(alpha, (bool, np.bool)) for alpha in alphas]):", "elif False:"),
    ),
    "best_alpha raises ValueError instead of TypeError for boolean alphas": (
        ("""raise TypeError("Alphas can't be boolean.")""", """raise ValueError("Alphas can't be boolean.")"""),
    ),
    "best_alpha accepts negative alphas (drops the negative check)": (
        ("elif np.any(alphas < 0):", "elif False:"),
    ),
    "best_alpha raises TypeError instead of ValueError for negative alphas": (
        ('raise ValueError("All Alphas must be non-negative.")', 'raise TypeError("All Alphas must be non-negative.")'),
    ),
    "best_alpha rejects a zero alpha": (("elif np.any(alphas < 0):", "elif np.any(alphas <= 0):"),),
    "best_alpha drops the scalar-type check": (
        ("if not np.all([isinstance(alpha, (float, int, np.floating, np.integer)) for alpha in alphas]) or not np.all(np.isfinite(alphas)):", "if not np.all(np.isfinite(alphas)):"),
    ),
    "best_alpha drops the finite check": (
        ("if not np.all([isinstance(alpha, (float, int, np.floating, np.integer)) for alpha in alphas]) or not np.all(np.isfinite(alphas)):", "if not np.all([isinstance(alpha, (float, int, np.floating, np.integer)) for alpha in alphas]):"),
    ),
    "best_alpha raises ValueError instead of TypeError for non-scalar alphas": (
        ('raise TypeError("Each alpha must be a scalar float or int.")', 'raise ValueError("Each alpha must be a scalar float or int.")'),
    ),
    "final_test_score accepts numpy arrays as best_alpha": (
        ("(np.integer,np.floating, float, int)", "(np.integer,np.floating, float, int, np.ndarray)"),
    ),
    "final_test_score accepts booleans for best_alpha": (
        ("isinstance(best_alpha,(bool,np.bool_)) or ", ""),
    ),
    "final_test_score drops the nan/inf check on best_alpha": (
        (" or np.isfinite(best_alpha) == False", ""),
    ),
    "final_test_score drops the negative-alpha check": (
        ("""    if best_alpha < 0:\n        raise ValueError("Alphas can't be negative.")\n""", ""),
    ),
    "final_test_score raises TypeError for unsupported models": (
        ('raise ValueError("Unsupported model type.")', 'raise TypeError("Unsupported model type.")'),
    ),
    "final_test_score modifies the caller's pipeline (no clone)": (
        ("        test = clone(model)\n        ridge_step =", "        test = model\n        ridge_step ="),
    ),
    "final_test_score modifies the caller's Ridge (no clone)": (
        ("    elif isinstance(model, Ridge):\n        test = clone(model)", "    elif isinstance(model, Ridge):\n        test = model"),
    ),
    "final_test_score ignores best_alpha for pipelines": (
        ('test.set_params(**{f"{ridge_step}__alpha": best_alpha})', 'test.set_params(**{f"{ridge_step}__alpha": 1.0})'),
    ),
    "variation returns the standard deviation instead of the variance": (
        ("return X_poly.var(axis=0)", "return X_poly.std(axis=0)"),
    ),
    "variation uses the sample variance": (
        ("return X_poly.var(axis=0)", "return X_poly.var(axis=0, ddof=1)"),
    ),
    "variation default degree is 2 instead of 3": (
        ("def variation(X: np.ndarray, degree: int=3):", "def variation(X: np.ndarray, degree: int=2):"),
    ),
    "build_pipelines returns an extra pipeline": (
        ("    return pipelines", '    pipelines["extra"] = clone(pipelines["scale_first"])\n    return pipelines'),
    ),
    "scale_afterpoly puts the scaler before the polynomial step": (
        ('''"scale_afterpoly": Pipeline([("polynomial", PolynomialFeatures(degree=degree)),('scaler', StandardScaler()), ("ridge", Ridge())]),''', '''"scale_afterpoly": Pipeline([('scaler', StandardScaler()), ("polynomial", PolynomialFeatures(degree=degree)), ("ridge", Ridge())]),'''),
    ),
    "scale_both uses degree + 1": (
        ('''("polynomial", PolynomialFeatures(degree=degree)), ('scaler_after', StandardScaler())''', '''("polynomial", PolynomialFeatures(degree=degree + 1)), ('scaler_after', StandardScaler())'''),
    ),
    "build_pipelines caches and reuses step objects across calls (patched wrapper)": (
        ("    return pipelines", '    if not hasattr(build_pipelines, "_cache"):\n        build_pipelines._cache = {}\n    build_pipelines._cache.setdefault(degree, pipelines)\n    return build_pipelines._cache[degree]'),
    ),
    "scale_both names its final step regressor instead of ridge": (
        ('''('scaler_after', StandardScaler()), ("ridge", Ridge())''', '''('scaler_after', StandardScaler()), ("regressor", Ridge())'''),
    ),
    "scale_afterpoly uses Ridge(alpha=-1.0)": (
        ('''"scale_afterpoly": Pipeline([("polynomial", PolynomialFeatures(degree=degree)),('scaler', StandardScaler()), ("ridge", Ridge())]),''', '''"scale_afterpoly": Pipeline([("polynomial", PolynomialFeatures(degree=degree)),('scaler', StandardScaler()), ("ridge", Ridge(alpha=-1.0))]),'''),
    ),
    "scale_afterpoly ends with LinearRegression after a transform-capable Ridge (patched wrapper)": (
        ("from sklearn.linear_model import Ridge", "from sklearn.linear_model import LinearRegression, Ridge"),
        ('''"scale_afterpoly": Pipeline([("polynomial", PolynomialFeatures(degree=degree)),('scaler', StandardScaler()), ("ridge", Ridge())]),''', '''"scale_afterpoly": Pipeline([("polynomial", PolynomialFeatures(degree=degree)),('scaler', StandardScaler()), ("ridge", Ridge()), ("linear", LinearRegression())]),'''),
    ),
    "build_pipelines no longer type-checks degree": (
        ("if isinstance(degree, (bool, np.bool)) or not isinstance(degree, int):", "if False:"),
    ),
    "make_poly_data caps n at 100": (
        ("    gene=default_rng(seed)", "    n = min(n, 100)\n    gene=default_rng(seed)"),
    ),
    "make_poly_data reuses seed 42 when seed is 43": (
        ("gene=default_rng(seed)", "gene=default_rng(42 if seed == 43 else seed)"),
    ),
    "make_poly_data overshoots x_high by 1 when the range is 10 or wider": (
        ("gene.uniform(x_low, x_high, n)", "gene.uniform(x_low, x_high + (x_high - x_low >= 10), n)"),
    ),
    "make_poly_data evaluates the polynomial on x clipped to [-3, 3]": (
        ("y_clean = polyval(c=true_coefs, x=X)", "y_clean = polyval(c=true_coefs, x=np.clip(X, -3, 3))"),
    ),
    "make_poly_data ignores coefficients beyond the quadratic term": (
        ("y_clean = polyval(c=true_coefs, x=X)", "y_clean = polyval(c=true_coefs[:3], x=X)"),
    ),
    "make_poly_data doubles the noise scale when n >= 500": (
        ("gene.normal(scale=noise, size=n)", "gene.normal(scale=noise * (2 if n >= 500 else 1), size=n)"),
    ),
    "realized_noise_rms uses the mean absolute residual instead of the RMS": (
        ("return np.sqrt(np.mean((y-polyval(x=x, c=true_coefs))**2))", "return np.mean(np.abs(y-polyval(x=x, c=true_coefs)))"),
    ),
    "realized_noise_rms uses the max absolute residual instead of the RMS": (
        ("return np.sqrt(np.mean((y-polyval(x=x, c=true_coefs))**2))", "return np.max(np.abs(y-polyval(x=x, c=true_coefs)))"),
    ),
    "realized_noise_rms returns 0 when the residuals have no spread": (
        ("return np.sqrt(np.mean((y-polyval(x=x, c=true_coefs))**2))", "return 0.0 if np.std(y-polyval(x=x, c=true_coefs)) == 0 else np.sqrt(np.mean((y-polyval(x=x, c=true_coefs))**2))"),
    ),
    "realized_noise_rms adds 1e-20 inside the square root": (
        ("np.sqrt(np.mean((y-polyval(x=x, c=true_coefs))**2))", "np.sqrt(np.mean((y-polyval(x=x, c=true_coefs))**2) + 1e-20)"),
    ),
    "cv_rmse_at always uses the last row of scores": (
        ("alpha_score = scores[np.argwhere(alphas==alpha)]", "alpha_score = scores[-1]"),
    ),
    "cv_rmse_at returns a std of 0 when the grid has one alpha": (
        ("return alpha_score.mean(), alpha_score.std(ddof=ddof)", "return alpha_score.mean(), alpha_score.std(ddof=ddof) * (len(alphas) > 1)"),
    ),
    "cv_rmse_at raises TypeError instead of ValueError for an alpha not in the grid": (
        ('raise ValueError("Alpha must be in the alphas grid.")', 'raise TypeError("Alpha must be in the alphas grid.")'),
    ),
    "cv_rmse_at drops the alpha-in-grid check": (
        ("if alpha not in alphas:", "if False:"),
    ),
    "cv_rmse_at ignores ddof (always uses ddof=0)": (
        ("alpha_score.std(ddof=ddof)", "alpha_score.std(ddof=0)"),
    ),
    "cv_rmse_at adds 1e-12 to the std": (
        ("alpha_score.std(ddof=ddof)", "alpha_score.std(ddof=ddof) + 1e-12"),
    ),
    "cv_rmse_at allows a fold count equal to ddof": (
        ("if scores.shape[1] <= ddof:", "if scores.shape[1] < ddof:"),
    ),
    "cv_rmse_at drops the folds-vs-ddof check": (
        ("if scores.shape[1] <= ddof:", "if False:"),
    ),
    "cv_rmse_at raises TypeError instead of ValueError when folds <= ddof": (
        ('raise ValueError("Number of folds must be greater than ddof.")', 'raise TypeError("Number of folds must be greater than ddof.")'),
    ),
    "cv_rmse_per_fold uses the per-fold mean squared residual (drops the square root) instead of the RMS": (
        ("np.sqrt(np.mean(residuals[split[1]]**2))", "np.mean(residuals[split[1]]**2)"),
    ),
    "cv_rmse_per_fold pools the residuals across folds instead of averaging the fold RMS values": (
        ("cv_means= []", "cv_means= []\n    all_residuals = []"),
        ("cv_means.append(np.sqrt(np.mean(residuals[split[1]]**2)))", "all_residuals.extend(residuals[split[1]]**2)"),
        ("return np.mean(cv_means)", "return np.sqrt(np.mean(all_residuals))"),
    ),
    "cv_rmse_per_fold uses the per-fold mean residual instead of the RMS": (
        ("np.sqrt(np.mean(residuals[split[1]]**2))", "np.mean(residuals[split[1]])"),
    ),
    "cv_rmse_per_fold uses the per-fold mean absolute residual instead of the RMS": (
        ("np.sqrt(np.mean(residuals[split[1]]**2))", "np.mean(np.abs(residuals[split[1]]))"),
    ),
    "cv_rmse_per_fold scores the training indices instead of the test indices": (
        ("residuals[split[1]]", "residuals[split[0]]"),
    ),
    "cv_rmse_per_fold takes the median of the fold RMS values instead of the mean": (
        ("return np.mean(cv_means)", "return np.median(cv_means)"),
    ),
}


def read_mutation_matrix(path: Path) -> tuple[list[str], list[list[str]]]:
    namespace = {
        "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "relationships": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
        "package_relationships": "http://schemas.openxmlformats.org/package/2006/relationships",
    }
    with ZipFile(path) as workbook:
        workbook_xml = ET.fromstring(workbook.read("xl/workbook.xml"))
        relationships_xml = ET.fromstring(
            workbook.read("xl/_rels/workbook.xml.rels")
        )
        sheet = next(
            (
                item
                for item in workbook_xml.findall("main:sheets/main:sheet", namespace)
                if item.attrib.get("name") == "Mutation matrix"
            ),
            None,
        )
        if sheet is None:
            raise ValueError(f"No 'Mutation matrix' worksheet found in {path}.")

        relationship_id = sheet.attrib[
            f"{{{namespace['relationships']}}}id"
        ]
        relationship = next(
            (
                item
                for item in relationships_xml.findall(
                    "package_relationships:Relationship", namespace
                )
                if item.attrib.get("Id") == relationship_id
            ),
            None,
        )
        if relationship is None:
            raise ValueError(f"Worksheet relationship is missing in {path}.")

        target = relationship.attrib["Target"]
        worksheet_path = (
            target.lstrip("/")
            if target.startswith("/")
            else posixpath.normpath(posixpath.join("xl", target))
        )
        shared_strings_path = "xl/sharedStrings.xml"
        if shared_strings_path in workbook.namelist():
            shared_strings_xml = ET.fromstring(
                workbook.read(shared_strings_path)
            )
            shared_strings = [
                "".join(
                    text.text or ""
                    for text in item.iter(f"{{{namespace['main']}}}t")
                )
                for item in shared_strings_xml.findall("main:si", namespace)
            ]
        else:
            shared_strings = []

        worksheet_xml = ET.fromstring(workbook.read(worksheet_path))

    rows: list[list[str]] = []
    for row in worksheet_xml.findall("main:sheetData/main:row", namespace):
        values: dict[int, str] = {}
        for cell in row.findall("main:c", namespace):
            reference = cell.attrib.get("r", "")
            column_match = re.match(r"[A-Z]+", reference)
            if column_match is None:
                raise ValueError(f"Invalid cell reference {reference!r} in {path}.")
            column = 0
            for character in column_match.group():
                column = column * 26 + ord(character) - ord("A") + 1
            column -= 1

            cell_type = cell.attrib.get("t")
            if cell_type == "inlineStr":
                value = "".join(
                    text.text or ""
                    for text in cell.iter(f"{{{namespace['main']}}}t")
                )
            else:
                value_node = cell.find("main:v", namespace)
                value = "" if value_node is None else value_node.text or ""
                if cell_type == "s" and value:
                    value = shared_strings[int(value)]
            values[column] = value
        if values:
            rows.append(
                [values.get(column, "") for column in range(max(values) + 1)]
            )

    if not rows or tuple(rows[0][:2]) != MATRIX_HEADERS:
        raise ValueError(f"{path} must start with the headers {MATRIX_HEADERS!r}.")
    if rows[0][-1] != "Tests failing":
        raise ValueError(f"{path} must end its header row with 'Tests failing'.")
    return rows[0], rows[1:]


def run_pytest(arguments: list[str], cwd: Path, timeout: int) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["JOBLIB_MULTIPROCESSING"] = "0"
    return subprocess.run(
        [sys.executable, "-m", "pytest", *arguments],
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def collect_tests(test_files: list[str], timeout: int) -> list[str]:
    result = run_pytest(
        ["--collect-only", "-q", "--no-header", *test_files], ROOT, timeout
    )
    if result.returncode:
        raise RuntimeError(
            "Pytest collection failed:\n" + (result.stdout + result.stderr)[-8000:]
        )
    return [
        line.strip()
        for line in result.stdout.splitlines()
        if "::" in line and not line.lstrip().startswith("<")
    ]


def read_junit_results(path: Path, tests: list[str]) -> dict[str, str]:
    results = {test: "ERROR" for test in tests}
    if not path.exists():
        return results

    root = ET.parse(path).getroot()
    for case in root.iter("testcase"):
        classname = case.attrib.get("classname", "").replace(".", "/")
        node_id = f"{classname}.py::{case.attrib.get('name', '')}"
        if node_id not in results:
            continue
        if case.find("failure") is not None or case.find("error") is not None:
            results[node_id] = "FAIL"
        elif case.find("skipped") is not None:
            results[node_id] = "SKIP"
        else:
            results[node_id] = "PASS"
    return results


@contextmanager
def temporary_directory(prefix: str) -> Iterator[Path]:
    temp = tempfile.TemporaryDirectory(prefix=prefix)
    try:
        yield Path(temp.name)
    finally:
        for attempt in range(5):
            try:
                temp.cleanup()
                break
            except PermissionError as error:
                if (
                    sys.platform != "win32"
                    or getattr(error, "winerror", None) != 32
                    or attempt == 4
                ):
                    raise
                time.sleep(0.1 * (2**attempt))


def run_test_suite(
    cwd: Path, tests: list[str], timeout: int
) -> tuple[dict[str, str], str]:
    with temporary_directory(prefix="mutant-junit-") as temp:
        junit_path = temp / "results.xml"
        try:
            result = run_pytest(
                [
                    "-q",
                    "--tb=short",
                    "--no-header",
                    "--disable-warnings",
                    f"--junitxml={junit_path}",
                    *tests,
                ],
                cwd,
                timeout,
            )
        except subprocess.TimeoutExpired:
            return {test: "ERROR" for test in tests}, "timed out"
        statuses = read_junit_results(junit_path, tests)
        return statuses, "passed" if result.returncode == 0 else "failed"


def insert_failure_mutation(source: str, function_name: str, mutant_id: str) -> str:
    tree = ast.parse(source)
    function = next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == function_name
        ),
        None,
    )
    if function is None:
        raise ValueError(f"Function {function_name!r} was not found in poly.py.")

    first_statement = function.body[0]
    has_docstring = (
        isinstance(first_statement, ast.Expr)
        and isinstance(first_statement.value, ast.Constant)
        and isinstance(getattr(first_statement.value, "value", None), str)
    )
    insertion_index = (
        first_statement.end_lineno if has_docstring else first_statement.lineno - 1
    )
    statement_line = (
        first_statement.end_lineno + 1
        if has_docstring
        else first_statement.lineno
    )
    lines = source.splitlines(keepends=True)
    indent = re.match(r"\s*", lines[statement_line - 1]).group()
    message = f"MUTANT-HARNESS: {mutant_id}"
    lines.insert(insertion_index, f"{indent}raise RuntimeError({message!r})\n")
    return "".join(lines)


def replace_mutation(source: str, before: str, after: str, mutant_id: str) -> str:
    if source.count(before) != 1:
        raise ValueError(
            f"Mutant {mutant_id} expected exactly one occurrence of {before!r}."
        )
    return source.replace(before, after, 1)


def copy_test_files(temp_root: Path, tests: list[str]) -> None:
    for relative_test in {test.split("::", 1)[0] for test in tests}:
        destination = temp_root / relative_test
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative_test, destination)


def run_mutant(
    index: int,
    node_id: str,
    function_name: str,
    source: str,
    tests: list[str],
    timeout: int,
) -> dict[str, object]:
    mutant_id = f"M{index:03d}"

    with temporary_directory(prefix=f"mutant-{mutant_id}-") as temp_root:
        copy_test_files(temp_root, tests)
        shutil.copy2(ROOT / "seed_runs.py", temp_root / "seed_runs.py")
        (temp_root / "poly.py").write_text(
            insert_failure_mutation(source, function_name, mutant_id),
            encoding="utf-8",
        )

        statuses, suite_result = run_test_suite(temp_root, tests, timeout)

    failed = [test for test, status in statuses.items() if status == "FAIL"]
    details = (
        f"Injected RuntimeError at {function_name} entry; "
        f"{len(failed)} test(s) failed in the full test suite."
    )
    if suite_result == "timed out":
        details = f"Full test suite exceeded {timeout} seconds."
    return {
        "id": mutant_id,
        "target": node_id,
        "function": function_name,
        "statuses": statuses,
        "failed": failed,
        "suite_result": suite_result,
        "details": details,
    }


def run_poly_source_mutant(
    mutant_id: str,
    description: str,
    replacements: tuple[tuple[str, str], ...],
    expected_failures: tuple[str, ...],
    source: str,
    tests: list[str],
    timeout: int,
) -> dict[str, object]:
    with temporary_directory(prefix=f"mutant-{mutant_id}-") as temp_root:
        copy_test_files(temp_root, tests)
        shutil.copy2(ROOT / "seed_runs.py", temp_root / "seed_runs.py")
        mutated_source = source
        for before, after in replacements:
            mutated_source = replace_mutation(
                mutated_source, before, after, mutant_id
            )
        (temp_root / "poly.py").write_text(mutated_source, encoding="utf-8")
        statuses, suite_result = run_test_suite(temp_root, tests, timeout)

    failed = [test for test, status in statuses.items() if status == "FAIL"]
    expected_nodes = [
        next(test for test in tests if test.endswith(f"::{name}"))
        for name in expected_failures
    ]
    unexpected_failures = sorted(set(failed) - set(expected_nodes))
    matched = set(expected_nodes).issubset(failed)
    details = (
        f"Expected at least {expected_nodes!r} to fail; observed {failed!r}; "
        f"extra failures: {unexpected_failures!r} ({suite_result})."
    )
    return {
        "id": mutant_id,
        "target": ", ".join(expected_nodes),
        "description": description,
        "statuses": statuses,
        "failed": failed,
        "expected_failures": expected_nodes,
        "unexpected_failures": unexpected_failures,
        "matched": matched,
        "suite_result": suite_result,
        "details": details,
    }


def run_seed_runs_mutant(
    mutant_id: str,
    description: str,
    before: str,
    after: str,
    expected_failures: tuple[str, ...],
    tests: list[str],
    timeout: int,
) -> dict[str, object]:
    expected_nodes = [
        next(test for test in tests if test.endswith(f"::{name}"))
        for name in expected_failures
    ]
    with temporary_directory(prefix=f"mutant-{mutant_id}-") as temp_root:
        copy_test_files(temp_root, tests)
        shutil.copy2(ROOT / "poly.py", temp_root / "poly.py")
        shutil.copy2(ROOT / "seed_runs.py", temp_root / "seed_runs.py")
        source = (temp_root / "seed_runs.py").read_text(encoding="utf-8")
        mutated_source = replace_mutation(source, before, after, mutant_id)
        (temp_root / "seed_runs.py").write_text(
            mutated_source, encoding="utf-8"
        )
        statuses, suite_result = run_test_suite(temp_root, tests, timeout)

    failed = [test for test, status in statuses.items() if status == "FAIL"]
    unexpected_failures = sorted(set(failed) - set(expected_nodes))
    matched = set(expected_nodes).issubset(failed)
    details = (
        f"Expected at least {expected_nodes!r} to fail; observed {failed!r}; "
        f"extra failures: {unexpected_failures!r} ({suite_result})."
    )
    return {
        "id": mutant_id,
        "target": ", ".join(expected_nodes),
        "description": description,
        "statuses": statuses,
        "failed": failed,
        "expected_failures": expected_nodes,
        "unexpected_failures": unexpected_failures,
        "matched": matched,
        "suite_result": suite_result,
        "details": details,
    }


def run_imported_mutant(
    mutation_name: str,
    description: str,
    replacements: tuple[tuple[str, str], ...],
    source: str,
    tests: list[str],
    timeout: int,
) -> dict[str, object]:
    with temporary_directory(prefix="imported-mutant-") as temp_root:
        copy_test_files(temp_root, tests)
        shutil.copy2(ROOT / "seed_runs.py", temp_root / "seed_runs.py")
        mutated_source = source
        for before, after in replacements:
            mutated_source = replace_mutation(
                mutated_source, before, after, mutation_name
            )
        (temp_root / "poly.py").write_text(mutated_source, encoding="utf-8")
        statuses, suite_result = run_test_suite(temp_root, tests, timeout)

    failed = [test for test, status in statuses.items() if status == "FAIL"]
    details = (
        f"{len(failed)} current test(s) failed ({suite_result})."
        if failed
        else f"No current tests failed ({suite_result})."
    )
    return {
        "id": mutation_name,
        "target": mutation_name,
        "description": description,
        "statuses": statuses,
        "failed": failed,
        "suite_result": suite_result,
        "details": details,
    }


def excel_column(number: int) -> str:
    label = ""
    while number:
        number, remainder = divmod(number - 1, 26)
        label = chr(65 + remainder) + label
    return label


def matrix_status(status: str) -> str:
    return {
        "PASS": "pass",
        "FAIL": "FAIL",
        "SKIP": "skip",
        "NOT RUN": "not run",
        "N/A": "N/A",
    }.get(status.strip().upper(), "ERROR")


def worksheet_xml(
    rows: list[list[object]], widths: list[float], sheet_kind: str
) -> str:
    sheet_rows = []
    for row_number, row in enumerate(rows, start=1):
        cells = []
        for column_number, value in enumerate(row, start=1):
            reference = f"{excel_column(column_number)}{row_number}"
            if sheet_kind == "summary" and row_number in {1, 8}:
                style = 7 if column_number == 3 else 6
            elif row_number == 1:
                style = 2 if column_number > 2 else 1
            elif sheet_kind == "summary":
                style = 9 if column_number == 3 else 8
            else:
                style = 3 if column_number <= 2 else 4
            if isinstance(value, (int, float)):
                cells.append(
                    f'<c r="{reference}" s="{style}"><v>{value}</v></c>'
                )
            else:
                text = escape(str(value))
                cells.append(
                    f'<c r="{reference}" s="{style}" t="inlineStr"><is><t xml:space="preserve">'
                    f"{text}</t></is></c>"
                )
        sheet_rows.append(f'<row r="{row_number}">{"".join(cells)}</row>')

    columns = "".join(
        f'<col min="{index}" max="{index}" width="{width}" customWidth="1"/>'
        for index, width in enumerate(widths, start=1)
    )
    pane = (
        '<sheetViews><sheetView workbookViewId="0">'
        '<pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>'
        '</sheetView></sheetViews>'
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'{pane}<cols>{columns}</cols>'
        f'<sheetData>{"".join(sheet_rows)}</sheetData></worksheet>'
    )
def write_workbook(
    path: Path,
    matrix_rows: list[list[object]],
    summary_rows: list[list[object]],
    probe_rows: list[list[object]],
    test_count: int,
) -> None:
    matrix_widths = [47.29, 38.0, *([34.0] * test_count)]
    summary_widths = [47.29, 38.0, 14.43, 68.71, 34.0]
    worksheets = [
        worksheet_xml(matrix_rows, matrix_widths, "matrix"),
        worksheet_xml(summary_rows, summary_widths, "summary"),
        worksheet_xml(probe_rows, matrix_widths, "matrix"),
    ]
    styles = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="4">'
        '<font><sz val="10"/><color rgb="FF000000"/><name val="Calibri"/></font>'
        '<font><b/><sz val="10"/><color rgb="FFFFFFFF"/><name val="Arial"/></font>'
        '<font><sz val="10"/><color rgb="FF000000"/><name val="Arial"/></font>'
        '<font><sz val="10"/><color rgb="FF7F7F7F"/><name val="Arial"/></font>'
        '</fonts>'
        '<fills count="3">'
        '<fill><patternFill patternType="none"/></fill>'
        '<fill><patternFill patternType="gray125"/></fill>'
        '<fill><patternFill patternType="solid"><fgColor rgb="FF1F3764"/>'
        '<bgColor indexed="64"/></patternFill></fill>'
        '</fills>'
        '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="10">'
        '<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
        '<xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1" applyAlignment="1"><alignment horizontal="left" vertical="center" wrapText="1"/></xf>'
        '<xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1" applyAlignment="1"><alignment horizontal="center" vertical="center" wrapText="1"/></xf>'
        '<xf numFmtId="0" fontId="2" fillId="0" borderId="0" xfId="0" applyFont="1" applyAlignment="1"><alignment horizontal="left" vertical="center" wrapText="1"/></xf>'
        '<xf numFmtId="0" fontId="3" fillId="0" borderId="0" xfId="0" applyFont="1" applyAlignment="1"><alignment horizontal="center" vertical="center"/></xf>'
        '<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
        '<xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1" applyAlignment="1"><alignment horizontal="left" vertical="center" wrapText="1"/></xf>'
        '<xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1" applyAlignment="1"><alignment horizontal="center" vertical="center"/></xf>'
        '<xf numFmtId="0" fontId="2" fillId="0" borderId="0" xfId="0" applyFont="1" applyAlignment="1"><alignment vertical="center" wrapText="1"/></xf>'
        '<xf numFmtId="0" fontId="2" fillId="0" borderId="0" xfId="0" applyFont="1" applyAlignment="1"><alignment horizontal="center" vertical="center" wrapText="1"/></xf>'
        '</cellXfs>'
        '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
        '</styleSheet>'
    )
    files = {
        "[Content_Types].xml": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            '<Override PartName="/xl/worksheets/sheet2.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            '<Override PartName="/xl/worksheets/sheet3.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
            "</Types>"
        ),
        "_rels/.rels": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            "</Relationships>"
        ),
        "xl/workbook.xml": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<sheets><sheet name="Mutation matrix" sheetId="1" r:id="rId1"/>'
            '<sheet name="Multi-test mutants" sheetId="2" r:id="rId2"/>'
            '<sheet name="Entry-failure probes" sheetId="3" r:id="rId3"/></sheets>'
            "</workbook>"
        ),
        "xl/_rels/workbook.xml.rels": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
            '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet2.xml"/>'
            '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet3.xml"/>'
            '<Relationship Id="rId4" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
            "</Relationships>"
        ),
        "xl/worksheets/sheet1.xml": worksheets[0],
        "xl/worksheets/sheet2.xml": worksheets[1],
        "xl/worksheets/sheet3.xml": worksheets[2],
        "xl/styles.xml": styles,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "w", ZIP_DEFLATED) as workbook:
        for name, content in files.items():
            workbook.writestr(name, content)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Test targeted mutants and include entries from the poly mutation matrix."
        )
    )
    parser.add_argument(
        "-o",
        "--output",
        default="mutant_report.xlsx",
        help="XLSX report path (default: mutant_report.xlsx)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Maximum isolated mutant tests to run concurrently (default: 2)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Per-test-suite timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Run only the first N collected tests (useful for a quick smoke test)",
    )
    args = parser.parse_args()
    if args.workers < 1 or args.timeout < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--workers, --timeout, and --limit must be positive integers")

    test_files = sorted(
        str(path.relative_to(ROOT))
        for path in TESTS_DIR.glob("test_*.py")
        if path.is_file()
    )
    if not test_files:
        raise RuntimeError(f"No test_*.py files were found in {TESTS_DIR}.")
    all_tests = collect_tests(test_files, args.timeout)
    paired_tests = [
        test
        for test in all_tests
        if Path(test.split("::", 1)[0]).name == "test_paired_summary.py"
    ]
    if not paired_tests:
        raise RuntimeError("No tests were collected from test_paired_summary.py.")
    final_score_tests = [
        test
        for test in all_tests
        if Path(test.split("::", 1)[0]).name == "test_final_test_score.py"
    ]
    for test_name in FINAL_SCORE_TESTS:
        if not any(
            test.endswith(f"::{test_name}") for test in final_score_tests
        ):
            raise RuntimeError(
                f"No test named {test_name!r} was collected from "
                "test_final_test_score.py."
            )
    noise_floor_tests = [
        test
        for test in all_tests
        if Path(test.split("::", 1)[0]).name == "test_noise_floor.py"
    ]
    for test_name in NOISE_FLOOR_TESTS:
        if not any(test.endswith(f"::{test_name}") for test in noise_floor_tests):
            raise RuntimeError(
                f"No test named {test_name!r} was collected from "
                "test_noise_floor.py."
            )
    seed_pipeline_tests = [
        test
        for test in all_tests
        if Path(test.split("::", 1)[0]).name == "test_seed_pipeline_results.py"
    ]
    for test_name in SEED_PIPELINE_TESTS:
        if not any(test.endswith(f"::{test_name}") for test in seed_pipeline_tests):
            raise RuntimeError(
                f"No test named {test_name!r} was collected from "
                "test_seed_pipeline_results.py."
            )
    rank_pipeline_tests = [
        test
        for test in all_tests
        if Path(test.split("::", 1)[0]).name == "test_rank_pipelines.py"
    ]
    for test_name in RANK_PIPELINE_TESTS:
        if not any(test.endswith(f"::{test_name}") for test in rank_pipeline_tests):
            raise RuntimeError(
                f"No test named {test_name!r} was collected from "
                "test_rank_pipelines.py."
            )
    placement_tests = [
        test
        for test in all_tests
        if Path(test.split("::", 1)[0]).name == "test_placement.py"
    ]
    for test_name in PLACEMENT_TESTS:
        if not any(test.endswith(f"::{test_name}") for test in placement_tests):
            raise RuntimeError(
                f"No test named {test_name!r} was collected from "
                "test_placement.py."
            )
    bias_results_tests = [
        test
        for test in all_tests
        if Path(test.split("::", 1)[0]).name == "test_bias_results.py"
    ]
    for test_name in BIAS_RESULTS_TESTS:
        if not any(test.endswith(f"::{test_name}") for test in bias_results_tests):
            raise RuntimeError(
                f"No test named {test_name!r} was collected from "
                "test_bias_results.py."
            )
    pipeline_tests = [
        test
        for test in all_tests
        if Path(test.split("::", 1)[0]).name == "test_buildpipelines.py"
        and any(test.endswith(f"::{name}") for name in PIPELINE_TESTS)
    ]
    for test_name in PIPELINE_TESTS:
        if not any(test.endswith(f"::{test_name}") for test in pipeline_tests):
            raise RuntimeError(
                f"No test named {test_name!r} was collected from "
                "test_buildpipelines.py."
            )
    tests = all_tests
    if args.limit is not None:
        tests = tests[: args.limit]
    if not tests:
        raise RuntimeError("No pytest cases were collected.")

    source = (ROOT / "poly.py").read_text(encoding="utf-8")
    jobs = []
    noise_floor_source_tests = {
        test
        for test in noise_floor_tests
        if any(
            test.endswith(f"::{test_name}")
            for test_name in NOISE_FLOOR_TESTS[1:]
        )
    }
    for node_id in tests:
        test_path = node_id.split("::", 1)[0]
        module_name = Path(test_path).stem
        if (
            module_name in {"test_seed_pipeline_results", "test_rank_pipelines"}
            or node_id in noise_floor_source_tests
        ):
            continue
        function_name = TEST_TARGETS.get(module_name)
        if function_name is None:
            raise RuntimeError(
                f"No mutant target is configured for {test_path!r}; "
                "add its production function to TEST_TARGETS."
            )
        jobs.append((node_id, function_name))

    baseline_statuses, baseline_result = run_test_suite(ROOT, tests, args.timeout)
    baseline_failures = [
        test
        for test, status in baseline_statuses.items()
        if status in {"FAIL", "ERROR"}
    ]
    if baseline_result != "passed" or baseline_failures:
        print(
            "Baseline test suite failed; aborting before mutant runs. "
            f"Failing tests: {baseline_failures!r} "
            f"({baseline_result})."
        )
        return 1
    missing_paired_tests = [
        test for test in paired_tests if test not in baseline_statuses
    ]
    if missing_paired_tests:
        paired_baseline_statuses, paired_baseline_result = run_test_suite(
            ROOT, missing_paired_tests, args.timeout
        )
        supplemental_baseline_results = [paired_baseline_result]
        baseline_statuses.update(paired_baseline_statuses)
    else:
        supplemental_baseline_results = []
    missing_noise_floor_tests = [
        test for test in noise_floor_tests if test not in baseline_statuses
    ]
    if missing_noise_floor_tests:
        noise_floor_baseline_statuses, noise_floor_baseline_result = run_test_suite(
            ROOT, missing_noise_floor_tests, args.timeout
        )
        supplemental_baseline_results.append(noise_floor_baseline_result)
        baseline_statuses.update(noise_floor_baseline_statuses)
    missing_seed_tests = [
        test for test in seed_pipeline_tests if test not in baseline_statuses
    ]
    if missing_seed_tests:
        seed_baseline_statuses, seed_baseline_result = run_test_suite(
            ROOT, missing_seed_tests, args.timeout
        )
        supplemental_baseline_results.append(seed_baseline_result)
        baseline_statuses.update(seed_baseline_statuses)
    missing_rank_tests = [
        test for test in rank_pipeline_tests if test not in baseline_statuses
    ]
    if missing_rank_tests:
        rank_baseline_statuses, rank_baseline_result = run_test_suite(
            ROOT, missing_rank_tests, args.timeout
        )
        supplemental_baseline_results.append(rank_baseline_result)
        baseline_statuses.update(rank_baseline_statuses)
    missing_placement_tests = [
        test for test in placement_tests if test not in baseline_statuses
    ]
    if missing_placement_tests:
        placement_baseline_statuses, placement_baseline_result = run_test_suite(
            ROOT, missing_placement_tests, args.timeout
        )
        supplemental_baseline_results.append(placement_baseline_result)
        baseline_statuses.update(placement_baseline_statuses)
    missing_bias_results_tests = [
        test for test in bias_results_tests if test not in baseline_statuses
    ]
    if missing_bias_results_tests:
        bias_results_baseline_statuses, bias_results_baseline_result = run_test_suite(
            ROOT, missing_bias_results_tests, args.timeout
        )
        supplemental_baseline_results.append(bias_results_baseline_result)
        baseline_statuses.update(bias_results_baseline_statuses)
    missing_pipeline_tests = [
        test for test in pipeline_tests if test not in baseline_statuses
    ]
    if missing_pipeline_tests:
        pipeline_baseline_statuses, pipeline_baseline_result = run_test_suite(
            ROOT, missing_pipeline_tests, args.timeout
        )
        supplemental_baseline_results.append(pipeline_baseline_result)
        baseline_statuses.update(pipeline_baseline_statuses)
    missing_final_score_tests = [
        test for test in final_score_tests if test not in baseline_statuses
    ]
    if missing_final_score_tests:
        final_score_baseline_statuses, final_score_baseline_result = run_test_suite(
            ROOT, missing_final_score_tests, args.timeout
        )
        supplemental_baseline_results.append(final_score_baseline_result)
        baseline_statuses.update(final_score_baseline_statuses)
    baseline_failures = [
        test
        for test, status in baseline_statuses.items()
        if status in {"FAIL", "ERROR"}
    ]
    if baseline_failures or any(
        result != "passed" for result in supplemental_baseline_results
    ):
        print(
            "Supplemental baseline tests failed; aborting before mutant runs. "
            f"Failing tests: {baseline_failures!r}; "
            f"suite results: {supplemental_baseline_results!r}."
        )
        return 1

    source_headers, source_mutants = read_mutation_matrix(MUTATION_MATRIX_PATH)
    output_tests = list(tests)
    for extra_test in [
        *paired_tests,
        *noise_floor_tests,
        *seed_pipeline_tests,
        *rank_pipeline_tests,
        *placement_tests,
        *bias_results_tests,
        *pipeline_tests,
        *final_score_tests,
    ]:
        if extra_test not in output_tests:
            output_tests.append(extra_test)
    results: dict[str, dict[str, object]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_mutant,
                index,
                node_id,
                function_name,
                source,
                output_tests,
                args.timeout,
            ): node_id
            for index, (node_id, function_name) in enumerate(jobs, start=1)
        }
        for future in as_completed(futures):
            result = future.result()
            results[str(result["target"])] = result
            failed_count = len(result["failed"])
            print(
                f'{result["id"]}: {failed_count} test(s) failed - '
                f'{result["target"]}'
            )

    matrix_rows: list[list[object]] = [
        [*MATRIX_HEADERS, *output_tests],
        [
            "Baseline",
            "No mutation",
            *[
                matrix_status(baseline_statuses.get(test, "NOT RUN"))
                for test in output_tests
            ],
        ],
    ]
    probe_matrix_rows: list[list[object]] = [
        [*MATRIX_HEADERS, *output_tests],
        [
            "Baseline",
            "No mutation",
            *[
                matrix_status(baseline_statuses.get(test, "NOT RUN"))
                for test in output_tests
            ],
        ],
    ]
    summary_rows: list[list[object]] = [list(SUMMARY_HEADERS)]
    for node_id, function_name in jobs:
        result = results[node_id]
        target_description = f'{result["id"]}: {node_id}'
        code_change = (
            f'raise RuntimeError("MUTANT-HARNESS: {result["id"]}") '
            f"at {function_name} entry"
        )
        statuses = result["statuses"]
        probe_matrix_rows.append(
            [
                target_description,
                code_change,
                *[
                    matrix_status(statuses.get(test, "NOT RUN"))
                    for test in output_tests
                ],
            ]
        )
    final_score_mutant_results = []
    for mutant_id, description, replacements, expected_failures in FINAL_SCORE_MUTANTS:
        mutant_result = run_poly_source_mutant(
            mutant_id,
            description,
            replacements,
            expected_failures,
            source,
            output_tests,
            args.timeout,
        )
        final_score_mutant_results.append(mutant_result)
        print(
            f'{mutant_result["id"]}: '
            f'{"PASS" if mutant_result["matched"] else "MISMATCH"} - '
            f'{mutant_result["details"]}'
        )
        mutant_statuses = mutant_result["statuses"]
        target_description = (
            f'{mutant_result["id"]}: {mutant_result["description"]}'
        )
        matrix_rows.append(
            [
                target_description,
                str(mutant_result["description"]),
                *[
                    matrix_status(mutant_statuses.get(test, "NOT RUN"))
                    for test in output_tests
                ],
            ]
        )
        summary_rows.append(
            [
                target_description,
                str(mutant_result["description"]),
                len(mutant_result["failed"]),
                str(mutant_result["details"]),
                "Executed: source mutation",
            ]
        )
    paired_mutant_results = []
    for mutant_id, description, before, after, expected_failures in PAIRED_DIFF_MUTANTS:
        mutant_result = run_poly_source_mutant(
            mutant_id,
            description,
            ((before, after),),
            expected_failures,
            source,
            output_tests,
            args.timeout,
        )
        paired_mutant_results.append(mutant_result)
        print(
            f'{mutant_result["id"]}: '
            f'{"PASS" if mutant_result["matched"] else "MISMATCH"} - '
            f'{mutant_result["details"]}'
        )
        mutant_statuses = mutant_result["statuses"]
        target_description = (
            f'{mutant_result["id"]}: {mutant_result["description"]}'
        )
        matrix_rows.append(
            [
                target_description,
                str(mutant_result["description"]),
                *[
                    matrix_status(mutant_statuses.get(test, "NOT RUN"))
                    for test in output_tests
                ],
            ]
        )
        summary_rows.append(
            [
                target_description,
                str(mutant_result["description"]),
                len(mutant_result["failed"]),
                str(mutant_result["details"]),
                "Executed: source mutation",
            ]
        )

    noise_floor_mutant_results = []
    for mutant_id, description, replacements, expected_failures in NOISE_FLOOR_MUTANTS:
        mutant_result = run_poly_source_mutant(
            mutant_id,
            description,
            replacements,
            expected_failures,
            source,
            output_tests,
            args.timeout,
        )
        noise_floor_mutant_results.append(mutant_result)
        print(
            f'{mutant_result["id"]}: '
            f'{"PASS" if mutant_result["matched"] else "MISMATCH"} - '
            f'{mutant_result["details"]}'
        )
        mutant_statuses = mutant_result["statuses"]
        target_description = (
            f'{mutant_result["id"]}: {mutant_result["description"]}'
        )
        matrix_rows.append(
            [
                target_description,
                str(mutant_result["description"]),
                *[
                    matrix_status(mutant_statuses.get(test, "NOT RUN"))
                    for test in output_tests
                ],
            ]
        )
        summary_rows.append(
            [
                target_description,
                str(mutant_result["description"]),
                len(mutant_result["failed"]),
                str(mutant_result["details"]),
                "Executed: source mutation",
            ]
        )

    seed_pipeline_mutant_results = []
    for (
        mutant_id,
        description,
        before,
        after,
        expected_failure,
    ) in SEED_PIPELINE_MUTANTS:
        mutant_result = run_seed_runs_mutant(
            mutant_id,
            description,
            before,
            after,
            expected_failure,
            output_tests,
            args.timeout,
        )
        seed_pipeline_mutant_results.append(mutant_result)
        print(
            f'{mutant_result["id"]}: '
            f'{"PASS" if mutant_result["matched"] else "MISMATCH"} - '
            f'{mutant_result["details"]}'
        )
        mutant_statuses = mutant_result["statuses"]
        target_description = (
            f'{mutant_result["id"]}: {mutant_result["description"]}'
        )
        matrix_rows.append(
            [
                target_description,
                str(mutant_result["description"]),
                *[
                    matrix_status(mutant_statuses.get(test, "NOT RUN"))
                    for test in output_tests
                ],
            ]
        )
        summary_rows.append(
            [
                target_description,
                str(mutant_result["description"]),
                len(mutant_result["failed"]),
                str(mutant_result["details"]),
                "Executed: source mutation",
            ]
        )

    rank_pipeline_mutant_results = []
    for (
        mutant_id,
        description,
        before,
        after,
        expected_failure,
    ) in RANK_PIPELINE_MUTANTS:
        mutant_result = run_seed_runs_mutant(
            mutant_id,
            description,
            before,
            after,
            expected_failure,
            output_tests,
            args.timeout,
        )
        rank_pipeline_mutant_results.append(mutant_result)
        print(
            f'{mutant_result["id"]}: '
            f'{"PASS" if mutant_result["matched"] else "MISMATCH"} - '
            f'{mutant_result["details"]}'
        )
        mutant_statuses = mutant_result["statuses"]
        target_description = (
            f'{mutant_result["id"]}: {mutant_result["description"]}'
        )
        matrix_rows.append(
            [
                target_description,
                str(mutant_result["description"]),
                *[
                    matrix_status(mutant_statuses.get(test, "NOT RUN"))
                    for test in output_tests
                ],
            ]
        )
        summary_rows.append(
            [
                target_description,
                str(mutant_result["description"]),
                len(mutant_result["failed"]),
                str(mutant_result["details"]),
                "Executed: source mutation",
            ]
        )

    placement_mutant_results = []
    for mutant_id, description, replacements, expected_failures in PLACEMENT_MUTANTS:
        mutant_result = run_poly_source_mutant(
            mutant_id,
            description,
            replacements,
            expected_failures,
            source,
            output_tests,
            args.timeout,
        )
        placement_mutant_results.append(mutant_result)
        print(
            f'{mutant_result["id"]}: '
            f'{"PASS" if mutant_result["matched"] else "MISMATCH"} - '
            f'{mutant_result["details"]}'
        )
        mutant_statuses = mutant_result["statuses"]
        target_description = (
            f'{mutant_result["id"]}: {mutant_result["description"]}'
        )
        matrix_rows.append(
            [
                target_description,
                str(mutant_result["description"]),
                *[
                    matrix_status(mutant_statuses.get(test, "NOT RUN"))
                    for test in output_tests
                ],
            ]
        )
        summary_rows.append(
            [
                target_description,
                str(mutant_result["description"]),
                len(mutant_result["failed"]),
                str(mutant_result["details"]),
                "Executed: source mutation",
            ]
        )

    bias_results_mutant_results = []
    for mutant_id, description, replacements, expected_failures in BIAS_RESULTS_MUTANTS:
        mutant_result = run_poly_source_mutant(
            mutant_id,
            description,
            replacements,
            expected_failures,
            source,
            output_tests,
            args.timeout,
        )
        bias_results_mutant_results.append(mutant_result)
        print(
            f'{mutant_result["id"]}: '
            f'{"PASS" if mutant_result["matched"] else "MISMATCH"} - '
            f'{mutant_result["details"]}'
        )
        mutant_statuses = mutant_result["statuses"]
        target_description = (
            f'{mutant_result["id"]}: {mutant_result["description"]}'
        )
        matrix_rows.append(
            [
                target_description,
                str(mutant_result["description"]),
                *[
                    matrix_status(mutant_statuses.get(test, "NOT RUN"))
                    for test in output_tests
                ],
            ]
        )
        summary_rows.append(
            [
                target_description,
                str(mutant_result["description"]),
                len(mutant_result["failed"]),
                str(mutant_result["details"]),
                "Executed: source mutation",
            ]
        )

    pipeline_mutant_results = []
    for mutant_id, description, replacements, expected_failures in PIPELINE_MUTANTS:
        mutant_result = run_poly_source_mutant(
            mutant_id,
            description,
            replacements,
            expected_failures,
            source,
            output_tests,
            args.timeout,
        )
        pipeline_mutant_results.append(mutant_result)
        print(
            f'{mutant_result["id"]}: '
            f'{"PASS" if mutant_result["matched"] else "MISMATCH"} - '
            f'{mutant_result["details"]}'
        )
        mutant_statuses = mutant_result["statuses"]
        target_description = str(mutant_result["description"])
        matrix_rows.append(
            [
                target_description,
                str(mutant_result["description"]),
                *[
                    matrix_status(mutant_statuses.get(test, "NOT RUN"))
                    for test in output_tests
                ],
            ]
        )
        summary_rows.append(
            [
                target_description,
                str(mutant_result["description"]),
                len(mutant_result["failed"]),
                str(mutant_result["details"]),
                "Executed: source mutation",
            ]
        )

    existing_mutants = {str(row[0]) for row in matrix_rows[1:]}
    imported_mutants_to_run = []
    stale_rows = []
    for source_mutant in source_mutants:
        if len(source_mutant) < len(source_headers):
            source_mutant.extend([""] * (len(source_headers) - len(source_mutant)))
        mutation_name = source_mutant[0].strip()
        if not mutation_name or mutation_name in existing_mutants:
            continue

        if mutation_name in STALE_IMPORTED_MUTANTS:
            stale_rows.append(source_mutant)
            continue
        replacements = IMPORTED_MUTATION_REPLACEMENTS.get(mutation_name)
        if replacements is None:
            raise RuntimeError(
                f"No executable mutation is registered for imported row "
                f"{mutation_name!r}."
            )
        imported_mutants_to_run.append(
            (mutation_name, source_mutant[1], replacements)
        )

    imported_results: dict[str, dict[str, object]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_imported_mutant,
                mutation_name,
                description,
                replacements,
                source,
                output_tests,
                args.timeout,
            ): mutation_name
            for mutation_name, description, replacements in imported_mutants_to_run
        }
        for future in as_completed(futures):
            mutation_name = futures[future]
            result = future.result()
            imported_results[mutation_name] = result
            print(
                f'{mutation_name}: {len(result["failed"])} current test(s) '
                f'failed ({result["suite_result"]})'
            )

    for mutation_name, code_change, _ in imported_mutants_to_run:
        result = imported_results[mutation_name]
        mutant_statuses = result["statuses"]
        matrix_rows.append(
            [
                mutation_name,
                code_change,
                *[
                    matrix_status(mutant_statuses.get(test, "NOT RUN"))
                    for test in output_tests
                ],
            ]
        )
        failed_tests = [
            test
            for test in result["failed"]
            if baseline_statuses.get(test) != "FAIL"
        ]
        summary_rows.append(
            [
                mutation_name,
                code_change,
                len(failed_tests),
                (
                    "; ".join(f"{test} failed." for test in failed_tests)
                    if failed_tests
                    else str(result["details"])
                ),
                "Executed: imported source mutation",
            ]
        )
        existing_mutants.add(mutation_name)

    stale_reasons = {
        "final_test_score reverts to model.score (R2) while the curve scores with MSE":
            "The historical patch assumes abs(scorer(...)); current final_test_score returns scorer(...) directly.",
        "final_test_score drops abs() on the score":
            "The historical change is a no-op on current source; executable SCORE001 and SCORE002 cover the abs() regressions.",
        "add_noise_floor sign not flipped":
            "Covered by executed mutant N001.",
        "add_noise_floor uses test residuals":
            "Covered by executed mutant N002.",
        "add_noise_floor drawn before plot_curve (red line)":
            "Covered by executed mutant N003.",
    }
    for stale_row in stale_rows:
        summary_rows.append(
            [
                stale_row[0],
                stale_row[1],
                0,
                stale_reasons[stale_row[0]],
                "Stale: not executed",
            ]
        )

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    executed_results = [
        *final_score_mutant_results,
        *paired_mutant_results,
        *noise_floor_mutant_results,
        *seed_pipeline_mutant_results,
        *rank_pipeline_mutant_results,
        *placement_mutant_results,
        *bias_results_mutant_results,
        *pipeline_mutant_results,
        *imported_results.values(),
    ]
    executed_mutants = len(executed_results)
    killed = sum(
        any(
            status == "FAIL" and baseline_statuses.get(test) != "FAIL"
            for test, status in result["statuses"].items()
        )
        for result in executed_results
    )
    survived = executed_mutants - killed
    probe_count = len(results)
    errors = sum(
        result["suite_result"] == "timed out"
        or "ERROR" in result["statuses"].values()
        for result in [*results.values(), *executed_results]
    )
    summary_rows = [
        ["Metric", "Count"],
        ["Executed", executed_mutants],
        ["Killed", killed],
        ["Survived", survived],
        ["Probes", probe_count],
        ["Stale", len(stale_rows)],
        [],
        *summary_rows,
    ]
    write_workbook(
        output_path,
        matrix_rows,
        summary_rows,
        probe_matrix_rows,
        len(output_tests),
    )
    print(
        f"Report: {output_path}\n"
        f"Executed source mutations: {executed_mutants}; killed: {killed}; "
        f"survived: {survived}; entry-failure probes: {probe_count}; "
        f"stale rows: {len(stale_rows)}; "
        f"suite errors: {errors}; baseline: {baseline_result}."
    )
    paired_mutants_matched = all(
        mutant["matched"] for mutant in paired_mutant_results
    )
    noise_floor_mutants_matched = all(
        mutant["matched"] for mutant in noise_floor_mutant_results
    )
    seed_pipeline_mutants_matched = all(
        mutant["matched"] for mutant in seed_pipeline_mutant_results
    )
    rank_pipeline_mutants_matched = all(
        mutant["matched"] for mutant in rank_pipeline_mutant_results
    )
    placement_mutants_matched = all(
        mutant["matched"] for mutant in placement_mutant_results
    )
    bias_results_mutants_matched = all(
        mutant["matched"] for mutant in bias_results_mutant_results
    )
    pipeline_mutants_matched = all(
        mutant["matched"] for mutant in pipeline_mutant_results
    )
    final_score_mutants_matched = all(
        mutant["matched"] for mutant in final_score_mutant_results
    )
    return (
        0
        if baseline_result == "passed"
        and errors == 0
        and len(imported_results) == len(imported_mutants_to_run)
        and final_score_mutants_matched
        and paired_mutants_matched
        and noise_floor_mutants_matched
        and seed_pipeline_mutants_matched
        and rank_pipeline_mutants_matched
        and placement_mutants_matched
        and bias_results_mutants_matched
        and pipeline_mutants_matched
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
