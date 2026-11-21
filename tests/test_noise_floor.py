import ast
from collections import Counter
import inspect

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import src.poly as poly
import src.poly_ex as poly_ex
from src.poly import add_noise_floor


def test_add_noise_floor_adds_labeled_line_and_closes_figure():
    floor = 2.5
    fig, ax = plt.subplots()
    initial_line_count = len(ax.lines)

    try:
        add_noise_floor(ax, floor)

        assert len(ax.lines) == initial_line_count + 1
        np.testing.assert_array_equal(ax.lines[-1].get_ydata(), [-floor, -floor])
        assert str(floor) in ax.lines[-1].get_label()
        assert ax.lines[-1].get_color() == "grey"
        assert plt.fignum_exists(fig.number)
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    ("floor", "expected_line", "expected_label"),
    [
        (0.0, 0.0, "0.0"),
        (0.5, -0.5, "-0.5"),
        (-0.5, -0.5, "-0.5"),
    ],
)
def test_add_noise_floor_handles_zero_small_positive_and_negative_values(
    floor, expected_line, expected_label
):
    fig, ax = plt.subplots()
    try:
        add_noise_floor(ax, floor)

        np.testing.assert_array_equal(
            ax.lines[-1].get_ydata(), [expected_line, expected_line]
        )
        assert ax.lines[-1].get_label() == expected_label
    finally:
        plt.close(fig)


def _main_guard():
    module = ast.parse(inspect.getsource(poly_ex))
    for node in module.body:
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == "__name__"
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Eq)
            and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value == "__main__"
        ):
            return node
    raise AssertionError("poly_ex.py must have a __main__ block.")


def _function_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def test_poly_library_has_no_main_block():
    module = ast.parse(inspect.getsource(poly))

    assert not any(
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
        and len(node.test.ops) == 1
        and isinstance(node.test.ops[0], ast.Eq)
        and len(node.test.comparators) == 1
        and isinstance(node.test.comparators[0], ast.Constant)
        and node.test.comparators[0].value == "__main__"
        for node in module.body
    )


def _axis_name(node):
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and isinstance(node.slice, ast.Constant)
    ):
        return f"{node.value.id}[{node.slice.value}]"
    return None


def test_main_uses_training_cv_noise_floor_not_test_residuals():
    calls = [
        node
        for node in ast.walk(_main_guard())
        if isinstance(node, ast.Call)
        and _function_name(node.func) == "add_noise_floor"
    ]

    assert len(calls) == 4
    assert Counter(
        argument.id
        for call in calls
        for argument in call.args[1:2]
        if isinstance(argument, ast.Name)
    ) == Counter(
        {
            "big_cv_rmse": 2,
            "small_cv_rmse": 2,
        }
    )


def test_main_draws_overlay_noise_floor_after_pipeline_curves():
    calls = [
        node
        for node in ast.walk(_main_guard())
        if isinstance(node, ast.Call)
    ]
    for axis in ("axe[0]", "axe[1]"):
        plot_lines = [
            call.lineno
            for call in calls
            if _function_name(call.func) == "plot_curve"
            and any(
                keyword.arg == "ax" and _axis_name(keyword.value) == axis
                for keyword in call.keywords
            )
        ]
        floor_lines = [
            call.lineno
            for call in calls
            if _function_name(call.func) == "add_noise_floor"
            and call.args
            and _axis_name(call.args[0]) == axis
        ]

        assert len(plot_lines) == 3
        assert len(floor_lines) == 1
        assert max(plot_lines) < floor_lines[0]
