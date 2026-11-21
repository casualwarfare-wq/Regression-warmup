import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_allclose

from src.poly import plot_curve


def test_plot_curve_plots_fold_means_and_parameter_name():
    alphas = np.array([0.01, 0.1, 1.0, 10.0])
    train_scores = np.array(
        [[0.5, 0.6], [0.7, 0.8], [0.8, 0.9], [0.6, 0.7]]
    )
    validation_scores = np.array(
        [[0.4, 0.5], [0.6, 0.7], [0.7, 0.8], [0.5, 0.6]]
    )
    fig, ax = plt.subplots()

    try:
        display = plot_curve(
            (alphas, train_scores, validation_scores),
            parameter="ridge__alpha",
            ax=ax,
        )

        assert display.ax_ is ax
        assert len(ax.lines) == 2
        assert_allclose(ax.lines[0].get_xdata(), alphas)
        assert_allclose(ax.lines[0].get_ydata(), train_scores.mean(axis=1))
        assert ax.lines[0].get_label() == "Train"
        assert_allclose(ax.lines[1].get_xdata(), alphas)
        assert_allclose(ax.lines[1].get_ydata(), validation_scores.mean(axis=1))
        assert ax.lines[1].get_label() == "Test"
        assert ax.get_xlabel() == "ridge__alpha"
    finally:
        plt.close(fig)


def test_plot_curve_uses_score_argument_as_axis_label():
    scores = np.array([[0.4, 0.5], [0.6, 0.7]])
    fig, ax = plt.subplots()

    try:
        plot_curve(
            (np.array([0.1, 1.0]), scores, scores),
            score="Negative Root Mean Squared Error",
            ax=ax,
        )

        assert ax.get_ylabel() == "Negative Root Mean Squared Error"
    finally:
        plt.close(fig)
