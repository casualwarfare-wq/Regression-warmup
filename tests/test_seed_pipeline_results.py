import numpy as np

from src.seed_runs import seed_pipeline_results


RMSE_TABLE = {
    (1, 101): {"A": 10.0, "B": 10.01, "C": 20.0},
    (2, 102): {"A": 10.0, "B": 10.01, "C": 20.0},
    (3, 103): {"A": 10.0, "B": 10.01, "C": 20.0},
    (4, 104): {"A": 100.0, "B": 1.0, "C": 101.0},
    (5, 105): {"A": 5.0, "B": 5.0, "C": 10.0},
}


def test_seed_pipeline_pairwise_diffs_match_hand_built_table():
    rmse_table = {
        (1, 101): {"A": 1.0, "B": 3.0, "C": 2.0},
        (2, 102): {"A": 1.0, "B": 3.0, "C": 2.0},
        (3, 103): {"A": 1.0, "B": 3.0, "C": 2.0},
    }
    expected = {
        "A vs. B": (-2.0, 0.0, 3),
        "A vs. C": (-1.0, 0.0, 3),
        "B vs. C": (1.0, 0.0, 0),
    }

    _, diffs = seed_pipeline_results(rmse_table)

    assert list(diffs.keys()) == list(expected.keys())
    for key, (mean, se, wins) in expected.items():
        actual = diffs[key]
        assert np.isclose(actual[0], mean)
        assert np.isclose(actual[1], se)
        assert actual[2] == wins


def test_seed_pipeline_win_counts_match_hand_calculated_winners():
    # A wins seeds 1-3, B wins seed 4, and seed 5 is tied.
    win_counts, _ = seed_pipeline_results(RMSE_TABLE)

    assert win_counts == {"A": 3, "B": 1, "C": 0}


def test_win_tally_can_differ_from_mean_rmse_ranking():
    rmse_table = {
        (1, 101): {"A": 10.0, "B": 10.01, "C": 20.0},
        (2, 102): {"A": 10.0, "B": 10.01, "C": 20.0},
        (3, 103): {"A": 10.0, "B": 10.01, "C": 20.0},
        (4, 104): {"A": 100.0, "B": 1.0, "C": 101.0},
        (5, 105): {"A": 100.0, "B": 1.0, "C": 101.0},
    }
    pipelines = tuple(next(iter(rmse_table.values())))

    win_counts, _ = seed_pipeline_results(rmse_table)

    mean_rmses = {
        pipeline: np.mean([row[pipeline] for row in rmse_table.values()])
        for pipeline in pipelines
    }

    assert mean_rmses["B"] < mean_rmses["A"]
    assert win_counts["A"] > win_counts["B"]


def test_exact_ties_award_no_pipeline_a_win():
    tied_seed = {
        (1, 101): {"A": 5.0, "B": 5.0, "C": 10.0},
        (2, 102): {"A": 4.0, "B": 5.0, "C": 6.0},
    }

    win_counts, _ = seed_pipeline_results(tied_seed)

    assert win_counts == {"A": 1, "B": 0, "C": 0}


def test_win_count_sum_excludes_tied_seeds():
    win_counts, _ = seed_pipeline_results(RMSE_TABLE)

    # Exact ties award no win, so only the four untied seeds are counted.
    assert sum(win_counts.values()) == 4


def test_seed_pipeline_order_comes_from_first_seed_row():
    rmse_table = {
        (1, 101): {"A": 1.0, "B": 2.0, "C": 3.0},
        (2, 102): {"C": 1.0, "B": 2.0, "A": 3.0},
    }

    win_counts, diffs = seed_pipeline_results(rmse_table)

    assert list(win_counts) == ["A", "B", "C"]
    assert list(diffs) == ["A vs. B", "A vs. C", "B vs. C"]


def test_seed_pipeline_third_pair_uses_arithmetic_difference():
    rmse_table = {
        (1, 101): {"A": 1.0, "B": 9.0, "C": 4.0},
        (2, 102): {"A": 2.0, "B": 9.0, "C": 4.0},
    }

    _, diffs = seed_pipeline_results(rmse_table)

    assert diffs["B vs. C"][0] == 5.0
