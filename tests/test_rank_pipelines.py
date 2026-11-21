from src.seed_runs import rank_pipelines


def test_rank_pipelines_orders_by_lowest_error_not_closest_to_floor():
    cv_rmses = {"lowest_error": 2.0, "near_floor": 8.0, "highest_error": 14.0}
    noise_floor = 9.0
    win_counts = {"lowest_error": 3, "near_floor": 2, "highest_error": 1}

    lowest_error_order = sorted(cv_rmses, key=cv_rmses.get)
    closest_to_floor_order = sorted(
        cv_rmses, key=lambda name: abs(cv_rmses[name] - noise_floor)
    )

    assert lowest_error_order == ["lowest_error", "near_floor", "highest_error"]
    assert closest_to_floor_order == ["near_floor", "highest_error", "lowest_error"]
    assert rank_pipelines(cv_rmses, {}, win_counts) == (
        [
            ("lowest_error", 2.0, 3),
            ("near_floor", 8.0, 2),
            ("highest_error", 14.0, 1),
        ],
        {},
    )
