import runpy
import sys

import pytest


def test_main_rejects_mismatched_seed_list_lengths(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scripts/seed_runs.py",
            "--poly_seeds",
            "1",
            "2",
            "--p_seeds",
            "3",
            "--t_seeds",
            "4",
            "5",
        ],
    )

    with pytest.raises(
        ValueError,
        match="p_seeds, t_seeds and poly_seeds must have the same length",
    ):
        runpy.run_path("src/seed_results.py", run_name="__main__")
