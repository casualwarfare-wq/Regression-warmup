"""Regression warmup source package."""

import sys
from pathlib import Path

_SOURCE_DIR = str(Path(__file__).resolve().parent)
if _SOURCE_DIR not in sys.path:
    sys.path.insert(0, _SOURCE_DIR)

from . import poly, seed_runs, unregLS, degree_sweep, degreeplot

__all__ = ["poly", "seed_runs", "unregLS", 'degree_sweep','degreeplot']
