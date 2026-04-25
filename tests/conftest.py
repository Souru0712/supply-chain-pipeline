"""Shared pytest fixtures for the supply-chain-pipeline test suite."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"


def _import_script(module_name: str, filename: str):
    """Import a numerically-named script using importlib."""
    spec = importlib.util.spec_from_file_location(module_name, SCRIPTS_DIR / filename)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@pytest.fixture(scope="session")
def train_module():
    return _import_script("train_forecast", "6.train_forecast.py")


@pytest.fixture(scope="session")
def inference_module():
    return _import_script("inference", "7.inference.py")


@pytest.fixture(scope="session")
def sample_data_module():
    return _import_script("generate_sample_data", "0.generate_sample_data.py")


@pytest.fixture
def weekly_df() -> pd.DataFrame:
    """Minimal training DataFrame with the expected schema."""
    rng = np.random.default_rng(0)
    n = 100
    dates = pd.date_range("2020-01-06", periods=n, freq="W-MON")
    prices = 5.0 + np.cumsum(rng.normal(0, 0.1, n))

    df = pd.DataFrame(
        {
            "week_start": dates,
            "price_avg_weekly": prices,
            "price_min_weekly": prices - 0.1,
            "price_max_weekly": prices + 0.1,
            "n_reports_in_week": rng.integers(3, 10, n).astype(float),
            "price_lag_1w": prices - rng.normal(0, 0.05, n),
            "price_lag_4w": prices - rng.normal(0, 0.1, n),
            "price_lag_13w": prices - rng.normal(0, 0.2, n),
            "price_hist_avg_12w": prices - rng.normal(0, 0.08, n),
            "price_hist_std_12w": np.abs(rng.normal(0.15, 0.03, n)),
            "di_ppi": 100.0 + rng.normal(0, 1, n),
            "macro_ppi": 230.0 + rng.normal(0, 2, n),
            "year": dates.year.astype(float),
            "week_of_year": dates.isocalendar().week.astype(float),  # type: ignore[union-attr]
            "month": dates.month.astype(float),
            "quarter": dates.quarter.astype(float),
        }
    )
    return df
