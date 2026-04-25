"""Tests for the inference pipeline in 7.inference.py."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb


# ---------------------------------------------------------------------------
# load_metadata
# ---------------------------------------------------------------------------


def test_load_metadata_raises_on_missing(inference_module):
    with pytest.raises(FileNotFoundError, match="metadata.json"):
        inference_module.load_metadata("NONEXISTENT_INGREDIENT_XYZ")


def test_load_metadata_returns_dict(inference_module, tmp_path, monkeypatch):
    # Patch MODELS_DIR to point at a temp directory
    ingredient_dir = tmp_path / "testingredient"
    ingredient_dir.mkdir()
    meta = {
        "ingredient": "TESTINGREDIENT",
        "features": ["price_avg_weekly"],
        "horizons_weeks": [1],
    }
    (ingredient_dir / "metadata.json").write_text(json.dumps(meta))

    monkeypatch.setattr(inference_module, "MODELS_DIR", tmp_path)
    result = inference_module.load_metadata("TESTINGREDIENT")
    assert result["ingredient"] == "TESTINGREDIENT"


# ---------------------------------------------------------------------------
# load_models
# ---------------------------------------------------------------------------


def test_load_models_empty_when_no_files(inference_module, tmp_path, monkeypatch):
    (tmp_path / "testingredient").mkdir()
    monkeypatch.setattr(inference_module, "MODELS_DIR", tmp_path)
    models = inference_module.load_models("TESTINGREDIENT", [1, 2])
    assert models == {}


def test_load_models_loads_existing_json(inference_module, tmp_path, monkeypatch):
    ingredient_dir = tmp_path / "testingredient"
    ingredient_dir.mkdir()

    # Train and save a tiny booster
    rng = np.random.default_rng(99)
    X = rng.standard_normal((60, 3)).astype(np.float32)
    y = X[:, 0].astype(np.float32)
    params = {"objective": "reg:quantileerror", "quantile_alpha": 0.5, "verbosity": 0}
    dtrain = xgb.DMatrix(X, label=y)
    booster = xgb.train(params, dtrain, num_boost_round=5)
    booster.save_model(str(ingredient_dir / "xgb_h1.json"))

    monkeypatch.setattr(inference_module, "MODELS_DIR", tmp_path)
    models = inference_module.load_models("TESTINGREDIENT", [1])
    assert "h1_q50" in models
    assert isinstance(models["h1_q50"], xgb.Booster)


# ---------------------------------------------------------------------------
# load_feature_row
# ---------------------------------------------------------------------------


def test_load_feature_row_raises_on_missing_parquet(
    inference_module, tmp_path, monkeypatch
):
    monkeypatch.setattr(inference_module, "MATERIALIZED", tmp_path)
    with pytest.raises(FileNotFoundError):
        inference_module.load_feature_row("TESTINGREDIENT", ["price_avg_weekly"], None)


def test_load_feature_row_returns_last_row(inference_module, tmp_path, monkeypatch):
    ing_dir = tmp_path / "TESTINGREDIENT"
    ing_dir.mkdir()
    dates = pd.date_range("2024-01-01", periods=10, freq="W-MON")
    df = pd.DataFrame(
        {
            "week_start": dates,
            "price_avg_weekly": np.arange(10, dtype=float),
            "di_ppi": 100.0,
            "macro_ppi": 220.0,
        }
    )
    df.to_parquet(ing_dir / "training_weekly.parquet", index=False)

    monkeypatch.setattr(inference_module, "MATERIALIZED", tmp_path)
    row, cutoff = inference_module.load_feature_row(
        "TESTINGREDIENT", ["price_avg_weekly"], None
    )
    assert float(row["price_avg_weekly"].iloc[0]) == pytest.approx(9.0)
    assert cutoff == dates[-1]


def test_load_feature_row_respects_as_of(inference_module, tmp_path, monkeypatch):
    ing_dir = tmp_path / "TESTINGREDIENT"
    ing_dir.mkdir()
    dates = pd.date_range("2024-01-01", periods=20, freq="W-MON")
    df = pd.DataFrame(
        {"week_start": dates, "price_avg_weekly": np.arange(20, dtype=float)}
    )
    df.to_parquet(ing_dir / "training_weekly.parquet", index=False)

    monkeypatch.setattr(inference_module, "MATERIALIZED", tmp_path)
    cutoff_str = "2024-02-01"
    row, cutoff = inference_module.load_feature_row(
        "TESTINGREDIENT", ["price_avg_weekly"], cutoff_str
    )
    assert cutoff <= pd.Timestamp(cutoff_str)


# ---------------------------------------------------------------------------
# Dashboard risk computation (imported directly)
# ---------------------------------------------------------------------------


def test_dashboard_risk_compute():
    """Risk score should be in [0, 100] and level should be a valid string."""
    import importlib.util
    import sys

    dashboard_path = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"
    spec = importlib.util.spec_from_file_location("dashboard_app", dashboard_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["dashboard_app"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    rng = np.random.default_rng(7)
    dates = pd.date_range("2024-01-01", periods=52, freq="W-MON")
    hist = pd.DataFrame(
        {"week_start": dates, "price_avg_weekly": 5.0 + rng.normal(0, 0.2, 52)}
    )
    horizons = list(range(1, 14))
    fc = pd.DataFrame(
        {
            "forecast_date": dates[-1],
            "horizon_weeks": horizons,
            "target_week_start": [dates[-1] + pd.Timedelta(weeks=h) for h in horizons],
            "p10": [5.0 - 0.2 * h for h in horizons],
            "p50": [5.0 for _ in horizons],
            "p90": [5.0 + 0.2 * h for h in horizons],
        }
    )
    risk = mod.compute_risk(hist, fc)
    assert 0 <= risk["composite"] <= 100
    assert risk["level"] in {"LOW", "MODERATE", "HIGH"}
    assert risk["color"].startswith("#")
