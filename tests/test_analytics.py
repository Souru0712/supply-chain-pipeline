"""Unit tests for analytics modules 8–12."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_weekly_df():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2020-01-06", periods=200, freq="W-MON")
    prices = 5.0 + np.cumsum(rng.normal(0, 0.05, 200))
    prices = np.clip(prices, 2.0, 10.0)
    return pd.DataFrame({
        "week_start": dates,
        "price_avg_weekly": prices,
        "price_min_weekly": prices - 0.1,
        "price_max_weekly": prices + 0.1,
        "macro_ppi": 220.0 + rng.normal(0, 2, 200),
        "di_ppi": 100.0 + rng.normal(0, 1, 200),
        "week_of_year": dates.isocalendar().week.astype(int).values,
        "price_lag_1w": np.concatenate([[np.nan], prices[:-1]]),
        "price_lag_4w": np.concatenate([[np.nan]*4, prices[:-4]]),
        "price_hist_avg_4w": pd.Series(prices).rolling(4).mean().values,
        "price_hist_avg_12w": pd.Series(prices).rolling(12).mean().values,
        "price_hist_avg_52w": pd.Series(prices).rolling(52).mean().values,
    })


# ---------------------------------------------------------------------------
# Script 9 — Margin model
# ---------------------------------------------------------------------------

def test_margin_transport_cost_positive():
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location(
        "margin_model", REPO_ROOT / "scripts" / "9.margin_model.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["margin_model"] = mod
    spec.loader.exec_module(mod)

    cost = mod.transportation_cost_per_unit(3.80, 200, 6.5, 1000)
    assert cost > 0
    assert cost < 5.0  # sanity: should be < $5/bu for a 200-mile haul


def test_margin_carry_cost_scales_with_rate():
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location(
        "margin_model2", REPO_ROOT / "scripts" / "9.margin_model.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["margin_model2"] = mod
    spec.loader.exec_module(mod)

    low = mod.carry_cost_per_unit(5.0, 0.01, 30)
    high = mod.carry_cost_per_unit(5.0, 0.10, 30)
    assert high > low


def test_margin_build_cost_stack_columns(sample_weekly_df):
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location(
        "margin_model3", REPO_ROOT / "scripts" / "9.margin_model.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["margin_model3"] = mod
    spec.loader.exec_module(mod)

    result = mod.build_cost_stack(sample_weekly_df, mod.DEFAULT_PARAMS)
    for col in ["transport_cost", "energy_cost", "carry_cost", "total_landed_cost"]:
        assert col in result.columns
    assert (result["total_landed_cost"] > result["price_avg_weekly"]).all()


# ---------------------------------------------------------------------------
# Script 10 — Disruption score
# ---------------------------------------------------------------------------

def test_disruption_score_range(sample_weekly_df):
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location(
        "disruption", REPO_ROOT / "scripts" / "10.disruption_score.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["disruption"] = mod
    spec.loader.exec_module(mod)

    scored = mod.composite_disruption_score(sample_weekly_df)
    valid = scored["disruption_score"].dropna()
    assert len(valid) > 0
    assert valid.between(0, 100).all()


def test_disruption_alert_levels():
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location(
        "disruption2", REPO_ROOT / "scripts" / "10.disruption_score.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["disruption2"] = mod
    spec.loader.exec_module(mod)

    assert mod.determine_alert_level(20)["level"] == "LOW"
    assert mod.determine_alert_level(65)["level"] == "MODERATE"
    assert mod.determine_alert_level(80)["level"] == "HIGH"
    assert mod.determine_alert_level(95)["level"] == "EMERGENCY"


# ---------------------------------------------------------------------------
# Script 11 — Procurement optimizer
# ---------------------------------------------------------------------------

def test_procurement_seasonal_curve_keys(sample_weekly_df):
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location(
        "procurement", REPO_ROOT / "scripts" / "11.procurement_optimizer.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["procurement"] = mod
    spec.loader.exec_module(mod)

    stats = mod.build_seasonal_curve(sample_weekly_df)
    assert len(stats) > 0
    for v in stats.values():
        assert "median" in v and "p25" in v and "p75" in v


def test_procurement_lp_sums_to_volume(sample_weekly_df):
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location(
        "procurement2", REPO_ROOT / "scripts" / "11.procurement_optimizer.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["procurement2"] = mod
    spec.loader.exec_module(mod)

    stats = mod.build_seasonal_curve(sample_weekly_df)
    curve = mod.build_forward_price_curve(stats, 5.0, 1)
    annual_vol = 100_000.0
    constraints = {
        "min_weekly_purchase": annual_vol * 0.005,
        "max_weekly_purchase": annual_vol * 0.10,
        "max_forward_commitment_pct": 0.20,
    }
    schedule = mod.optimise_procurement(curve, annual_vol, constraints)
    total = schedule["recommended_purchase"].sum()
    assert abs(total - annual_vol) / annual_vol < 0.01  # within 1%


# ---------------------------------------------------------------------------
# Script 12 — Risk score
# ---------------------------------------------------------------------------

def test_risk_score_range(sample_weekly_df):
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location(
        "risk_score", REPO_ROOT / "scripts" / "12.risk_score.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["risk_score"] = mod
    spec.loader.exec_module(mod)

    weights = mod.DEFAULT_WEIGHTS
    scored = mod.compute_risk_scores(sample_weekly_df, weights)
    assert scored["composite_risk"].between(0, 100).all()
    for dim in ["supply_risk", "cost_risk", "logistics_risk", "demand_risk"]:
        assert dim in scored.columns


def test_risk_classify():
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location(
        "risk_score2", REPO_ROOT / "scripts" / "12.risk_score.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["risk_score2"] = mod
    spec.loader.exec_module(mod)

    assert mod.classify_risk(10)["level"] == "MINIMAL"
    assert mod.classify_risk(50)["level"] == "MODERATE"
    assert mod.classify_risk(85)["level"] == "CRITICAL"
