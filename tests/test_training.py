"""Tests for the training utility functions in 6.train_forecast.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb


# ---------------------------------------------------------------------------
# build_horizon_targets
# ---------------------------------------------------------------------------


def test_build_horizon_targets_adds_columns(train_module, weekly_df):
    horizons = [1, 4, 13]
    result = train_module.build_horizon_targets(weekly_df, horizons)
    for h in horizons:
        assert f"y_h{h}" in result.columns


def test_build_horizon_targets_shift_is_correct(train_module, weekly_df):
    result = train_module.build_horizon_targets(weekly_df, [1])
    # y_h1 at row i should equal price at row i+1
    prices = weekly_df["price_avg_weekly"].values
    y_h1 = result["y_h1"].values
    # Compare non-NaN portion
    valid = ~np.isnan(y_h1)
    np.testing.assert_allclose(y_h1[valid], prices[1:][valid[:-1]], rtol=1e-6)


def test_build_horizon_targets_does_not_mutate_input(train_module, weekly_df):
    cols_before = set(weekly_df.columns)
    train_module.build_horizon_targets(weekly_df, [1, 2])
    assert set(weekly_df.columns) == cols_before


# ---------------------------------------------------------------------------
# select_feature_columns
# ---------------------------------------------------------------------------


def test_select_feature_columns_excludes_date_and_targets(train_module, weekly_df):
    horizons = [1, 2]
    df = train_module.build_horizon_targets(weekly_df, horizons)
    cols = train_module.select_feature_columns(df, horizons)
    assert "week_start" not in cols
    for h in horizons:
        assert f"y_h{h}" not in cols


def test_select_feature_columns_all_numeric(train_module, weekly_df):
    df = train_module.build_horizon_targets(weekly_df, [1])
    cols = train_module.select_feature_columns(df, [1])
    for c in cols:
        assert pd.api.types.is_numeric_dtype(df[c]), f"{c} is not numeric"


def test_select_feature_columns_returns_list(train_module, weekly_df):
    df = train_module.build_horizon_targets(weekly_df, [1])
    result = train_module.select_feature_columns(df, [1])
    assert isinstance(result, list)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# xgb_params
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("q", [0.1, 0.5, 0.9])
def test_xgb_params_structure(train_module, q):
    params = train_module.xgb_params(q)
    assert params["objective"] == "reg:quantileerror"
    assert params["quantile_alpha"] == pytest.approx(q)
    assert "learning_rate" in params
    assert "max_depth" in params


# ---------------------------------------------------------------------------
# cv_metrics — lightweight single-split smoke test
# ---------------------------------------------------------------------------


def test_cv_metrics_returns_expected_keys(train_module):
    rng = np.random.default_rng(1)
    X = rng.standard_normal((80, 5)).astype(np.float32)
    y = (X[:, 0] * 2 + rng.standard_normal(80) * 0.1).astype(np.float32)
    params = train_module.xgb_params(0.5)
    metrics = train_module.cv_metrics(X, y, params, n_splits=2)
    assert set(metrics.keys()) == {"mae", "rmse", "mape"}
    for v in metrics.values():
        assert v >= 0.0


# ---------------------------------------------------------------------------
# fit_final — smoke test
# ---------------------------------------------------------------------------


def test_fit_final_returns_booster(train_module):
    rng = np.random.default_rng(2)
    X = rng.standard_normal((80, 5)).astype(np.float32)
    y = (X[:, 0] + rng.standard_normal(80) * 0.1).astype(np.float32)
    booster = train_module.fit_final(X, y, train_module.xgb_params(0.5))
    assert isinstance(booster, xgb.Booster)
    preds = booster.predict(xgb.DMatrix(X))
    assert preds.shape == (80,)


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------


def test_generate_corn_prices_shape(sample_data_module):
    df = sample_data_module.generate_corn_prices(n_weeks=120)
    assert len(df) == 120
    assert "week_start" in df.columns
    assert "price_avg_weekly" in df.columns


def test_generate_corn_prices_no_future_leakage(sample_data_module):
    df = sample_data_module.generate_corn_prices(n_weeks=120)
    # price_lag_1w at row i should equal price_avg_weekly at row i-1
    lag1 = df["price_lag_1w"].values
    price = df["price_avg_weekly"].values
    valid = ~np.isnan(lag1)
    np.testing.assert_allclose(lag1[valid], price[:-1][valid[1:]], rtol=1e-6)


def test_generate_corn_prices_price_range(sample_data_module):
    df = sample_data_module.generate_corn_prices(n_weeks=200)
    assert df["price_avg_weekly"].between(2.0, 12.0).all()
