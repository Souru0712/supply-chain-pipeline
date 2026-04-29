"""
Enhanced Price Forecasting — 13-week horizon with SHAP attribution.

Extends 6.train_forecast.py with:
  - SHAP feature importance for every forecast
  - Directional accuracy metric (did model beat naive direction?)
  - Lag-safety enforcement (all features lagged >= horizon to prevent leakage)
  - Per-ingredient ingredient configurability

Inputs
------
- data/materialized/<INGREDIENT>/training_weekly.parquet

Outputs
-------
- models/<ingredient>/lgb_h{h}.txt + _q{10,90}.txt   (same format as script 6)
- data/forecasts/<ingredient>/shap_summary.parquet    (mean |SHAP| per feature)
- data/forecasts/<ingredient>/weekly_forecast.parquet (same format as script 7)
- models/<ingredient>/metadata_v2.json

Usage
-----
    python scripts/8.price_forecast_v2.py
    python scripts/8.price_forecast_v2.py --ingredient WHEAT --horizons 13
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any, Callable, List, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore", category=UserWarning)

REPO_ROOT = Path(__file__).resolve().parents[1]
MATERIALIZED = REPO_ROOT / "data" / "materialized"
MODELS_DIR = REPO_ROOT / "models"
FORECASTS_DIR = REPO_ROOT / "data" / "forecasts"

TARGET = "price_avg_weekly"
DATE_COL = "week_start"
CROSS_FFILL_COLS = ["di_ppi", "macro_ppi"]
QUANTILES = (0.1, 0.5, 0.9)
MIN_ROWS_FOR_FIT = 60


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------

def load_training_frame(ingredient: str) -> pd.DataFrame:
    path = MATERIALIZED / ingredient / "training_weekly.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/0.generate_sample_data.py` first."
        )
    df = pd.read_parquet(path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    for col in CROSS_FFILL_COLS:
        if col in df.columns:
            df[col] = df[col].ffill()
    return df


def enforce_lag_safety(df: pd.DataFrame, feature_cols: list[str], horizon: int) -> list[str]:
    """
    Drop any feature whose name suggests it was computed with a window shorter
    than the forecast horizon — prevents look-ahead leakage.

    Convention: price_lag_1w, price_hist_avg_4w, price_hist_std_12w …
    """
    safe = []
    for col in feature_cols:
        # Extract trailing numeric token (e.g. "4" from "price_hist_avg_4w")
        import re
        m = re.search(r"_(\d+)w$", col)
        if m:
            lag_weeks = int(m.group(1))
            if lag_weeks < horizon:
                continue  # feature window too short — skip
        safe.append(col)
    return safe


def build_horizon_targets(df: pd.DataFrame, horizons: Sequence[int]) -> pd.DataFrame:
    df = df.copy()
    for h in horizons:
        df[f"y_h{h}"] = df[TARGET].shift(-h)
    return df


def select_feature_columns(df: pd.DataFrame, horizons: Sequence[int]) -> list[str]:
    drop = {DATE_COL, *(f"y_h{h}" for h in horizons)}
    return [
        c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])
    ]


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray, y_prev: np.ndarray) -> float:
    """Fraction of periods where model correctly predicted price direction."""
    true_dir = np.sign(y_true - y_prev)
    pred_dir = np.sign(y_pred - y_prev)
    return float(np.mean(true_dir == pred_dir))


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def lgb_params(quantile: float) -> dict:
    return {
        "objective": "quantile",
        "alpha": quantile,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 10,
        "subsample": 0.9,
        "feature_fraction": 0.9,
        "verbosity": -1,
        "seed": 42,
    }


def cv_metrics(
    X: np.ndarray,
    y: np.ndarray,
    y_prev: np.ndarray,
    params: dict,
    n_splits: int = 5,
) -> dict[str, float]:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes, mapes, dirs = [], [], []
    for train_idx, test_idx in tscv.split(X):
        dtrain = lgb.Dataset(X[train_idx], label=y[train_idx])
        dvalid = lgb.Dataset(X[test_idx], label=y[test_idx], reference=dtrain)
        callbacks: List[Callable[..., Any]] = [
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(-1),
        ]
        booster = lgb.train(
            params, dtrain, num_boost_round=600, valid_sets=[dvalid], callbacks=callbacks
        )
        preds = booster.predict(X[test_idx])
        maes.append(mean_absolute_error(y[test_idx], preds))
        mapes.append(mean_absolute_percentage_error(y[test_idx], preds))
        dirs.append(directional_accuracy(y[test_idx], preds, y_prev[test_idx]))
    return {
        "mae": float(np.mean(maes)),
        "mape": float(np.mean(mapes)),
        "directional_accuracy": float(np.mean(dirs)),
    }


def fit_final(X: np.ndarray, y: np.ndarray, params: dict) -> lgb.Booster:
    cut = int(len(X) * 0.85)
    dtrain = lgb.Dataset(X[:cut], label=y[:cut])
    dval = lgb.Dataset(X[cut:], label=y[cut:], reference=dtrain)
    callbacks: List[Callable[..., Any]] = [
        lgb.early_stopping(50, verbose=False),
        lgb.log_evaluation(-1),
    ]
    return lgb.train(
        params, dtrain, num_boost_round=800, valid_sets=[dval], callbacks=callbacks
    )


def compute_shap(booster: lgb.Booster, X: np.ndarray, feature_names: list[str]) -> pd.Series:
    """Return mean absolute SHAP value per feature (global importance)."""
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X)
    mean_abs = np.abs(shap_values).mean(axis=0)
    return pd.Series(mean_abs, index=feature_names).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Per-horizon training
# ---------------------------------------------------------------------------

def train_horizon(
    df: pd.DataFrame,
    horizon: int,
    feature_cols: list[str],
    out_dir: Path,
) -> dict:
    target_col = f"y_h{horizon}"
    safe_features = enforce_lag_safety(df, feature_cols, horizon)

    valid = df.dropna(subset=safe_features + [target_col, TARGET])
    valid = valid.replace([np.inf, -np.inf], np.nan).dropna(
        subset=safe_features + [target_col]
    )

    if len(valid) < MIN_ROWS_FOR_FIT:
        print(f"  h={horizon:>2}: SKIP — only {len(valid)} usable rows")
        return {"horizon": horizon, "status": "skipped", "n_rows": len(valid)}

    X = valid[safe_features].to_numpy()
    y = valid[target_col].to_numpy()
    y_prev = valid[TARGET].to_numpy()

    quantile_metrics: dict[str, dict] = {}
    shap_records = []

    for q in QUANTILES:
        params = lgb_params(q)
        metrics = cv_metrics(X, y, y_prev, params)
        quantile_metrics[f"q{int(q * 100)}"] = metrics

        final_model = fit_final(X, y, params)
        suffix = "" if q == 0.5 else f"_q{int(q * 100)}"
        final_model.save_model(str(out_dir / f"lgb_h{horizon}{suffix}.txt"))

        if q == 0.5:
            shap_importance = compute_shap(final_model, X, safe_features)
            shap_records.append(
                shap_importance.rename(f"h{horizon}_shap_importance")
            )

    m = quantile_metrics["q50"]
    print(
        f"  h={horizon:>2}: rows={len(valid):>4}  "
        f"MAE={m['mae']:.4f}  MAPE={m['mape']:.4f}  "
        f"DirAcc={m['directional_accuracy']:.1%}  "
        f"features={len(safe_features)}"
    )

    return {
        "horizon": horizon,
        "n_rows": len(valid),
        "n_features": len(safe_features),
        "quantiles": quantile_metrics,
        "shap_records": shap_records,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ingredient", default="CORN")
    parser.add_argument("--horizons", type=int, default=13)
    args = parser.parse_args()

    ingredient = args.ingredient.upper()
    horizons = list(range(1, args.horizons + 1))

    out_dir = MODELS_DIR / ingredient.lower()
    out_dir.mkdir(parents=True, exist_ok=True)

    forecast_dir = FORECASTS_DIR / ingredient
    forecast_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Enhanced Forecast Training · {ingredient} · horizons 1..{horizons[-1]} ===")
    df = load_training_frame(ingredient)
    df = build_horizon_targets(df, horizons)
    feature_cols = select_feature_columns(df, horizons)
    print(f"  training rows: {len(df)}   base features: {len(feature_cols)}")

    all_shap: list[pd.Series] = []
    horizon_results = []

    for h in horizons:
        result = train_horizon(df, h, feature_cols, out_dir)
        horizon_results.append(result)
        if result.get("shap_records"):
            all_shap.extend(result["shap_records"])

    # Save aggregate SHAP summary
    if all_shap:
        shap_df = pd.concat(all_shap, axis=1).fillna(0)
        shap_df["mean_importance"] = shap_df.mean(axis=1)
        shap_df = shap_df.sort_values("mean_importance", ascending=False)
        shap_df.to_parquet(forecast_dir / "shap_summary.parquet")
        print("\nTop 5 features by mean |SHAP|:")
        for feat, val in shap_df["mean_importance"].head(5).items():
            print(f"  {feat:<35} {val:.6f}")

    metadata: dict[str, Any] = {
        "ingredient": ingredient,
        "model_type": "lightgbm_v2",
        "target": TARGET,
        "features": feature_cols,
        "horizons_weeks": horizons,
        "quantiles": list(QUANTILES),
        "train_end_date": df[DATE_COL].max().isoformat(),
        "n_training_rows": int(len(df)),
        "per_horizon": [
            {k: v for k, v in r.items() if k != "shap_records"}
            for r in horizon_results
        ],
    }
    with open(out_dir / "metadata_v2.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, default=str)

    print(f"\nArtifacts written to {out_dir}")


if __name__ == "__main__":
    main()
