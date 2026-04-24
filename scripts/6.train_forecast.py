"""
Train a 13-week direct multi-output price forecast model.

Design
------
Instead of recursively predicting day-by-day (error compounds), we train one
LightGBM model per horizon h in 1..13 weeks. Each horizon model learns:

    y_{t+h}  =  f_h( features available at time t )

For each horizon we fit three quantile regressors (p10, p50, p90) so the
dashboard can draw a prediction band, not just a point.

Inputs
------
- data/materialized/<INGREDIENT>/training_weekly.parquet
  (produced by dbt model `mart_corn_training_weekly`)

Outputs
-------
- models/<ingredient>/lgbm_h{h}_q{q}.txt         (26 files: 13 horizons x 2 non-median)
- models/<ingredient>/lgbm_h{h}.txt               (the median / point model)
- models/<ingredient>/metadata.json               (features, metrics, train_end_date)
- mlruns/                                         (local MLflow tracking store)

Usage
-----
    python scripts/6.train_forecast.py                 # defaults to CORN
    python scripts/6.train_forecast.py --ingredient CORN --horizons 13
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path
from typing import Sequence

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
MATERIALIZED = REPO_ROOT / "data" / "materialized"
MODELS_DIR = REPO_ROOT / "models"
MLFLOW_DIR = REPO_ROOT / "mlruns"

TARGET = "price_avg_weekly"
DATE_COL = "week_start"
CROSS_FFILL_COLS = ["di_ppi", "macro_ppi"]
QUANTILES = (0.1, 0.5, 0.9)
MIN_ROWS_FOR_FIT = 60   # ~1 year of weekly data


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------
def load_training_frame(ingredient: str) -> pd.DataFrame:
    path = MATERIALIZED / ingredient / "training_weekly.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `dbt run --select mart_corn_training_weekly` first."
        )

    df = pd.read_parquet(path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    # Forward-fill cross features — macro PPI is monthly, dollar index is
    # daily-but-gappy. Only ffill; no bfill so we never use future info.
    for col in CROSS_FFILL_COLS:
        if col in df.columns:
            df[col] = df[col].ffill()

    return df


def build_horizon_targets(df: pd.DataFrame, horizons: Sequence[int]) -> pd.DataFrame:
    """Attach y_h columns (h weeks ahead)."""
    df = df.copy()
    for h in horizons:
        df[f"y_h{h}"] = df[TARGET].shift(-h)
    return df


def select_feature_columns(df: pd.DataFrame, horizons: Sequence[int]) -> list[str]:
    """Every column except date, target-family, and future horizon targets."""
    drop = {DATE_COL, *(f"y_h{h}" for h in horizons)}
    # price_avg_weekly is allowed as a feature (value at time t, not future)
    return [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def lgbm_params(quantile: float) -> dict:
    return {
        "objective": "quantile",
        "alpha": quantile,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_data_in_leaf": 10,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 5,
        "verbose": -1,
        "random_state": 42,
    }


def cv_metrics(
    X: np.ndarray,
    y: np.ndarray,
    params: dict,
    n_splits: int = 5,
) -> dict[str, float]:
    """Time-series CV. Returns mean MAE / RMSE / MAPE for the given params."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes, rmses, mapes = [], [], []

    for train_idx, test_idx in tscv.split(X):
        dtrain = lgb.Dataset(X[train_idx], label=y[train_idx])
        dtest = lgb.Dataset(X[test_idx], label=y[test_idx], reference=dtrain)
        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=600,
            valid_sets=[dtest],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        preds = booster.predict(X[test_idx])
        maes.append(mean_absolute_error(y[test_idx], preds))
        rmses.append(float(np.sqrt(np.mean((y[test_idx] - preds) ** 2))))
        mapes.append(mean_absolute_percentage_error(y[test_idx], preds))

    return {
        "mae": float(np.mean(maes)),
        "rmse": float(np.mean(rmses)),
        "mape": float(np.mean(mapes)),
    }


def naive_baseline_mae(df: pd.DataFrame, horizon: int, n_splits: int = 5) -> float:
    """Naive forecast: y_{t+h} = y_t. Mean MAE on the same CV folds."""
    valid = df.dropna(subset=[TARGET, f"y_h{horizon}"])
    y = valid[f"y_h{horizon}"].to_numpy()
    y_pred = valid[TARGET].to_numpy()

    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes = []
    for _, test_idx in tscv.split(y):
        maes.append(mean_absolute_error(y[test_idx], y_pred[test_idx]))
    return float(np.mean(maes))


def fit_final(X: np.ndarray, y: np.ndarray, params: dict) -> lgb.Booster:
    """Refit on the full series with a held-out tail for early stopping."""
    cut = int(len(X) * 0.85)
    dtrain = lgb.Dataset(X[:cut], label=y[:cut])
    dval = lgb.Dataset(X[cut:], label=y[cut:], reference=dtrain)
    return lgb.train(
        params,
        dtrain,
        num_boost_round=800,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )


def train_horizon(
    df: pd.DataFrame,
    horizon: int,
    feature_cols: list[str],
    out_dir: Path,
) -> dict:
    """Train the p10 / p50 / p90 models for one horizon. Returns a metrics dict."""
    target_col = f"y_h{horizon}"
    valid = df.dropna(subset=feature_cols + [target_col])
    valid = valid.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols + [target_col])

    if len(valid) < MIN_ROWS_FOR_FIT:
        print(f"  h={horizon:>2}: SKIP — only {len(valid)} usable rows")
        return {"horizon": horizon, "status": "skipped", "n_rows": len(valid)}

    X = valid[feature_cols].to_numpy()
    y = valid[target_col].to_numpy()

    baseline_mae = naive_baseline_mae(df, horizon)

    quantile_metrics = {}
    for q in QUANTILES:
        params = lgbm_params(q)
        metrics = cv_metrics(X, y, params)
        quantile_metrics[f"q{int(q * 100)}"] = metrics

        final_model = fit_final(X, y, params)
        suffix = "" if q == 0.5 else f"_q{int(q * 100)}"
        final_model.save_model(str(out_dir / f"lgbm_h{horizon}{suffix}.txt"))

    median = quantile_metrics["q50"]
    skill = 1 - (median["mae"] / baseline_mae) if baseline_mae > 0 else float("nan")
    print(
        f"  h={horizon:>2}: rows={len(valid):>4}  "
        f"MAE={median['mae']:.4f}  RMSE={median['rmse']:.4f}  "
        f"MAPE={median['mape']:.4f}  baseline_MAE={baseline_mae:.4f}  "
        f"skill={skill:+.2%}"
    )

    # MLflow per-horizon run
    with mlflow.start_run(run_name=f"horizon_{horizon}", nested=True):
        mlflow.log_param("horizon_weeks", horizon)
        mlflow.log_param("n_rows", len(valid))
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_metric("mae", median["mae"])
        mlflow.log_metric("rmse", median["rmse"])
        mlflow.log_metric("mape", median["mape"])
        mlflow.log_metric("baseline_mae", baseline_mae)
        mlflow.log_metric("skill_vs_naive", skill)
        for q_key, m in quantile_metrics.items():
            mlflow.log_metric(f"{q_key}_mae", m["mae"])
        mlflow.log_artifact(str(out_dir / f"lgbm_h{horizon}.txt"))

    return {
        "horizon": horizon,
        "n_rows": len(valid),
        "quantiles": quantile_metrics,
        "baseline_mae": baseline_mae,
        "skill_vs_naive": skill,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ingredient", default="CORN", help="Ingredient folder name under data/materialized/")
    parser.add_argument("--horizons", type=int, default=13, help="Max horizon in weeks (1..N)")
    args = parser.parse_args()

    ingredient = args.ingredient.upper()
    horizons = list(range(1, args.horizons + 1))

    out_dir = MODELS_DIR / ingredient.lower()
    out_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(f"file:{MLFLOW_DIR.as_posix()}")
    mlflow.set_experiment(f"price_forecast_{ingredient.lower()}")

    print(f"=== Training {ingredient} · horizons 1..{horizons[-1]} weeks ===")
    df = load_training_frame(ingredient)
    df = build_horizon_targets(df, horizons)
    feature_cols = select_feature_columns(df, horizons)
    print(f"  training rows: {len(df)}   features: {len(feature_cols)}")

    with mlflow.start_run(run_name=f"{ingredient}_multihorizon"):
        mlflow.log_param("ingredient", ingredient)
        mlflow.log_param("max_horizon_weeks", horizons[-1])
        mlflow.log_param("quantiles", list(QUANTILES))
        mlflow.log_param("target", TARGET)
        mlflow.log_dict({"features": feature_cols}, "features.json")

        horizon_results = [
            train_horizon(df, h, feature_cols, out_dir) for h in horizons
        ]

        successful = [r for r in horizon_results if r.get("quantiles")]
        if successful:
            mlflow.log_metric(
                "mean_median_mae",
                float(np.mean([r["quantiles"]["q50"]["mae"] for r in successful])),
            )
            mlflow.log_metric(
                "mean_skill_vs_naive",
                float(np.mean([r["skill_vs_naive"] for r in successful])),
            )

    metadata = {
        "ingredient": ingredient,
        "target": TARGET,
        "features": feature_cols,
        "horizons_weeks": horizons,
        "quantiles": list(QUANTILES),
        "train_end_date": df[DATE_COL].max().isoformat(),
        "n_training_rows": int(len(df)),
        "per_horizon": horizon_results,
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, default=str)

    print(f"\nArtifacts written to {out_dir}")
    print(f"MLflow store: {MLFLOW_DIR}  (run `mlflow ui` to browse)")


if __name__ == "__main__":
    main()
