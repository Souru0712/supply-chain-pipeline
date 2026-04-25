"""
Inference pipeline: load trained XGBoost models and produce a weekly forecast table.

Takes the most recent observation from the training parquet, applies all 13 horizon
models (p10/p50/p90), and writes a 13-row forecast schedule.

Inputs
------
- models/<ingredient>/xgb_h{h}.json  (and _q10, _q90 variants)
- models/<ingredient>/metadata.json
- data/materialized/<ingredient>/training_weekly.parquet

Outputs
-------
- data/forecasts/<ingredient>/weekly_forecast.parquet
- data/forecasts/<ingredient>/weekly_forecast.csv

Usage
-----
    python scripts/7.inference.py
    python scripts/7.inference.py --ingredient CORN --as-of 2026-04-07
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings("ignore", category=UserWarning)

REPO_ROOT = Path(__file__).resolve().parents[1]
MATERIALIZED = REPO_ROOT / "data" / "materialized"
MODELS_DIR = REPO_ROOT / "models"
FORECASTS_DIR = REPO_ROOT / "data" / "forecasts"

DATE_COL = "week_start"
CROSS_FFILL_COLS = ["di_ppi", "macro_ppi"]


def load_metadata(ingredient: str) -> dict:
    path = MODELS_DIR / ingredient.lower() / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/6.train_forecast.py` first."
        )
    with open(path) as fh:
        return json.load(fh)


def load_models(ingredient: str, horizons: list[int]) -> dict[str, xgb.Booster]:
    """Load the p10/p50/p90 XGBoost boosters for every horizon."""
    model_dir = MODELS_DIR / ingredient.lower()
    models: dict[str, xgb.Booster] = {}
    for h in horizons:
        for suffix, key in [("", "q50"), ("_q10", "q10"), ("_q90", "q90")]:
            model_path = model_dir / f"xgb_h{h}{suffix}.json"
            if model_path.exists():
                booster = xgb.Booster()
                booster.load_model(str(model_path))
                models[f"h{h}_{key}"] = booster
    return models


def load_feature_row(
    ingredient: str,
    feature_cols: list[str],
    as_of: str | None,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Return the last feature row at or before `as_of` and the cutoff date."""
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

    if as_of:
        df = df[df[DATE_COL] <= pd.Timestamp(as_of)]

    if df.empty:
        raise ValueError(f"No data at or before {as_of}.")

    cutoff_date: pd.Timestamp = df[DATE_COL].iloc[-1]
    available = [c for c in feature_cols if c in df.columns]
    return df.tail(1)[available].reset_index(drop=True), cutoff_date


def run_inference(ingredient: str, as_of: str | None = None) -> pd.DataFrame:
    """Run inference and return the 13-row forecast DataFrame."""
    meta = load_metadata(ingredient)
    feature_cols: list[str] = meta["features"]
    horizons: list[int] = meta["horizons_weeks"]

    models = load_models(ingredient, horizons)
    feature_row, forecast_date = load_feature_row(ingredient, feature_cols, as_of)

    X = xgb.DMatrix(feature_row.to_numpy().astype(np.float32))

    rows = []
    for h in horizons:
        target_week = forecast_date + pd.Timedelta(weeks=h)
        p50_model = models.get(f"h{h}_q50")
        p10_model = models.get(f"h{h}_q10")
        p90_model = models.get(f"h{h}_q90")

        rows.append(
            {
                "forecast_date": forecast_date,
                "horizon_weeks": h,
                "target_week_start": target_week,
                "p10": float(p10_model.predict(X)[0]) if p10_model else np.nan,
                "p50": float(p50_model.predict(X)[0]) if p50_model else np.nan,
                "p90": float(p90_model.predict(X)[0]) if p90_model else np.nan,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ingredient", default="CORN")
    parser.add_argument(
        "--as-of", default=None, dest="as_of", help="Cutoff date YYYY-MM-DD"
    )
    args = parser.parse_args()

    ingredient = args.ingredient.upper()
    print(f"=== Inference · {ingredient} ===")

    forecast_df = run_inference(ingredient, args.as_of)

    out_dir = FORECASTS_DIR / ingredient
    out_dir.mkdir(parents=True, exist_ok=True)
    forecast_df.to_parquet(out_dir / "weekly_forecast.parquet", index=False)
    forecast_df.to_csv(out_dir / "weekly_forecast.csv", index=False)

    print(f"Forecast as of {forecast_df['forecast_date'].iloc[0].date()}\n")
    print(
        forecast_df[
            ["horizon_weeks", "target_week_start", "p10", "p50", "p90"]
        ].to_string(index=False, float_format="{:.4f}".format)
    )
    print(f"\nArtifacts → {out_dir}")


if __name__ == "__main__":
    main()
