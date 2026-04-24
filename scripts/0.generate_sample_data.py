"""
Generate synthetic weekly corn price training data.

Produces data/materialized/CORN/training_weekly.parquet with the exact schema
expected by 6.train_forecast.py, enabling development and testing without API keys.

Corn prices are simulated with realistic trend, seasonality, and macro features.

Usage
-----
    python scripts/0.generate_sample_data.py
    python scripts/0.generate_sample_data.py --ingredient CORN --weeks 310
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
MATERIALIZED = REPO_ROOT / "data" / "materialized"


def generate_corn_prices(n_weeks: int = 310, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n_weeks)
    dates = pd.date_range("2020-01-06", periods=n_weeks, freq="W-MON")

    # Corn price: base ~$4.50/bushel with seasonal pattern, trend, and random walk
    seasonal = 0.40 * np.sin(2 * np.pi * t / 52) + 0.15 * np.sin(4 * np.pi * t / 52)
    trend = 0.003 * t
    shocks = rng.normal(0, 0.06, n_weeks)
    random_walk = np.cumsum(shocks * 0.15)
    prices = np.clip(4.5 + trend + seasonal + random_walk, 3.0, 9.0)

    spread = rng.uniform(0.05, 0.30, n_weeks)
    n_reports = rng.integers(3, 15, n_weeks).astype(float)

    df = pd.DataFrame(
        {
            "week_start": dates,
            "price_avg_weekly": prices,
            "price_min_weekly": prices - spread / 2,
            "price_max_weekly": prices + spread / 2,
            "n_reports_in_week": n_reports,
        }
    )

    # Backward lag features (no leakage: shift pushes values into the future)
    for lag in [1, 2, 4, 8, 13, 26, 52]:
        df[f"price_lag_{lag}w"] = df["price_avg_weekly"].shift(lag)

    # Historical trailing rolling averages (strictly backward: shift(1) before rolling)
    price_s = df["price_avg_weekly"]
    for w in [4, 12, 26, 52]:
        df[f"price_hist_avg_{w}w"] = price_s.shift(1).rolling(w, min_periods=2).mean()

    df["price_hist_std_12w"] = price_s.shift(1).rolling(12, min_periods=2).std()

    # Trade-weighted dollar index (~95–105 range with slow drift)
    di = 100 + 5 * np.sin(2 * np.pi * t / 104) + np.cumsum(rng.normal(0, 0.08, n_weeks))
    df["di_ppi"] = np.clip(di, 85.0, 120.0)

    # Macro PPI — monthly cadence stamped weekly, gaps forward-filled by caller
    macro_base = (
        220 + 0.18 * t + 5 * np.sin(2 * np.pi * t / 52) + rng.normal(0, 1.5, n_weeks)
    )
    macro_ppi = np.where(t % 4 == 0, np.clip(macro_base, 180.0, 290.0), np.nan)
    df["macro_ppi"] = pd.Series(macro_ppi).ffill()

    # Calendar features — use .values to avoid UInt32 → float NaN issue
    df["year"] = dates.year.astype(float)
    df["week_of_year"] = dates.isocalendar().week.values.astype(float)
    df["month"] = dates.month.astype(float)
    df["quarter"] = dates.quarter.astype(float)

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ingredient", default="CORN")
    parser.add_argument("--weeks", type=int, default=310)
    args = parser.parse_args()

    ingredient = args.ingredient.upper()
    out_dir = MATERIALIZED / ingredient
    out_dir.mkdir(parents=True, exist_ok=True)

    df = generate_corn_prices(n_weeks=args.weeks)
    out_path = out_dir / "training_weekly.parquet"
    df.to_parquet(out_path, index=False)

    print(f"Generated {len(df)} rows of synthetic {ingredient} data → {out_path}")
    print(
        f"  Price: ${df['price_avg_weekly'].min():.2f} – ${df['price_avg_weekly'].max():.2f}/bushel"
    )
    print(f"  Dates: {df['week_start'].min().date()} – {df['week_start'].max().date()}")


if __name__ == "__main__":
    main()
