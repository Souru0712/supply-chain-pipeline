"""
Supply Disruption Detection — Composite early-warning score (0–100).

Detects emerging supply disruptions by combining four signal components:

  1. Supply Deviation  — how far supply-side metrics are from trend
  2. Price Gap         — supply signals vs. market price response lag
  3. Price Momentum    — rate of acceleration in wholesale prices
  4. Volatility Spike  — rolling volatility vs. long-run baseline

A score ≥ 60 → informational alert
A score ≥ 75 → procurement review recommended
A score ≥ 90 → emergency sourcing action

Inputs
------
- data/materialized/<INGREDIENT>/training_weekly.parquet

Outputs
-------
- data/forecasts/<INGREDIENT>/disruption_score.json
- data/forecasts/<INGREDIENT>/disruption_history.parquet

Usage
-----
    python scripts/10.disruption_score.py
    python scripts/10.disruption_score.py --ingredient CORN --as-of 2025-06-01
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

REPO_ROOT = Path(__file__).resolve().parents[1]
MATERIALIZED = REPO_ROOT / "data" / "materialized"
FORECASTS_DIR = REPO_ROOT / "data" / "forecasts"

DATE_COL = "week_start"
PRICE_COL = "price_avg_weekly"

ALERT_THRESHOLDS = {
    "informational": 60,
    "procurement_review": 75,
    "emergency_action": 90,
}

COMPONENT_WEIGHTS = {
    "supply_deviation": 0.30,
    "price_gap": 0.30,
    "price_momentum": 0.20,
    "volatility_spike": 0.20,
}


# ---------------------------------------------------------------------------
# Individual signal scorers (each returns 0–100)
# ---------------------------------------------------------------------------

def score_supply_deviation(series: pd.Series, window: int = 52) -> pd.Series:
    """
    How much has recent supply-proxy (macro PPI as cost-push proxy) deviated
    from its long-run trend? High cost-push pressure = supply tightness.
    Score increases when current value is above historical percentile.
    """
    rolling_median = series.rolling(window, min_periods=max(4, window // 4)).median()
    rolling_std = series.rolling(window, min_periods=max(4, window // 4)).std().clip(lower=1e-6)
    z_score = (series - rolling_median) / rolling_std
    # Map z-score to 0–100: z=0 → 50, z=+2 → 90, z=-2 → 10
    score = 50 + z_score.clip(-3, 3) * 15
    return score.clip(0, 100)


def score_price_gap(price: pd.Series, supply_proxy: pd.Series, window: int = 12) -> pd.Series:
    """
    Supply stress is accelerating but market price hasn't reacted yet.
    Gap between supply-stress direction and price direction = detection window.
    High score = supply worsening while price is flat/falling (lagged market).
    """
    supply_change = supply_proxy.pct_change(window).fillna(0)
    price_change = price.pct_change(window).fillna(0)
    # Divergence: supply rising faster than price = unpriced risk
    divergence = supply_change - price_change
    rolling_std = divergence.rolling(52, min_periods=4).std().clip(lower=1e-6)
    rolling_mean = divergence.rolling(52, min_periods=4).mean()
    z = (divergence - rolling_mean) / rolling_std
    score = 50 + z.clip(-3, 3) * 15
    return score.clip(0, 100)


def score_price_momentum(price: pd.Series, short: int = 4, long: int = 26) -> pd.Series:
    """
    Rate of price acceleration relative to longer-term trend.
    Rapid upward momentum (price rising much faster than long-run average) = high score.
    """
    momentum = price.pct_change(short) - price.pct_change(long)
    rolling_std = momentum.rolling(52, min_periods=4).std().clip(lower=1e-6)
    rolling_mean = momentum.rolling(52, min_periods=4).mean()
    z = (momentum - rolling_mean) / rolling_std
    score = 50 + z.clip(-3, 3) * 15
    return score.clip(0, 100)


def score_volatility_spike(price: pd.Series, short: int = 12, long: int = 52) -> pd.Series:
    """
    Current short-term volatility vs. long-run volatility baseline.
    A spike in volatility signals market uncertainty = supply disruption.
    """
    short_vol = price.pct_change().rolling(short, min_periods=4).std()
    long_vol = price.pct_change().rolling(long, min_periods=12).std().clip(lower=1e-6)
    vol_ratio = (short_vol / long_vol).clip(0, 5)
    # Map: ratio=1 (normal) → 50; ratio=3 → ~90
    score = (vol_ratio / 3 * 100).clip(0, 100)
    return score


def composite_disruption_score(
    df: pd.DataFrame,
    weights: dict[str, float] = COMPONENT_WEIGHTS,
) -> pd.DataFrame:
    """Compute all components and weighted composite. Returns df with score columns."""
    out = df.copy()
    price = out[PRICE_COL]
    supply_proxy = out["macro_ppi"] if "macro_ppi" in out.columns else price

    out["score_supply_deviation"] = score_supply_deviation(supply_proxy)
    out["score_price_gap"] = score_price_gap(price, supply_proxy)
    out["score_price_momentum"] = score_price_momentum(price)
    out["score_volatility_spike"] = score_volatility_spike(price)

    out["disruption_score"] = (
        out["score_supply_deviation"] * weights["supply_deviation"]
        + out["score_price_gap"] * weights["price_gap"]
        + out["score_price_momentum"] * weights["price_momentum"]
        + out["score_volatility_spike"] * weights["volatility_spike"]
    ).clip(0, 100)

    return out


def determine_alert_level(score: float) -> dict:
    if score >= ALERT_THRESHOLDS["emergency_action"]:
        return {"level": "EMERGENCY", "color": "#d32f2f", "action": "Emergency sourcing action required"}
    elif score >= ALERT_THRESHOLDS["procurement_review"]:
        return {"level": "HIGH", "color": "#f57c00", "action": "Procurement review recommended"}
    elif score >= ALERT_THRESHOLDS["informational"]:
        return {"level": "MODERATE", "color": "#fbc02d", "action": "Monitor closely — informational alert"}
    else:
        return {"level": "LOW", "color": "#388e3c", "action": "No action required"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ingredient", default="CORN")
    parser.add_argument("--as-of", default=None, dest="as_of",
                        help="Cutoff date YYYY-MM-DD (default: latest)")
    args = parser.parse_args()

    ingredient = args.ingredient.upper()
    out_dir = FORECASTS_DIR / ingredient
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = MATERIALIZED / ingredient / "training_weekly.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} not found.")

    df = pd.read_parquet(data_path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    if args.as_of:
        df = df[df[DATE_COL] <= pd.Timestamp(args.as_of)]

    if df.empty:
        raise ValueError("No data available for the specified date range.")

    print(f"=== Disruption Score · {ingredient} ===")
    scored = composite_disruption_score(df)

    current = scored.iloc[-1]
    current_score = float(current["disruption_score"])
    alert = determine_alert_level(current_score)

    print(f"  As of:              {current[DATE_COL].date()}")
    print(f"  Composite score:    {current_score:.1f} / 100")
    print(f"  Alert level:        {alert['level']}")
    print(f"  Action:             {alert['action']}")
    print()
    print(f"  Component scores:")
    for component in ["score_supply_deviation", "score_price_gap",
                      "score_price_momentum", "score_volatility_spike"]:
        print(f"    {component:<30} {float(current[component]):.1f}")

    # Save full history
    score_cols = [DATE_COL, PRICE_COL, "disruption_score",
                  "score_supply_deviation", "score_price_gap",
                  "score_price_momentum", "score_volatility_spike"]
    available = [c for c in score_cols if c in scored.columns]
    scored[available].to_parquet(out_dir / "disruption_history.parquet", index=False)

    # Save current snapshot
    snapshot = {
        "ingredient": ingredient,
        "as_of": str(current[DATE_COL].date()),
        "composite_score": round(current_score, 2),
        "alert": alert,
        "components": {
            "supply_deviation": round(float(current["score_supply_deviation"]), 2),
            "price_gap": round(float(current["score_price_gap"]), 2),
            "price_momentum": round(float(current["score_price_momentum"]), 2),
            "volatility_spike": round(float(current["score_volatility_spike"]), 2),
        },
        "weights": COMPONENT_WEIGHTS,
        "thresholds": ALERT_THRESHOLDS,
        "12w_trend": [
            round(float(v), 2) for v in scored["disruption_score"].tail(12).tolist()
        ],
    }
    with open(out_dir / "disruption_score.json", "w") as fh:
        json.dump(snapshot, fh, indent=2)

    print(f"\nArtifacts → {out_dir}")


if __name__ == "__main__":
    main()
