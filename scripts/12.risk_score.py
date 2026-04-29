"""
Composite Risk Quantification — Four-dimension supply chain risk score (0–100).

Dimensions
----------
  1. Supply Risk    — is supply contracting? (price trend vs. historical norms)
  2. Cost Risk      — are input costs rising? (PPI / macro cost-push pressure)
  3. Logistics Risk — are transportation costs elevated? (dollar index proxy)
  4. Demand Risk    — is demand destroying price? (price vs. volatility dynamics)

Each dimension is scored via historical percentile ranking (0 = best ever,
100 = worst ever). A composite score is the weighted sum.

Default weights can be overridden via --weights to reflect a specific
organisation's exposure profile (e.g., a well-hedged buyer may reduce cost risk).

Inputs
------
- data/materialized/<INGREDIENT>/training_weekly.parquet

Outputs
-------
- data/forecasts/<INGREDIENT>/risk_score.json
- data/forecasts/<INGREDIENT>/risk_history.parquet

Usage
-----
    python scripts/12.risk_score.py
    python scripts/12.risk_score.py --ingredient CORN --as-of 2025-01-01
    python scripts/12.risk_score.py --weights 0.40 0.30 0.20 0.10
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import pandas as pd
from scipy.stats import percentileofscore

warnings.filterwarnings("ignore", category=UserWarning)

REPO_ROOT = Path(__file__).resolve().parents[1]
MATERIALIZED = REPO_ROOT / "data" / "materialized"
FORECASTS_DIR = REPO_ROOT / "data" / "forecasts"

DATE_COL = "week_start"
PRICE_COL = "price_avg_weekly"

DEFAULT_WEIGHTS = {
    "supply_risk": 0.30,
    "cost_risk": 0.30,
    "logistics_risk": 0.20,
    "demand_risk": 0.20,
}

RISK_LEVELS = [
    (80, "CRITICAL",  "#b71c1c", "Immediate escalation required"),
    (60, "HIGH",      "#d32f2f", "Senior procurement review"),
    (40, "MODERATE",  "#f57c00", "Active monitoring"),
    (20, "LOW",       "#fbc02d", "Routine monitoring"),
    (0,  "MINIMAL",   "#388e3c", "No action required"),
]


# ---------------------------------------------------------------------------
# Dimension scorers
# ---------------------------------------------------------------------------

def percentile_score(series: pd.Series, value: float) -> float:
    """Percentile of value within the historical series (0–100)."""
    clean = series.dropna().to_numpy()
    if len(clean) < 4:
        return 50.0
    return float(percentileofscore(clean, value, kind="rank"))


def score_supply_risk(df: pd.DataFrame, lookback: int = 52) -> pd.Series:
    """
    Supply risk = price trend deviation above long-run level.
    When price is trending higher than its long-run norm,
    supply tightness is likely the cause.
    """
    price = df[PRICE_COL]
    long_avg = price.rolling(lookback, min_periods=12).mean()
    short_avg = price.rolling(lookback // 4, min_periods=4).mean()
    # Ratio of short-term to long-term price: >1 means supply tightening
    ratio = (short_avg / long_avg.clip(lower=1e-6)).clip(0.5, 2.0)
    scores = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        hist = ratio.iloc[: i + 1].dropna()
        if len(hist) < 4:
            scores.iloc[i] = 50.0
        else:
            scores.iloc[i] = percentile_score(hist, ratio.iloc[i])
    return scores


def score_cost_risk(df: pd.DataFrame, lookback: int = 52) -> pd.Series:
    """
    Cost risk = macro PPI cost-push pressure above long-run trend.
    Rising macro PPI = input costs increasing, margin pressure ahead.
    """
    col = "macro_ppi" if "macro_ppi" in df.columns else PRICE_COL
    series = df[col]
    yoy_change = series.pct_change(52).fillna(0)
    scores = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        hist = yoy_change.iloc[: i + 1].dropna()
        if len(hist) < 4:
            scores.iloc[i] = 50.0
        else:
            scores.iloc[i] = percentile_score(hist, yoy_change.iloc[i])
    return scores


def score_logistics_risk(df: pd.DataFrame, lookback: int = 52) -> pd.Series:
    """
    Logistics risk = dollar index weakness (strong dollar = cheap imports,
    weak dollar = expensive imports / transportation inputs rise).
    Use inverted percentile: low DI = high logistics cost risk.
    """
    col = "di_ppi" if "di_ppi" in df.columns else PRICE_COL
    series = df[col]
    # Invert: low dollar index = high risk
    inverted = -series
    scores = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        hist = inverted.iloc[: i + 1].dropna()
        if len(hist) < 4:
            scores.iloc[i] = 50.0
        else:
            scores.iloc[i] = percentile_score(hist, inverted.iloc[i])
    return scores


def score_demand_risk(df: pd.DataFrame, lookback: int = 26) -> pd.Series:
    """
    Demand risk = price falling despite low volatility = demand destruction.
    OR price falling rapidly = buyers stepping back.
    """
    price = df[PRICE_COL]
    momentum = price.pct_change(lookback).fillna(0)
    vol = price.pct_change().rolling(lookback, min_periods=4).std().fillna(0)
    # Demand destruction signal: negative momentum + low volatility
    # (if volatility were high, it's a supply shock, not demand)
    signal = -momentum * (1 - vol.clip(0, 0.1) / 0.1)
    scores = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        hist = signal.iloc[: i + 1].dropna()
        if len(hist) < 4:
            scores.iloc[i] = 50.0
        else:
            scores.iloc[i] = percentile_score(hist, signal.iloc[i])
    return scores


# ---------------------------------------------------------------------------
# Composite scorer
# ---------------------------------------------------------------------------

def compute_risk_scores(df: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    print("  Computing supply risk...", end=" ", flush=True)
    out["supply_risk"] = score_supply_risk(df)
    print("done")
    print("  Computing cost risk...", end=" ", flush=True)
    out["cost_risk"] = score_cost_risk(df)
    print("done")
    print("  Computing logistics risk...", end=" ", flush=True)
    out["logistics_risk"] = score_logistics_risk(df)
    print("done")
    print("  Computing demand risk...", end=" ", flush=True)
    out["demand_risk"] = score_demand_risk(df)
    print("done")

    out["composite_risk"] = (
        out["supply_risk"] * weights["supply_risk"]
        + out["cost_risk"] * weights["cost_risk"]
        + out["logistics_risk"] * weights["logistics_risk"]
        + out["demand_risk"] * weights["demand_risk"]
    ).clip(0, 100)

    return out


def classify_risk(score: float) -> dict:
    for threshold, level, color, action in RISK_LEVELS:
        if score >= threshold:
            return {"level": level, "color": color, "action": action, "score": round(score, 1)}
    return {"level": "MINIMAL", "color": "#388e3c", "action": "No action required", "score": round(score, 1)}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ingredient", default="CORN")
    parser.add_argument("--as-of", default=None, dest="as_of")
    parser.add_argument(
        "--weights", nargs=4, type=float, default=None,
        metavar=("SUPPLY", "COST", "LOGISTICS", "DEMAND"),
        help="Override dimension weights (must sum to 1.0)",
    )
    args = parser.parse_args()

    ingredient = args.ingredient.upper()
    out_dir = FORECASTS_DIR / ingredient
    out_dir.mkdir(parents=True, exist_ok=True)

    weights = dict(DEFAULT_WEIGHTS)
    if args.weights:
        total = sum(args.weights)
        keys = list(DEFAULT_WEIGHTS.keys())
        weights = {k: v / total for k, v in zip(keys, args.weights)}

    data_path = MATERIALIZED / ingredient / "training_weekly.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} not found.")

    df = pd.read_parquet(data_path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    if args.as_of:
        df = df[df[DATE_COL] <= pd.Timestamp(args.as_of)]

    print(f"=== Risk Score · {ingredient} ===")
    scored = compute_risk_scores(df, weights)

    current = scored.iloc[-1]
    composite = float(current["composite_risk"])
    classification = classify_risk(composite)

    print(f"\n  As of:               {current[DATE_COL].date()}")
    print(f"  Composite risk:      {composite:.1f} / 100  [{classification['level']}]")
    print(f"  Action:              {classification['action']}")
    print()
    print("  Dimension scores (weight → score):")
    for dim, w in weights.items():
        val = float(current[dim])
        print(f"    {dim:<18} {w:.0%} → {val:.1f}")

    # 12-week trend
    trend_12w = scored["composite_risk"].tail(12).tolist()

    # Save history
    risk_cols = [DATE_COL, PRICE_COL, "composite_risk",
                 "supply_risk", "cost_risk", "logistics_risk", "demand_risk"]
    available = [c for c in risk_cols if c in scored.columns]
    scored[available].to_parquet(out_dir / "risk_history.parquet", index=False)

    snapshot = {
        "ingredient": ingredient,
        "as_of": str(current[DATE_COL].date()),
        "composite_risk": round(composite, 2),
        "classification": classification,
        "dimensions": {
            "supply_risk": round(float(current["supply_risk"]), 2),
            "cost_risk": round(float(current["cost_risk"]), 2),
            "logistics_risk": round(float(current["logistics_risk"]), 2),
            "demand_risk": round(float(current["demand_risk"]), 2),
        },
        "weights": weights,
        "12w_trend": [round(v, 2) for v in trend_12w],
        # Radar chart data (for dashboard)
        "radar": [
            {"dimension": k, "score": round(float(current[k]), 2), "weight": w}
            for k, w in weights.items()
        ],
    }
    with open(out_dir / "risk_score.json", "w") as fh:
        json.dump(snapshot, fh, indent=2)

    print(f"\nArtifacts → {out_dir}")


if __name__ == "__main__":
    main()
