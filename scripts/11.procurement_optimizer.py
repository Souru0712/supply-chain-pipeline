"""
Procurement Timing Optimization — 52-week dynamic purchase calendar.

Builds a seasonal price curve and solves a linear programme to find the
cost-minimising weekly purchase schedule subject to:
  - Total annual volume requirement met
  - Minimum weekly coverage (safety stock)
  - Maximum single-week purchase (logistics constraint)
  - Maximum total forward commitment at any point (financial risk limit)

Also flags opportunistic buying windows when actual price drops below
the 25th-percentile seasonal band.

Inputs
------
- data/materialized/<INGREDIENT>/training_weekly.parquet

Outputs
-------
- data/forecasts/<INGREDIENT>/procurement_schedule.csv
- data/forecasts/<INGREDIENT>/procurement_schedule.parquet
- data/forecasts/<INGREDIENT>/seasonal_curve.json

Usage
-----
    python scripts/11.procurement_optimizer.py
    python scripts/11.procurement_optimizer.py --ingredient CORN --annual-volume 10000000
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from statsmodels.tsa.seasonal import seasonal_decompose

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
MATERIALIZED = REPO_ROOT / "data" / "materialized"
FORECASTS_DIR = REPO_ROOT / "data" / "forecasts"

DATE_COL = "week_start"
PRICE_COL = "price_avg_weekly"


# ---------------------------------------------------------------------------
# Seasonal curve
# ---------------------------------------------------------------------------

def build_seasonal_curve(df: pd.DataFrame) -> dict:
    """
    Decompose weekly price series and compute per-week-of-year statistics.
    Returns dict with p25, median, p75 per week-of-year (1–52).
    """
    price = df.set_index(DATE_COL)[PRICE_COL].copy()
    price = price.resample("W-MON").last().ffill()

    # Seasonal decomposition (additive, period=52)
    if len(price) >= 104:
        try:
            result = seasonal_decompose(price, model="additive", period=52, extrapolate_trend="freq")
            seasonal = result.seasonal
        except Exception:
            seasonal = pd.Series(0, index=price.index)
    else:
        seasonal = pd.Series(0, index=price.index)

    # Per week-of-year stats across all years
    df_seasonal = pd.DataFrame({"price": price, "seasonal": seasonal})
    df_seasonal["week_of_year"] = df_seasonal.index.isocalendar().week.astype(int)

    stats: dict[int, dict] = {}
    for week in range(1, 53):
        subset = df_seasonal[df_seasonal["week_of_year"] == week]["price"]
        if len(subset) == 0:
            continue
        stats[week] = {
            "p25": float(subset.quantile(0.25)),
            "median": float(subset.median()),
            "p75": float(subset.quantile(0.75)),
            "mean": float(subset.mean()),
            "seasonal_factor": float(
                df_seasonal[df_seasonal["week_of_year"] == week]["seasonal"].mean()
            ),
        }
    return stats


def build_forward_price_curve(
    seasonal_stats: dict,
    current_price: float,
    start_week: int,
    n_weeks: int = 52,
) -> pd.DataFrame:
    """
    Build a 52-week forward price curve anchored to current price,
    adjusted by seasonal factors.
    """
    rows = []
    median_annual = np.mean([v["median"] for v in seasonal_stats.values()])
    for i in range(n_weeks):
        week = ((start_week - 1 + i) % 52) + 1
        stats = seasonal_stats.get(week, {"median": current_price, "p25": current_price * 0.95,
                                           "p75": current_price * 1.05, "seasonal_factor": 0.0})
        # Anchor: scale so the median curve passes through current_price at week 0
        scale = current_price / median_annual if median_annual > 0 else 1.0
        rows.append({
            "week_index": i + 1,
            "week_of_year": week,
            "expected_price": stats["median"] * scale,
            "p25_price": stats["p25"] * scale,
            "p75_price": stats["p75"] * scale,
            "seasonal_factor": stats["seasonal_factor"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# LP optimiser
# ---------------------------------------------------------------------------

def optimise_procurement(
    price_curve: pd.DataFrame,
    annual_volume: float,
    constraints: dict,
) -> pd.DataFrame:
    """
    Linear programme: minimise sum(price_i * purchase_i)
    subject to:
      sum(purchase_i) = annual_volume
      purchase_i >= min_weekly_purchase  (safety stock coverage)
      purchase_i <= max_weekly_purchase  (logistics limit)
      sum(purchase_0..k) <= max_cumulative_at_k  (forward commitment limit)

    Returns price_curve with added column: recommended_purchase_units.
    """
    n = len(price_curve)
    prices = price_curve["expected_price"].to_numpy()

    min_weekly = constraints["min_weekly_purchase"]
    max_weekly = constraints["max_weekly_purchase"]
    max_fwd_commit_pct = constraints["max_forward_commitment_pct"]

    # Objective: minimise total cost
    c = prices / prices.mean()  # normalised for numerical stability

    # Inequality constraints: cumulative purchase ≤ max_fwd_commit at each week
    A_ub = []
    b_ub = []
    for k in range(n):
        row = [1.0 if j <= k else 0.0 for j in range(n)]
        A_ub.append(row)
        b_ub.append(annual_volume * min((k + 1) / n * (1 + max_fwd_commit_pct), 1.0))

    # Bounds per week
    bounds = [(min_weekly, max_weekly)] * n

    # Equality: total = annual_volume
    A_eq = [np.ones(n)]
    b_eq = [annual_volume]

    result = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    if result.success:
        purchases = result.x
    else:
        # Fallback: distribute evenly
        purchases = np.full(n, annual_volume / n)
        purchases = purchases.clip(min_weekly, max_weekly)
        purchases = purchases / purchases.sum() * annual_volume

    curve = price_curve.copy()
    curve["recommended_purchase"] = purchases
    curve["cumulative_purchase"] = purchases.cumsum()
    curve["total_cost"] = purchases * prices
    curve["cumulative_cost"] = curve["total_cost"].cumsum()

    # Flag opportunistic buy weeks: price < p25 AND not already max committed
    curve["opportunistic_flag"] = (
        (prices < price_curve["p25_price"].to_numpy()) &
        (curve["recommended_purchase"] < max_weekly * 0.9)
    )
    return curve


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ingredient", default="CORN")
    parser.add_argument("--annual-volume", type=float, default=1_000_000,
                        help="Annual volume requirement in bushels")
    parser.add_argument("--max-weekly-pct", type=float, default=0.10,
                        help="Max fraction of annual volume purchasable in one week")
    parser.add_argument("--min-weekly-pct", type=float, default=0.005,
                        help="Min fraction of annual volume (safety stock)")
    parser.add_argument("--max-fwd-commit", type=float, default=0.20,
                        help="Max fraction ahead-of-schedule commitment allowed")
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

    print(f"=== Procurement Optimiser · {ingredient} ===")

    # Build seasonal curve
    seasonal_stats = build_seasonal_curve(df)

    current_price = float(df[PRICE_COL].iloc[-1])
    start_week = int(df[DATE_COL].iloc[-1].isocalendar().week)

    price_curve = build_forward_price_curve(seasonal_stats, current_price, start_week)

    print(f"  Current price:         ${current_price:.4f}/bu")
    print(f"  52-week expected avg:  ${price_curve['expected_price'].mean():.4f}/bu")
    trough_week = price_curve.loc[price_curve["expected_price"].idxmin()]
    peak_week = price_curve.loc[price_curve["expected_price"].idxmax()]
    print(f"  Seasonal trough:       week {int(trough_week['week_index'])} "
          f"(woy={int(trough_week['week_of_year'])}) "
          f"${trough_week['expected_price']:.4f}/bu")
    print(f"  Seasonal peak:         week {int(peak_week['week_index'])} "
          f"(woy={int(peak_week['week_of_year'])}) "
          f"${peak_week['expected_price']:.4f}/bu")

    # Run LP
    annual_volume = args.annual_volume
    constraints = {
        "min_weekly_purchase": annual_volume * args.min_weekly_pct,
        "max_weekly_purchase": annual_volume * args.max_weekly_pct,
        "max_forward_commitment_pct": args.max_fwd_commit,
    }
    schedule = optimise_procurement(price_curve, annual_volume, constraints)

    total_cost = schedule["total_cost"].sum()
    naive_cost = annual_volume * price_curve["expected_price"].mean()
    saving_pct = (naive_cost - total_cost) / naive_cost * 100

    print(f"\n  Annual volume:         {annual_volume:,.0f} bu")
    print(f"  Optimised total cost:  ${total_cost:,.2f}")
    print(f"  Naive (flat) cost:     ${naive_cost:,.2f}")
    print(f"  Estimated saving:      {saving_pct:.2f}%")

    opp_weeks = schedule[schedule["opportunistic_flag"]]["week_index"].tolist()
    if opp_weeks:
        print(f"\n  Opportunistic buy windows: weeks {opp_weeks[:5]}")

    # Save outputs
    schedule.to_csv(out_dir / "procurement_schedule.csv", index=False)
    schedule.to_parquet(out_dir / "procurement_schedule.parquet", index=False)

    seasonal_export = {
        "ingredient": ingredient,
        "current_price": current_price,
        "seasonal_stats": {
            str(k): v for k, v in seasonal_stats.items()
        },
        "optimiser_summary": {
            "annual_volume": annual_volume,
            "total_cost": round(total_cost, 2),
            "naive_cost": round(naive_cost, 2),
            "saving_pct": round(saving_pct, 2),
            "opportunistic_buy_weeks": opp_weeks,
        },
    }
    with open(out_dir / "seasonal_curve.json", "w") as fh:
        json.dump(seasonal_export, fh, indent=2)

    print(f"\nArtifacts → {out_dir}")


if __name__ == "__main__":
    main()
