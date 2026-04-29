"""
Margin and Cost Modeling — Total landed cost stack from commodity to processing plant.

Reconstructs the full cost stack for a commodity:
    Total Landed Cost = Base Commodity Price
                      + Transportation Cost  (diesel-driven)
                      + Energy Cost          (natural gas-driven)
                      + Cost of Carry        (fed funds rate × inventory days)

Supports scenario analysis by shocking individual cost drivers.

Inputs
------
- data/materialized/<INGREDIENT>/training_weekly.parquet
- data/forecasts/<INGREDIENT>/weekly_forecast.parquet

Outputs
-------
- data/forecasts/<INGREDIENT>/margin_model.parquet
- data/forecasts/<INGREDIENT>/margin_model.csv
- data/forecasts/<INGREDIENT>/scenario_analysis.json

Usage
-----
    python scripts/9.margin_model.py
    python scripts/9.margin_model.py --ingredient CORN --diesel-shock 0.20
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

REPO_ROOT = Path(__file__).resolve().parents[1]
MATERIALIZED = REPO_ROOT / "data" / "materialized"
FORECASTS_DIR = REPO_ROOT / "data" / "forecasts"

DATE_COL = "week_start"

# ---------------------------------------------------------------------------
# Cost model parameters (calibrated from FRED averages, adjustable via CLI)
# ---------------------------------------------------------------------------
DEFAULT_PARAMS = {
    # Transportation: cents per bushel per 100 miles, calibrated to diesel
    "miles_to_terminal": 200,           # average haul distance (miles)
    "bushels_per_load": 1000,           # bushels per truckload
    "truck_mpg": 6.5,                   # fuel efficiency
    "diesel_base_usd_per_gallon": 3.80, # FRED baseline (DDFUELUSGULF)
    # Energy: natural gas cost per bushel processed
    "nat_gas_base_usd_per_mmbtu": 3.50, # FRED Henry Hub baseline
    "nat_gas_mmbtu_per_bushel": 0.002,  # processing energy intensity
    # Inventory / carry
    "inventory_days": 30,               # average days held
    "fed_funds_rate": 0.0525,           # FRED FEDFUNDS (annualised)
    # Selling price premium (processor margin target)
    "target_gross_margin_pct": 0.12,    # 12% gross margin target
}


# ---------------------------------------------------------------------------
# Cost component calculators
# ---------------------------------------------------------------------------

def transportation_cost_per_unit(
    diesel_usd_gal: float,
    miles: float,
    mpg: float,
    bushels_per_load: float,
) -> float:
    """USD per bushel transportation cost."""
    gallons = miles / mpg
    total_fuel = gallons * diesel_usd_gal
    # Add fixed per-mile cost (driver, equipment, insurance): ~$2.50/mile
    total_fixed = miles * 2.50
    return (total_fuel + total_fixed) / bushels_per_load


def energy_cost_per_unit(nat_gas_usd_mmbtu: float, mmbtu_per_bushel: float) -> float:
    """USD per bushel energy cost at processing plant."""
    return nat_gas_usd_mmbtu * mmbtu_per_bushel


def carry_cost_per_unit(
    commodity_price: float,
    fed_funds_rate: float,
    inventory_days: int,
) -> float:
    """USD per bushel financing cost of holding inventory."""
    daily_rate = fed_funds_rate / 365
    return commodity_price * daily_rate * inventory_days


# ---------------------------------------------------------------------------
# Build cost stack
# ---------------------------------------------------------------------------

def build_cost_stack(
    df: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """
    Attach cost components to each row in df (which must have price_avg_weekly).
    Returns df with added columns: transport_cost, energy_cost, carry_cost,
    total_landed_cost, gross_margin_usd, gross_margin_pct.
    """
    out = df.copy()

    # Use macro_ppi as a diesel proxy if FRED diesel not directly available
    diesel_proxy = params["diesel_base_usd_per_gallon"]
    if "di_ppi" in out.columns:
        # Normalise: di_ppi is dollar-index level; invert to proxy import cost pressure
        # When dollar is strong (high DI), domestic diesel tends to be relatively cheaper
        di_mean = out["di_ppi"].mean()
        diesel_proxy_series = params["diesel_base_usd_per_gallon"] * (di_mean / out["di_ppi"].clip(50))
    else:
        diesel_proxy_series = pd.Series(diesel_proxy, index=out.index)

    nat_gas_proxy = params["nat_gas_base_usd_per_mmbtu"]
    if "macro_ppi" in out.columns:
        ppi_mean = out["macro_ppi"].mean()
        nat_gas_proxy_series = params["nat_gas_base_usd_per_mmbtu"] * (out["macro_ppi"] / ppi_mean.clip(1))
    else:
        nat_gas_proxy_series = pd.Series(nat_gas_proxy, index=out.index)

    out["transport_cost"] = diesel_proxy_series.apply(
        lambda d: transportation_cost_per_unit(
            d,
            params["miles_to_terminal"],
            params["truck_mpg"],
            params["bushels_per_load"],
        )
    )
    out["energy_cost"] = nat_gas_proxy_series.apply(
        lambda ng: energy_cost_per_unit(ng, params["nat_gas_mmbtu_per_bushel"])
    )
    out["carry_cost"] = out["price_avg_weekly"].apply(
        lambda p: carry_cost_per_unit(
            p, params["fed_funds_rate"], params["inventory_days"]
        )
    )
    out["total_landed_cost"] = (
        out["price_avg_weekly"] + out["transport_cost"] + out["energy_cost"] + out["carry_cost"]
    )
    # Implied selling price at target gross margin
    out["implied_sell_price"] = out["total_landed_cost"] / (1 - params["target_gross_margin_pct"])
    out["gross_margin_usd"] = out["implied_sell_price"] - out["total_landed_cost"]
    out["gross_margin_pct"] = out["gross_margin_usd"] / out["implied_sell_price"] * 100

    return out


def apply_forecast(
    hist_stack: pd.DataFrame,
    forecast_df: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """Extend the cost stack into the forecast horizon."""
    rows = []
    for _, fc_row in forecast_df.iterrows():
        for quantile_col, price_col in [("p10", "p10"), ("p50", "p50"), ("p90", "p90")]:
            price = fc_row[price_col]
            transport = transportation_cost_per_unit(
                params["diesel_base_usd_per_gallon"],
                params["miles_to_terminal"],
                params["truck_mpg"],
                params["bushels_per_load"],
            )
            energy = energy_cost_per_unit(
                params["nat_gas_base_usd_per_mmbtu"],
                params["nat_gas_mmbtu_per_bushel"],
            )
            carry = carry_cost_per_unit(price, params["fed_funds_rate"], params["inventory_days"])
            tlc = price + transport + energy + carry
            rows.append({
                "week_start": fc_row["target_week_start"],
                "horizon_weeks": fc_row["horizon_weeks"],
                "quantile": quantile_col,
                "commodity_price": price,
                "transport_cost": transport,
                "energy_cost": energy,
                "carry_cost": carry,
                "total_landed_cost": tlc,
                "implied_sell_price": tlc / (1 - params["target_gross_margin_pct"]),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Scenario analysis
# ---------------------------------------------------------------------------

def scenario_analysis(base_cost: float, params: dict) -> dict:
    """
    Shock each cost driver by ±10%, ±20%. Return impact on total landed cost.
    """
    shocks = {"diesel": [-0.20, -0.10, +0.10, +0.20], "nat_gas": [-0.20, -0.10, +0.10, +0.20],
               "fed_funds": [-0.50, -0.25, +0.25, +0.50], "miles": [-0.20, +0.20]}
    base_price = 5.0  # representative commodity price
    base_tlc = (
        base_price
        + transportation_cost_per_unit(
            params["diesel_base_usd_per_gallon"],
            params["miles_to_terminal"],
            params["truck_mpg"],
            params["bushels_per_load"],
        )
        + energy_cost_per_unit(params["nat_gas_base_usd_per_mmbtu"], params["nat_gas_mmbtu_per_bushel"])
        + carry_cost_per_unit(base_price, params["fed_funds_rate"], params["inventory_days"])
    )

    results: dict[str, list] = {}
    for driver, shock_levels in shocks.items():
        results[driver] = []
        for shock in shock_levels:
            p = dict(params)
            if driver == "diesel":
                p["diesel_base_usd_per_gallon"] *= (1 + shock)
            elif driver == "nat_gas":
                p["nat_gas_base_usd_per_mmbtu"] *= (1 + shock)
            elif driver == "fed_funds":
                p["fed_funds_rate"] = max(0, p["fed_funds_rate"] + shock * 0.01)
            elif driver == "miles":
                p["miles_to_terminal"] *= (1 + shock)

            shocked_tlc = (
                base_price
                + transportation_cost_per_unit(
                    p["diesel_base_usd_per_gallon"],
                    p["miles_to_terminal"],
                    p["truck_mpg"],
                    p["bushels_per_load"],
                )
                + energy_cost_per_unit(p["nat_gas_base_usd_per_mmbtu"], p["nat_gas_mmbtu_per_bushel"])
                + carry_cost_per_unit(base_price, p["fed_funds_rate"], p["inventory_days"])
            )
            results[driver].append({
                "shock_pct": int(shock * 100),
                "delta_tlc": round(shocked_tlc - base_tlc, 4),
                "delta_pct": round((shocked_tlc / base_tlc - 1) * 100, 2),
            })
    return {"base_tlc": round(base_tlc, 4), "scenarios": results}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ingredient", default="CORN")
    parser.add_argument("--diesel-shock", type=float, default=0.0,
                        help="Fractional shock to diesel price (e.g. 0.20 = +20%%)")
    parser.add_argument("--nat-gas-shock", type=float, default=0.0)
    parser.add_argument("--fed-funds-rate", type=float, default=None,
                        help="Override fed funds rate (e.g. 0.055 for 5.5%%)")
    args = parser.parse_args()

    ingredient = args.ingredient.upper()
    out_dir = FORECASTS_DIR / ingredient
    out_dir.mkdir(parents=True, exist_ok=True)

    params = dict(DEFAULT_PARAMS)
    if args.diesel_shock:
        params["diesel_base_usd_per_gallon"] *= (1 + args.diesel_shock)
    if args.nat_gas_shock:
        params["nat_gas_base_usd_per_mmbtu"] *= (1 + args.nat_gas_shock)
    if args.fed_funds_rate is not None:
        params["fed_funds_rate"] = args.fed_funds_rate

    print(f"=== Margin Model · {ingredient} ===")

    # Load historical price data
    hist_path = MATERIALIZED / ingredient / "training_weekly.parquet"
    if not hist_path.exists():
        raise FileNotFoundError(f"{hist_path} not found.")
    hist_df = pd.read_parquet(hist_path)
    hist_df[DATE_COL] = pd.to_datetime(hist_df[DATE_COL])
    hist_df = hist_df.sort_values(DATE_COL).reset_index(drop=True)

    # Build historical cost stack
    hist_stack = build_cost_stack(hist_df, params)
    recent = hist_stack.tail(52)

    print("  Historical cost stack (last 52 weeks):")
    print(f"    Avg commodity price:    ${recent['price_avg_weekly'].mean():.4f}/bu")
    print(f"    Avg transport cost:     ${recent['transport_cost'].mean():.4f}/bu")
    print(f"    Avg energy cost:        ${recent['energy_cost'].mean():.4f}/bu")
    print(f"    Avg carry cost:         ${recent['carry_cost'].mean():.4f}/bu")
    print(f"    Avg total landed cost:  ${recent['total_landed_cost'].mean():.4f}/bu")
    print(f"    Avg implied sell price: ${recent['implied_sell_price'].mean():.4f}/bu")

    # Extend into forecast horizon if available
    forecast_path = out_dir / "weekly_forecast.parquet"
    forecast_stack = None
    if forecast_path.exists():
        fc_df = pd.read_parquet(forecast_path)
        fc_df["target_week_start"] = pd.to_datetime(fc_df["target_week_start"])
        forecast_stack = apply_forecast(hist_stack, fc_df, params)
        print("\n  Forecast horizon cost projections (p50):")
        p50_fc = forecast_stack[forecast_stack["quantile"] == "p50"]
        for _, row in p50_fc.head(4).iterrows():
            print(
                f"    h={int(row['horizon_weeks']):>2}  "
                f"commodity=${row['commodity_price']:.4f}  "
                f"TLC=${row['total_landed_cost']:.4f}"
            )

    # Scenario analysis
    scenarios = scenario_analysis(recent["total_landed_cost"].mean(), params)
    with open(out_dir / "scenario_analysis.json", "w") as fh:
        json.dump(scenarios, fh, indent=2)

    # Save outputs
    cols = [
        DATE_COL, "price_avg_weekly", "transport_cost", "energy_cost",
        "carry_cost", "total_landed_cost", "implied_sell_price",
        "gross_margin_usd", "gross_margin_pct",
    ]
    available_cols = [c for c in cols if c in hist_stack.columns]
    hist_stack[available_cols].to_parquet(out_dir / "margin_model.parquet", index=False)
    hist_stack[available_cols].to_csv(out_dir / "margin_model.csv", index=False)

    if forecast_stack is not None:
        forecast_stack.to_parquet(out_dir / "margin_forecast.parquet", index=False)

    print(f"\nArtifacts → {out_dir}")
    print(f"Scenario analysis → {out_dir}/scenario_analysis.json")


if __name__ == "__main__":
    main()
