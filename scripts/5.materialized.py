"""
help me create a SQL function that inputs a name to a query:

SELECT *
FROM fred_mapping file
WHERE ingredient_name like '%name%'
LIMIT 1000;

The function should let me run this query with any name I want
------------------------------------------------------------------------------------------------------------------------------
now, lets add some some functions that use SQL query JOINS. The files are all on staged/validated

the script should use search_ingredient to return the specific commodity table from fred_mapping and then join them to the following tables:

- dollar_index (standalone — no cross join)
- fred_mapping CROSS JOIN macroeconomic ON ingredient_id
- fred_mapping JOIN market_and_logistic ON ingredient_name
- fred_mapping JOIN nass ON ingredient_name and unit_of_measure

for each of the joins, add columns that forward rolling averages on ppi on dollar_index, averages on ppi on macroeconomic, averages price_avg on market_and_logistics, and averages on amount on nass for these periods:

- 7 days ahead
- 14 days ahead
- 30 days ahead
- 90 days ahead
- 180 days ahead
- 365 days ahead

main() could show the first 5 rows for each table
------------------------------------------------------------------------------------------------------------------------------
one thing you missed for the rolling averages, it would be better perhaps if you did datediff instead of rows between since not all the rows are reported daily and the frequencies may not be consistent.

for example dollar_index is not reported for 7 days a week so it is more ideal if you did datediff there

for macroeconomic, it would be better if you stuck with monthly gaps like 30 days = 1 month, 90 days = 3 months, and so forth

for market and nass, I would also suggest using using date difference instead since there can be a lot of reports on a single day rather than say one each day
------------------------------------------------------------------------------------------------------------------------------
source: fred_mapping

ingredient_id	ingredient_group	ingredient_description	ingredient_name	unit_of_measure	fred_series_id
10542	FIELD CROPS	ALCOHOL COPRODUCTS	CORN DISTILLERS OIL (CDO)	TON	WPU029201122
10543	FIELD CROPS	ALCOHOL COPRODUCTS	CORN GERM MEAL	TON	WPU029201122
10544	FIELD CROPS	ALCOHOL COPRODUCTS	CORN GLUTEN FEED MEDIUM PROTEIN FROM BRAN & STEEP WATER	TON	WPU029201122
10545	FIELD CROPS	ALCOHOL COPRODUCTS	CORN GLUTEN FEED WET 40 60 PCT MOISTURE	TON	WPU029201122
10546	FIELD CROPS	ALCOHOL COPRODUCTS	CORN GLUTEN MEAL HIGH PROTEIN CONCENTRATE	TON	WPU029201122
10843	FIELD CROPS	OIL	CORN	LB	WPU027B0101
10844	FIELD CROPS	OIL	CORN	TON	WPU027B0101
------------------------------------------------------------------------------------------------------------------------------
source: dollar_index

fred_series_id	name	ppi	ppi_date	ppi_frequency
DTWEXBGS	dollar_index	102.90270233154297	2015-01-02	daily
DTWEXBGS	dollar_index	103.49759674072266	2015-01-05	daily
DTWEXBGS	dollar_index	103.2938003540039	2015-01-06	daily
DTWEXBGS	dollar_index	103.63159942626953	2015-01-07	daily
DTWEXBGS	dollar_index	103.27989959716797	2015-01-08	daily

CROSS JOIN
------------------------------------------------------------------------------------------------------------------------------
source: ppi

ingredient_id	ppi	ppi_date	ppi_frequency
10001	180.6999969482422	2015-01-01	monthly
10001	173.1999969482422	2015-02-01	monthly
10001	175.5	2015-03-01	monthly
10001	175.8000030517578	2015-04-01	monthly
10001	179	2015-05-01	monthly

JOIN ON ingredient_id
------------------------------------------------------------------------------------------------------------------------------
source: market_and_logistic

slug_id	report_title	ams_ingredient_name	ams_ingredient_group	ams_ingredient_grade	price_min	price_max	price_avg	price_unit	sale_type	delivery_point	freight	trans_mode	market_location_state	report_date
3192	ILLINOIS GRAIN BIDS	SOYBEAN	COARSE	US #1	10.614999771118164	10.614999771118164	10.614999771118164	$ PER BUSHEL	BID	BARGE LOADING ELEVATORS	DELIVERED	TRUCK	IL	2025-12-31
3192	ILLINOIS GRAIN BIDS	SOYBEAN	COARSE	US #1	10.614999771118164	10.614999771118164	10.614999771118164	$ PER BUSHEL	BID	BARGE LOADING ELEVATORS	DELIVERED	TRUCK	IL	2025-12-31
3192	ILLINOIS GRAIN BIDS	SOYBEAN	COARSE	US #1	10.265000343322754	10.3149995803833	10.288299560546875	$ PER BUSHEL	BID	BARGE LOADING ELEVATORS	DELIVERED	TRUCK	IL	2025-12-31
3192	ILLINOIS GRAIN BIDS	SOYBEAN	COARSE	US #1	10.274999618530273	10.324999809265137	10.293299674987793	$ PER BUSHEL	BID	BARGE LOADING ELEVATORS	DELIVERED	TRUCK	IL	2025-12-31
3192	ILLINOIS GRAIN BIDS	SOYBEAN	COARSE	US #1	10.354999542236328	10.385000228881836	10.3725004196167	$ PER BUSHEL	BID	BARGE LOADING ELEVATORS	DELIVERED	TRUCK	IL	2025-12-31

JOIN ON ingredient_name (name needs to individually be singular)
------------------------------------------------------------------------------------------------------------------------------
source: nass

ingredient_name	unit_of_measure	amount	cv_%	year	frequency	range	load_time	state	country
ALL CLASSES	BU / ACRE	23.100000381469727		1972	ANNUAL	YEAR	2012-01-01	MI	UNITED STATES
ALL CLASSES	BU	2236000		1965	POINT IN TIME	FIRST OF DEC	2012-01-01	TN	UNITED STATES
ALL CLASSES	PCT	16.260000228881836		1983	ANNUAL	YEAR	2012-01-01	OH	UNITED STATES
ALL CLASSES	TONS	49500		1992	ANNUAL	YEAR	2012-01-01	MO	UNITED STATES
ALL CLASSES	TONS	184200		1991	ANNUAL	YEAR	2012-01-01	NY	UNITED STATES

JOIN ON ingredient_name (names SHOULD match because the name from fred_mapping is taken from here)
------------------------------------------------------------------------------------------------------------------------------
"""

import os
import duckdb

VALIDATED = "data/staged/validated"
FRED_MAPPING_PATH = f"{VALIDATED}/fred_mapping.parquet"
DOLLAR_INDEX_PATH = f"{VALIDATED}/dollar_index.parquet"
MACROECONOMIC_PATH = f"{VALIDATED}/macroeconomic.parquet"
MARKET_PATH = f"{VALIDATED}/market_and_logistic.parquet"
NASS_PATH = f"{VALIDATED}/nass.parquet"

ROLLING_DAYS = [7, 14, 30, 90, 180, 365]
ROLLING_MONTHS = [1, 3, 6, 12]


def search_ingredient(name: str):
    query = f"""
        SELECT *
        FROM read_parquet('{FRED_MAPPING_PATH}')
        WHERE ingredient_name LIKE '%{name}%'
    """
    df = duckdb.sql(query).df()
    if df.empty:
        print(f"No results found for '{name}'. Stopping.")
        exit()
    return df


def _rolling_days(value_col, date_col, partition_col="fm.ingredient_id"):
    """Forward rolling averages using date range intervals (days)."""
    parts = []
    for d in ROLLING_DAYS:
        parts.append(
            f"AVG({value_col}) OVER ("
            f"PARTITION BY {partition_col} "
            f"ORDER BY {date_col} "
            f"RANGE BETWEEN CURRENT ROW AND INTERVAL '{d} days' FOLLOWING"
            f") AS {value_col.split('.')[-1]}_fwd_{d}d"
        )
    return ",\n        ".join(parts)


def _rolling_months(value_col, date_col, partition_col="fm.ingredient_id"):
    """Forward rolling averages using month intervals (for monthly data)."""
    parts = []
    for m in ROLLING_MONTHS:
        parts.append(
            f"AVG({value_col}) OVER ("
            f"PARTITION BY {partition_col} "
            f"ORDER BY {date_col} "
            f"RANGE BETWEEN CURRENT ROW AND INTERVAL '{m} months' FOLLOWING"
            f") AS {value_col.split('.')[-1]}_fwd_{m}mo"
        )
    return ",\n        ".join(parts)


def _fwd_pct_days(value_col):
    """Percent change between each forward rolling avg and the current value (day-based columns)."""
    base = value_col.split(".")[-1]
    parts = []
    for d in ROLLING_DAYS:
        fwd = f"{base}_fwd_{d}d"
        parts.append(
            f"CASE WHEN {value_col} != 0 "
            f"THEN ({fwd} - {value_col}) / {value_col} END AS {fwd}_pct"
        )
    return ",\n        ".join(parts)


def _fwd_pct_months(value_col):
    """Percent change between each forward rolling avg and the current value (month-based columns)."""
    base = value_col.split(".")[-1]
    parts = []
    for m in ROLLING_MONTHS:
        fwd = f"{base}_fwd_{m}mo"
        parts.append(
            f"CASE WHEN {value_col} != 0 "
            f"THEN ({fwd} - {value_col}) / {value_col} END AS {fwd}_pct"
        )
    return ",\n        ".join(parts)


def materialize_dollar_index():
    """Standalone dollar_index table with forward rolling averages and pct changes on ppi."""
    rolling = _rolling_days("di.ppi", "di.ppi_date", partition_col="di.fred_series_id")
    pcts = _fwd_pct_days("ppi")
    query = f"""
        WITH base AS (
            SELECT di.fred_series_id, di.name, di.ppi, di.ppi_date, di.ppi_frequency,
            {rolling}
            FROM read_parquet('{DOLLAR_INDEX_PATH}') di
        )
        SELECT *,
        {pcts}
        FROM base
        ORDER BY ppi_date
    """
    return duckdb.sql(query).df()


def join_macroeconomic(name: str):
    """fred_mapping JOIN macroeconomic ON ingredient_id with forward rolling avg and pct changes on ppi."""
    rolling = _rolling_months("m.ppi", "m.ppi_date")
    pcts = _fwd_pct_months("ppi")
    query = f"""
        WITH base AS (
            SELECT fm.*, m.ppi, m.ppi_date, m.ppi_frequency,
            {rolling}
            FROM read_parquet('{FRED_MAPPING_PATH}') fm
            JOIN read_parquet('{MACROECONOMIC_PATH}') m ON fm.ingredient_id = m.ingredient_id
            WHERE fm.ingredient_name = '{name}'
        )
        SELECT *,
        {pcts}
        FROM base
        ORDER BY ingredient_id, ppi_date
    """
    return duckdb.sql(query).df()


def join_market_and_logistic(name: str):
    """fred_mapping JOIN market_and_logistic ON ingredient_name with forward rolling avg on price_avg.

    Aggregated: one row per (ingredient_id, report_date) using
    AVG(price_avg), MIN(price_min), MAX(price_max).
    """
    rolling = _rolling_days("agg.price_avg", "agg.report_date", "agg.ingredient_id")
    pcts = _fwd_pct_days("price_avg")
    query = f"""
        WITH daily_agg AS (
            SELECT fm.ingredient_id, fm.ingredient_group, fm.ingredient_description,
                   fm.ingredient_name, fm.unit_of_measure, fm.fred_series_id,
                   ml.report_title, ml.ams_ingredient_name,
                   MIN(ml.price_min) AS price_min,
                   MAX(ml.price_max) AS price_max,
                   AVG(ml.price_avg) AS price_avg,
                   ml.price_unit, ml.sale_type, ml.report_date
            FROM read_parquet('{FRED_MAPPING_PATH}') fm
            JOIN read_parquet('{MARKET_PATH}') ml ON fm.ingredient_name = ml.ams_ingredient_name
            WHERE fm.ingredient_name LIKE '%{name}%'
            GROUP BY fm.ingredient_id, fm.ingredient_group, fm.ingredient_description,
                     fm.ingredient_name, fm.unit_of_measure, fm.fred_series_id,
                     ml.report_title, ml.ams_ingredient_name,
                     ml.price_unit, ml.sale_type, ml.report_date
        ),
        with_rolling AS (
            SELECT agg.*,
            {rolling}
            FROM daily_agg agg
        )
        SELECT *,
        {pcts}
        FROM with_rolling
        ORDER BY ingredient_id, report_date
    """
    return duckdb.sql(query).df()


def join_nass(name: str):
    """fred_mapping JOIN nass ON ingredient_name and unit_of_measure with forward rolling avg and pct changes on amount."""
    rolling = _rolling_days("n.amount", "n.load_time")
    pcts = _fwd_pct_days("amount")
    query = f"""
        WITH base AS (
            SELECT fm.*, n.amount, n.year, n.frequency, n.range,
                   n.load_time, n.state, n.country,
            {rolling}
            FROM read_parquet('{FRED_MAPPING_PATH}') fm
            JOIN read_parquet('{NASS_PATH}') n
                ON fm.ingredient_name = n.ingredient_name
                AND fm.unit_of_measure = n.unit_of_measure
            WHERE fm.ingredient_name LIKE '%{name}%'
        )
        SELECT *,
        {pcts}
        FROM base
        ORDER BY ingredient_id, load_time
    """
    return duckdb.sql(query).df()


# ---------------------------------------------------------------------------
# Incremental support: check if source is newer than materialized output
# ---------------------------------------------------------------------------
# Maps each materialized dataset to the validated sources it depends on
SOURCE_DEPS = {
    "dollar_index": [DOLLAR_INDEX_PATH],
    "macroeconomic": [MACROECONOMIC_PATH, FRED_MAPPING_PATH],
    "market_and_logistic": [MARKET_PATH, FRED_MAPPING_PATH],
    "nass": [NASS_PATH, FRED_MAPPING_PATH],
}

MATERIALIZERS = {
    "dollar_index": ("dollar_index (standalone)", materialize_dollar_index),
    "macroeconomic": (
        "fred_mapping x macroeconomic (JOIN ON ingredient_id)",
        join_macroeconomic,
    ),
    "market_and_logistic": (
        "fred_mapping x market_and_logistic (JOIN ON ingredient_name)",
        join_market_and_logistic,
    ),
    "nass": (
        "fred_mapping x nass (JOIN ON ingredient_name, unit_of_measure)",
        join_nass,
    ),
}


def _needs_rebuild(output_path, source_paths):
    """Return True if any source is newer than the output, or output doesn't exist."""
    if not os.path.exists(output_path):
        return True
    out_mtime = os.path.getmtime(output_path)
    for src in source_paths:
        if os.path.exists(src) and os.path.getmtime(src) > out_mtime:
            return True
    return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Materialize joined tables")
    parser.add_argument("name", nargs="?", help="Ingredient name to search")
    parser.add_argument(
        "--only",
        choices=list(MATERIALIZERS.keys()) + ["all"],
        default="all",
        help="Materialize only a specific dataset (default: all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild even if sources haven't changed",
    )
    args = parser.parse_args()

    name = args.name or input("Enter ingredient name to search: ")
    search_ingredient(name)

    output_dir = f"data/materialized/{name}"
    os.makedirs(output_dir, exist_ok=True)

    targets = (
        MATERIALIZERS if args.only == "all" else {args.only: MATERIALIZERS[args.only]}
    )

    for dataset, (label, join_fn) in targets.items():
        out_path = f"{output_dir}/{dataset}.parquet"

        if not args.force and not _needs_rebuild(out_path, SOURCE_DEPS[dataset]):
            print(f"\n=== {label} === SKIPPED (sources unchanged)")
            continue

        print(f"\n=== {label} ===")
        df = join_fn() if dataset == "dollar_index" else join_fn(name)
        print(f"{len(df)} rows")
        print(df.head().to_string())
        df.to_parquet(out_path, index=False)

    print(f"\nSaved results to {output_dir}/")
