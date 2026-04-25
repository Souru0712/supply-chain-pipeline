"""Great Expectations — Raw Source Validation (Pre-Ingestion)

Validates parquet/CSV files in data/raw/ BEFORE they are transformed or
staged.  Bad rows are quarantined so only clean data proceeds to
4.transformed.py.

Domains: commodity_prices (FRED PPI), dollar_index, market_and_logistic (AMS)
"""

import glob
import logging
import os

import pandas as pd
import great_expectations as gx
from great_expectations.expectations import (
    ExpectColumnValuesToNotBeNull,
    ExpectColumnValuesToBeOfType,
    ExpectCompoundColumnsToBeUnique,
    ExpectColumnValuesToMatchRegex,
)

os.makedirs("logs", exist_ok=True)
os.makedirs("data/raw/quarantine", exist_ok=True)
os.makedirs("data/raw/validated", exist_ok=True)

context = gx.get_context(mode="ephemeral")

logger = logging.getLogger("ge_checkpoint")
logger.setLevel(logging.INFO)
fh = logging.FileHandler("logs/ge_checkpoint.log")
fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s  %(message)s"))
logger.addHandler(fh)


# =========================================================================================================
# commodity_prices  (data/raw/macroeconomic/commodity_prices_*.csv)
# Produced by 2a.ingest_fred.py
# Columns: date, ppi, ingredient_id, fred_series_id, ingredient_list_of_name
# =========================================================================================================
def validate_commodity_prices():
    """
    Expectations on raw FRED PPI CSV (pre-transform):
      1. (ingredient_id, date) has no duplicates
      2. ppi is numeric (float-castable), not NULL
      3. date is formatted yyyy-mm-dd, not NULL
      4. ingredient_id and fred_series_id are not NULL
    """
    logger.info("commodity_prices validation started")

    pattern = "data/raw/macroeconomic/commodity_prices_*.csv"
    files = sorted(glob.glob(pattern))
    if not files:
        logger.warning(f"no files matched {pattern} — skipping")
        return True

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    logger.info(f"loaded {len(df)} rows from {len(files)} commodity_prices file(s)")

    # Ensure ppi is float for GE type check
    df["ppi"] = pd.to_numeric(df["ppi"], errors="coerce")

    ds = context.data_sources.add_pandas("cp_ds")
    asset = ds.add_dataframe_asset("cp_asset")
    batch_def = asset.add_batch_definition_whole_dataframe("cp_batch")
    batch = batch_def.get_batch(batch_parameters={"dataframe": df})

    results = {}

    # 1 — no duplicate (ingredient_id, date) pairs
    r = batch.validate(
        ExpectCompoundColumnsToBeUnique(column_list=["ingredient_id", "date"])
    )
    results["compound_unique"] = r
    logger.info(f"compound_unique (ingredient_id, date): success={r.success}")

    # 2a — ppi not null
    r = batch.validate(ExpectColumnValuesToNotBeNull(column="ppi"))
    results["ppi_not_null"] = r
    logger.info(f"ppi_not_null: success={r.success}")

    # 2b — ppi is float type
    r = batch.validate(ExpectColumnValuesToBeOfType(column="ppi", type_="float64"))
    results["ppi_type"] = r
    logger.info(f"ppi_type (float64): success={r.success}")

    # 3a — date not null
    r = batch.validate(ExpectColumnValuesToNotBeNull(column="date"))
    results["date_not_null"] = r
    logger.info(f"date_not_null: success={r.success}")

    # 3b — date matches yyyy-mm-dd
    r = batch.validate(
        ExpectColumnValuesToMatchRegex(column="date", regex=r"^\d{4}-\d{2}-\d{2}$")
    )
    results["date_format"] = r
    logger.info(f"date_format (yyyy-mm-dd): success={r.success}")

    # 4a — ingredient_id not null
    r = batch.validate(ExpectColumnValuesToNotBeNull(column="ingredient_id"))
    results["ingredient_id_not_null"] = r
    logger.info(f"ingredient_id_not_null: success={r.success}")

    # 4b — fred_series_id not null
    r = batch.validate(ExpectColumnValuesToNotBeNull(column="fred_series_id"))
    results["fred_series_id_not_null"] = r
    logger.info(f"fred_series_id_not_null: success={r.success}")

    _quarantine(
        df,
        results,
        bad_masks=_commodity_prices_masks(df, results),
        domain="commodity_prices",
    )

    all_passed = all(r.success for r in results.values())
    logger.info(f"commodity_prices validation {'PASSED' if all_passed else 'FAILED'}")
    return all_passed


def _commodity_prices_masks(df, results):
    masks = []
    if not results["compound_unique"].success:
        masks.append(df.duplicated(subset=["ingredient_id", "date"], keep=False))
    if not results["ppi_not_null"].success:
        masks.append(df["ppi"].isna())
    if not results["date_not_null"].success or not results["date_format"].success:
        masks.append(
            df["date"].isna()
            | ~df["date"].astype(str).str.match(r"^\d{4}-\d{2}-\d{2}$")
        )
    if not results["ingredient_id_not_null"].success:
        masks.append(df["ingredient_id"].isna())
    if not results["fred_series_id_not_null"].success:
        masks.append(df["fred_series_id"].isna())
    return masks


# =========================================================================================================
# dollar_index  (data/raw/macroeconomic/dollar_index_*.csv)
# Produced by 2b.ingest_dollar_index.py
# Columns: fred_series_id, name, ppi, ppi_date, ppi_frequency
# =========================================================================================================
def validate_dollar_index():
    """
    Expectations on raw FRED dollar-index CSV (pre-transform):
      1. ppi is numeric, not NULL
      2. ppi_date is formatted yyyy-mm-dd, not NULL
      3. ppi_frequency is not NULL
    """
    logger.info("dollar_index validation started")

    pattern = "data/raw/macroeconomic/dollar_index_*.csv"
    files = sorted(glob.glob(pattern))
    if not files:
        logger.warning(f"no files matched {pattern} — skipping")
        return True

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    logger.info(f"loaded {len(df)} rows from {len(files)} dollar_index file(s)")

    df["ppi"] = pd.to_numeric(df["ppi"], errors="coerce")

    ds = context.data_sources.add_pandas("di_ds")
    asset = ds.add_dataframe_asset("di_asset")
    batch_def = asset.add_batch_definition_whole_dataframe("di_batch")
    batch = batch_def.get_batch(batch_parameters={"dataframe": df})

    results = {}

    # 1a — ppi not null
    r = batch.validate(ExpectColumnValuesToNotBeNull(column="ppi"))
    results["ppi_not_null"] = r
    logger.info(f"ppi_not_null: success={r.success}")

    # 1b — ppi is float type
    r = batch.validate(ExpectColumnValuesToBeOfType(column="ppi", type_="float64"))
    results["ppi_type"] = r
    logger.info(f"ppi_type (float64): success={r.success}")

    # 2a — ppi_date not null
    r = batch.validate(ExpectColumnValuesToNotBeNull(column="ppi_date"))
    results["ppi_date_not_null"] = r
    logger.info(f"ppi_date_not_null: success={r.success}")

    # 2b — ppi_date matches yyyy-mm-dd
    r = batch.validate(
        ExpectColumnValuesToMatchRegex(
            column="ppi_date", regex=r"^\d{4}-\d{2}-\d{2}$"
        )
    )
    results["ppi_date_format"] = r
    logger.info(f"ppi_date_format (yyyy-mm-dd): success={r.success}")

    # 3 — ppi_frequency not null
    r = batch.validate(ExpectColumnValuesToNotBeNull(column="ppi_frequency"))
    results["ppi_frequency_not_null"] = r
    logger.info(f"ppi_frequency_not_null: success={r.success}")

    _quarantine(
        df,
        results,
        bad_masks=_dollar_index_masks(df, results),
        domain="dollar_index",
    )

    all_passed = all(r.success for r in results.values())
    logger.info(f"dollar_index validation {'PASSED' if all_passed else 'FAILED'}")
    return all_passed


def _dollar_index_masks(df, results):
    masks = []
    if not results["ppi_not_null"].success:
        masks.append(df["ppi"].isna())
    if not results["ppi_date_not_null"].success or not results["ppi_date_format"].success:
        masks.append(
            df["ppi_date"].isna()
            | ~df["ppi_date"].astype(str).str.match(r"^\d{4}-\d{2}-\d{2}$")
        )
    if not results["ppi_frequency_not_null"].success:
        masks.append(df["ppi_frequency"].isna())
    return masks


# =========================================================================================================
# market_and_logistic  (data/raw/market_and_logistic/AMS_*/ReportDetail.csv)
# Produced by 2c.ingest_ams.py
# Key columns: report_date (MM/DD/YYYY from AMS API), commodity
# =========================================================================================================
def validate_market_and_logistic():
    """
    Expectations on raw AMS ReportDetail CSVs (pre-transform):
      1. report_date is not NULL
      2. report_date matches MM/DD/YYYY (raw AMS format before staging normalises to yyyy-mm-dd)
      3. commodity is not NULL
    """
    logger.info("market_and_logistic validation started")

    pattern = "data/raw/market_and_logistic/AMS_*/ReportDetail.csv"
    files = sorted(glob.glob(pattern))
    if not files:
        logger.warning(f"no files matched {pattern} — skipping")
        return True

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    logger.info(
        f"loaded {len(df)} rows from {len(files)} AMS ReportDetail file(s)"
    )

    ds = context.data_sources.add_pandas("ams_ds")
    asset = ds.add_dataframe_asset("ams_asset")
    batch_def = asset.add_batch_definition_whole_dataframe("ams_batch")
    batch = batch_def.get_batch(batch_parameters={"dataframe": df})

    results = {}

    if "report_date" in df.columns:
        # 1 — report_date not null
        r = batch.validate(ExpectColumnValuesToNotBeNull(column="report_date"))
        results["report_date_not_null"] = r
        logger.info(f"report_date_not_null: success={r.success}")

        # 2 — report_date matches MM/DD/YYYY (raw AMS format)
        r = batch.validate(
            ExpectColumnValuesToMatchRegex(
                column="report_date",
                regex=r"^\d{1,2}/\d{1,2}/\d{4}$",
            )
        )
        results["report_date_format"] = r
        logger.info(f"report_date_format (MM/DD/YYYY): success={r.success}")

    if "commodity" in df.columns:
        # 3 — commodity not null
        r = batch.validate(ExpectColumnValuesToNotBeNull(column="commodity"))
        results["commodity_not_null"] = r
        logger.info(f"commodity_not_null: success={r.success}")

    _quarantine(
        df,
        results,
        bad_masks=_ams_masks(df, results),
        domain="market_and_logistic",
    )

    all_passed = all(r.success for r in results.values())
    logger.info(
        f"market_and_logistic validation {'PASSED' if all_passed else 'FAILED'}"
    )
    return all_passed


def _ams_masks(df, results):
    masks = []
    if "report_date" in df.columns:
        if not results.get("report_date_not_null", type("", (), {"success": True})()).success:
            masks.append(df["report_date"].isna())
        if not results.get("report_date_format", type("", (), {"success": True})()).success:
            masks.append(
                df["report_date"].isna()
                | ~df["report_date"].astype(str).str.match(r"^\d{1,2}/\d{1,2}/\d{4}$")
            )
    if "commodity" in df.columns:
        if not results.get("commodity_not_null", type("", (), {"success": True})()).success:
            masks.append(df["commodity"].isna())
    return masks


# =========================================================================================================
# Shared quarantine helper
# =========================================================================================================
def _quarantine(df: pd.DataFrame, results: dict, bad_masks: list, domain: str):
    """Write failed rows to data/raw/quarantine/ and clean rows to data/raw/validated/."""
    import functools

    if bad_masks:
        bad_mask = functools.reduce(lambda a, b: a | b, bad_masks)
    else:
        bad_mask = pd.Series(False, index=df.index)

    quarantined = df[bad_mask]
    clean = df[~bad_mask]

    if len(quarantined) > 0:
        path = f"data/raw/quarantine/{domain}.parquet"
        quarantined.to_parquet(path, index=False)
        logger.info(f"  quarantined {len(quarantined)} rows → {path}")
    else:
        logger.info("  no rows quarantined — all raw checks passed")

    path = f"data/raw/validated/{domain}.parquet"
    clean.to_parquet(path, index=False)
    logger.info(f"  validated {len(clean)} rows → {path}")


# =========================================================================================================
# main
# =========================================================================================================
VALIDATORS = {
    "commodity_prices": validate_commodity_prices,
    "dollar_index": validate_dollar_index,
    "market_and_logistic": validate_market_and_logistic,
}

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Great Expectations pre-ingestion source validation"
    )
    parser.add_argument(
        "--only",
        choices=list(VALIDATORS.keys()) + ["all"],
        default="all",
        help="Validate only a specific source (default: all)",
    )
    args = parser.parse_args()

    targets = (
        VALIDATORS if args.only == "all" else {args.only: VALIDATORS[args.only]}
    )

    logger.info(f"ge_checkpoint (pre-ingestion) started (target={args.only})")
    try:
        for name, validate_fn in targets.items():
            ok = validate_fn()
            status = "PASS" if ok else "FAIL"
            logger.info(f"{name}: {status}")
            print(f"{name}: {status}")
    except Exception as e:
        logger.error(f"pipeline failed: {e}", exc_info=True)
        raise
    logger.info("ge_checkpoint (pre-ingestion) completed")
