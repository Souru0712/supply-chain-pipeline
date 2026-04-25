"""Great Expectations — Staged Data Validation

Validates parquet files in data/staged/ and quarantines rows that fail checks.
Domains: macroeconomic, market_and_logistic, production
"""

import logging
import os
import pandas as pd
import great_expectations as gx
from great_expectations.expectations import (
    ExpectColumnValuesToNotBeNull,
    ExpectColumnValuesToBeOfType,
    ExpectColumnValuesToBeUnique,
    ExpectCompoundColumnsToBeUnique,
    ExpectColumnValuesToMatchRegex,
    ExpectColumnValuesToNotMatchRegex,
)

os.makedirs("logs", exist_ok=True)
os.makedirs("data/staged/quarantine", exist_ok=True)
os.makedirs("data/staged/validated", exist_ok=True)

# fresh ephemeral context each run
context = gx.get_context(mode="ephemeral")

logger = logging.getLogger("ge_checkpoint")
logger.setLevel(logging.INFO)
fh = logging.FileHandler("logs/ge_checkpoint.log")
fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s  %(message)s"))
logger.addHandler(fh)


# =========================================================================================================
# macroeconomic
# =========================================================================================================
def validate_macroeconomic():
    """
    Expectations:
      1. ingredient_id + ppi_date has no duplicates
      2. ppi is float, not NULL, and not 0.0
      3. ppi_date is date-formatted yyyy-mm-dd, not NULL
      4. ppi_frequency is string, not NULL
    """
    logger.info("macroeconomic validation started")

    df = pd.read_parquet("data/staged/macroeconomic/")
    logger.info(f"loaded {len(df)} rows from data/staged/macroeconomic/")

    ds = context.data_sources.add_pandas("macro_ds")
    asset = ds.add_dataframe_asset("macro_asset")
    batch_def = asset.add_batch_definition_whole_dataframe("macro_batch")
    batch = batch_def.get_batch(batch_parameters={"dataframe": df})

    results = {}

    # 1 — no duplicate (ingredient_id, ppi_date) pairs
    r = batch.validate(
        ExpectCompoundColumnsToBeUnique(column_list=["ingredient_id", "ppi_date"])
    )
    results["compound_unique"] = r
    logger.info(f"compound_unique (ingredient_id, ppi_date): success={r.success}")

    # 2a — ppi not null
    r = batch.validate(ExpectColumnValuesToNotBeNull(column="ppi"))
    results["ppi_not_null"] = r
    logger.info(f"ppi_not_null: success={r.success}")

    # 2b — ppi is float type
    r = batch.validate(ExpectColumnValuesToBeOfType(column="ppi", type_="float32"))
    results["ppi_type"] = r
    logger.info(f"ppi_type (float32): success={r.success}")

    # 2c — ppi is not 0.0
    r = batch.validate(ExpectColumnValuesToNotMatchRegex(column="ppi", regex=r"^0\.0$"))
    results["ppi_not_zero"] = r
    logger.info(f"ppi_not_zero: success={r.success}")

    # 3a — ppi_date not null
    r = batch.validate(ExpectColumnValuesToNotBeNull(column="ppi_date"))
    results["ppi_date_not_null"] = r
    logger.info(f"ppi_date_not_null: success={r.success}")

    # 3b — ppi_date matches yyyy-mm-dd
    r = batch.validate(
        ExpectColumnValuesToMatchRegex(column="ppi_date", regex=r"^\d{4}-\d{2}-\d{2}$")
    )
    results["ppi_date_format"] = r
    logger.info(f"ppi_date_format (yyyy-mm-dd): success={r.success}")

    # 4a — ppi_frequency not null
    r = batch.validate(ExpectColumnValuesToNotBeNull(column="ppi_frequency"))
    results["ppi_frequency_not_null"] = r
    logger.info(f"ppi_frequency_not_null: success={r.success}")

    # --- quarantine failed rows ---
    quarantine_macroeconomic(df, results)

    all_passed = all(r.success for r in results.values())
    logger.info(f"macroeconomic validation {'PASSED' if all_passed else 'FAILED'}")
    return all_passed


def quarantine_macroeconomic(df, results):
    """Identify and move rows that failed any expectation to quarantine."""
    bad_mask = pd.Series(False, index=df.index)

    # duplicate (ingredient_id, ppi_date) rows
    if not results["compound_unique"].success:
        dupes = df.duplicated(subset=["ingredient_id", "ppi_date"], keep=False)
        bad_mask |= dupes
        logger.info(
            f"  quarantine: {dupes.sum()} rows have duplicate (ingredient_id, ppi_date)"
        )

    # ppi is null
    if not results["ppi_not_null"].success:
        mask = df["ppi"].isna()
        bad_mask |= mask
        logger.info(f"  quarantine: {mask.sum()} rows have null ppi")

    # ppi is 0.0
    if not results["ppi_not_zero"].success:
        mask = df["ppi"] == 0.0
        bad_mask |= mask
        logger.info(f"  quarantine: {mask.sum()} rows have ppi=0.0")

    # ppi_date null or bad format
    if (
        not results["ppi_date_not_null"].success
        or not results["ppi_date_format"].success
    ):
        mask = df["ppi_date"].isna() | ~df["ppi_date"].astype(str).str.match(
            r"^\d{4}-\d{2}-\d{2}$"
        )
        bad_mask |= mask
        logger.info(f"  quarantine: {mask.sum()} rows have null/bad ppi_date")

    # ppi_frequency null
    if not results["ppi_frequency_not_null"].success:
        mask = df["ppi_frequency"].isna()
        bad_mask |= mask
        logger.info(f"  quarantine: {mask.sum()} rows have null ppi_frequency")

    quarantined = df[bad_mask]
    clean = df[~bad_mask]

    if len(quarantined) > 0:
        quarantined.to_parquet(
            "data/staged/quarantine/macroeconomic.parquet", index=False
        )
        logger.info(
            f"  quarantined {len(quarantined)} rows → data/staged/quarantine/macroeconomic.parquet"
        )
    else:
        logger.info("  no rows quarantined — all checks passed at row level")

    clean.to_parquet("data/staged/validated/macroeconomic.parquet", index=False)
    logger.info(
        f"  validated {len(clean)} rows → data/staged/validated/macroeconomic.parquet"
    )


# =========================================================================================================
# dollar_index
# =========================================================================================================
def validate_dollar_index():
    """
    Expectations:
      1. ppi is float, not NULL, and not 0.0
      2. ppi_date is date-formatted yyyy-mm-dd, not NULL
      3. ppi_frequency is string, not NULL
    """
    logger.info("dollar_index validation started")

    df = pd.read_parquet("data/staged/dollar_index/")
    logger.info(f"loaded {len(df)} rows from data/staged/dollar_index/")

    ds = context.data_sources.add_pandas("dollar_ds")
    asset = ds.add_dataframe_asset("dollar_asset")
    batch_def = asset.add_batch_definition_whole_dataframe("dollar_batch")
    batch = batch_def.get_batch(batch_parameters={"dataframe": df})

    results = {}

    # 1a — ppi not null
    r = batch.validate(ExpectColumnValuesToNotBeNull(column="ppi"))
    results["ppi_not_null"] = r
    logger.info(f"ppi_not_null: success={r.success}")

    # 1b — ppi is float type
    r = batch.validate(ExpectColumnValuesToBeOfType(column="ppi", type_="float32"))
    results["ppi_type"] = r
    logger.info(f"ppi_type (float32): success={r.success}")

    # 1c — ppi is not 0.0
    r = batch.validate(ExpectColumnValuesToNotMatchRegex(column="ppi", regex=r"^0\.0$"))
    results["ppi_not_zero"] = r
    logger.info(f"ppi_not_zero: success={r.success}")

    # 2a — ppi_date not null
    r = batch.validate(ExpectColumnValuesToNotBeNull(column="ppi_date"))
    results["ppi_date_not_null"] = r
    logger.info(f"ppi_date_not_null: success={r.success}")

    # 2b — ppi_date matches yyyy-mm-dd
    r = batch.validate(
        ExpectColumnValuesToMatchRegex(column="ppi_date", regex=r"^\d{4}-\d{2}-\d{2}$")
    )
    results["ppi_date_format"] = r
    logger.info(f"ppi_date_format (yyyy-mm-dd): success={r.success}")

    # 3a — ppi_frequency not null
    r = batch.validate(ExpectColumnValuesToNotBeNull(column="ppi_frequency"))
    results["ppi_frequency_not_null"] = r
    logger.info(f"ppi_frequency_not_null: success={r.success}")

    # --- quarantine failed rows ---
    quarantine_dollar_index(df, results)

    all_passed = all(r.success for r in results.values())
    logger.info(f"dollar_index validation {'PASSED' if all_passed else 'FAILED'}")
    return all_passed


def quarantine_dollar_index(df, results):
    """Identify and move rows that failed any expectation to quarantine."""
    bad_mask = pd.Series(False, index=df.index)

    # ppi is null
    if not results["ppi_not_null"].success:
        mask = df["ppi"].isna()
        bad_mask |= mask
        logger.info(f"  quarantine: {mask.sum()} rows have null ppi")

    # ppi is 0.0
    if not results["ppi_not_zero"].success:
        mask = df["ppi"] == 0.0
        bad_mask |= mask
        logger.info(f"  quarantine: {mask.sum()} rows have ppi=0.0")

    # ppi_date null or bad format
    if (
        not results["ppi_date_not_null"].success
        or not results["ppi_date_format"].success
    ):
        mask = df["ppi_date"].isna() | ~df["ppi_date"].astype(str).str.match(
            r"^\d{4}-\d{2}-\d{2}$"
        )
        bad_mask |= mask
        logger.info(f"  quarantine: {mask.sum()} rows have null/bad ppi_date")

    # ppi_frequency null
    if not results["ppi_frequency_not_null"].success:
        mask = df["ppi_frequency"].isna()
        bad_mask |= mask
        logger.info(f"  quarantine: {mask.sum()} rows have null ppi_frequency")

    quarantined = df[bad_mask]
    clean = df[~bad_mask]

    if len(quarantined) > 0:
        quarantined.to_parquet(
            "data/staged/quarantine/dollar_index.parquet", index=False
        )
        logger.info(
            f"  quarantined {len(quarantined)} rows → data/staged/quarantine/dollar_index.parquet"
        )
    else:
        logger.info("  no rows quarantined — all checks passed at row level")

    clean.to_parquet("data/staged/validated/dollar_index.parquet", index=False)
    logger.info(
        f"  validated {len(clean)} rows → data/staged/validated/dollar_index.parquet"
    )


# =========================================================================================================
# market & logistic
# =========================================================================================================
MARKET_STRING_COLS = [
    "report_title",
    "ams_ingredient_name",
    "ams_ingredient_group",
    "ams_ingredient_grade",
    "price_unit",
    "sale_type",
    "delivery_point",
    "freight",
    "trans_mode",
    "market_location_state",
]


def validate_market_and_logistic():
    """
    Expectations:
      1. String columns are fully uppercased (no lowercase letters)
      2. ams_ingredient_name is not plural (does not end in S)
      3. price_min, price_max, price_avg are float (NULLs allowed)
      4. report_date is formatted yyyy-mm-dd
    """
    logger.info("market_and_logistic validation started")

    df = pd.read_parquet("data/staged/market_and_logistic/")
    logger.info(f"loaded {len(df)} rows from data/staged/market_and_logistic/")

    ds = context.data_sources.add_pandas("market_ds")
    asset = ds.add_dataframe_asset("market_asset")
    batch_def = asset.add_batch_definition_whole_dataframe("market_batch")
    batch = batch_def.get_batch(batch_parameters={"dataframe": df})

    results = {}

    # 1 — string columns are uppercased (no lowercase chars)
    for col_name in MARKET_STRING_COLS:
        r = batch.validate(
            ExpectColumnValuesToNotMatchRegex(column=col_name, regex=r"[a-z]")
        )
        results[f"{col_name}_upper"] = r
        logger.info(f"{col_name}_upper: success={r.success}")

    # 3 — price columns: float type (NULLs allowed)
    for col_name in ["price_min", "price_max", "price_avg"]:
        r = batch.validate(
            ExpectColumnValuesToBeOfType(column=col_name, type_="float32")
        )
        results[f"{col_name}_type"] = r
        logger.info(f"{col_name}_type (float32): success={r.success}")

    # 4 — report_date formatted yyyy-mm-dd
    r = batch.validate(
        ExpectColumnValuesToMatchRegex(
            column="report_date", regex=r"^\d{4}-\d{2}-\d{2}$"
        )
    )
    results["report_date_format"] = r
    logger.info(f"report_date_format (yyyy-mm-dd): success={r.success}")

    # --- quarantine failed rows ---
    quarantine_market_and_logistic(df, results)

    all_passed = all(r.success for r in results.values())
    logger.info(
        f"market_and_logistic validation {'PASSED' if all_passed else 'FAILED'}"
    )
    return all_passed


def quarantine_market_and_logistic(df, results):
    """Identify and move rows that failed any expectation to quarantine."""
    bad_mask = pd.Series(False, index=df.index)

    # string columns with lowercase
    for col_name in MARKET_STRING_COLS:
        if not results[f"{col_name}_upper"].success:
            mask = df[col_name].fillna("").str.contains(r"[a-z]", regex=True)
            bad_mask |= mask
            logger.info(f"  quarantine: {mask.sum()} rows have lowercase in {col_name}")

    # bad report_date
    if not results["report_date_format"].success:
        mask = df["report_date"].isna() | ~df["report_date"].astype(str).str.match(
            r"^\d{4}-\d{2}-\d{2}$"
        )
        bad_mask |= mask
        logger.info(f"  quarantine: {mask.sum()} rows have null/bad report_date")

    quarantined = df[bad_mask]
    clean = df[~bad_mask]

    if len(quarantined) > 0:
        quarantined.to_parquet(
            "data/staged/quarantine/market_and_logistic.parquet", index=False
        )
        logger.info(
            f"  quarantined {len(quarantined)} rows → data/staged/quarantine/market_and_logistic.parquet"
        )
    else:
        logger.info("  no rows quarantined — all checks passed at row level")

    clean.to_parquet("data/staged/validated/market_and_logistic.parquet", index=False)
    logger.info(
        f"  validated {len(clean)} rows → data/staged/validated/market_and_logistic.parquet"
    )


# =========================================================================================================
# production
# =========================================================================================================
NASS_STRING_COLS = [
    "ingredient_name",
    "unit_of_measure",
    "frequency",
    "range",
    "state",
    "country",
]


def validate_nass():
    """
    Expectations:
      1. String columns are fully uppercased (no lowercase letters)
      2. ingredient_name, unit_of_measure, amount are not NULL
      3. load_time is formatted yyyy-mm-dd
    """
    logger.info("nass validation started")

    df = pd.read_parquet("data/staged/production/nass/")
    logger.info(f"loaded {len(df)} rows from data/staged/production/nass/")

    ds = context.data_sources.add_pandas("nass_ds")
    asset = ds.add_dataframe_asset("nass_asset")
    batch_def = asset.add_batch_definition_whole_dataframe("nass_batch")
    batch = batch_def.get_batch(batch_parameters={"dataframe": df})

    results = {}

    # 1 — string columns are uppercased (no lowercase chars)
    for col_name in NASS_STRING_COLS:
        r = batch.validate(
            ExpectColumnValuesToNotMatchRegex(column=col_name, regex=r"[a-z]")
        )
        results[f"{col_name}_upper"] = r
        logger.info(f"{col_name}_upper: success={r.success}")

    # 2 — ingredient_name, unit_of_measure, amount not null
    for col_name in ["ingredient_name", "unit_of_measure", "amount"]:
        r = batch.validate(ExpectColumnValuesToNotBeNull(column=col_name))
        results[f"{col_name}_not_null"] = r
        logger.info(f"{col_name}_not_null: success={r.success}")

    # 3 — load_time formatted yyyy-mm-dd (Spark DateType → pandas object/date)
    r = batch.validate(
        ExpectColumnValuesToMatchRegex(column="load_time", regex=r"^\d{4}-\d{2}-\d{2}$")
    )
    results["load_time_format"] = r
    logger.info(f"load_time_format (yyyy-mm-dd): success={r.success}")

    # --- quarantine failed rows ---
    quarantine_nass(df, results)

    all_passed = all(r.success for r in results.values())
    logger.info(f"nass validation {'PASSED' if all_passed else 'FAILED'}")
    return all_passed


def quarantine_nass(df, results):
    """Identify and move rows that failed any expectation to quarantine."""
    bad_mask = pd.Series(False, index=df.index)

    # string columns with lowercase
    for col_name in NASS_STRING_COLS:
        if not results[f"{col_name}_upper"].success:
            mask = df[col_name].fillna("").str.contains(r"[a-z]", regex=True)
            bad_mask |= mask
            logger.info(f"  quarantine: {mask.sum()} rows have lowercase in {col_name}")

    # null checks
    for col_name in ["ingredient_name", "unit_of_measure", "amount"]:
        if not results[f"{col_name}_not_null"].success:
            mask = df[col_name].isna()
            bad_mask |= mask
            logger.info(f"  quarantine: {mask.sum()} rows have null {col_name}")

    # load_time null or bad format
    if not results["load_time_format"].success:
        mask = df["load_time"].isna() | ~df["load_time"].astype(str).str.match(
            r"^\d{4}-\d{2}-\d{2}$"
        )
        bad_mask |= mask
        logger.info(f"  quarantine: {mask.sum()} rows have null/bad load_time")

    quarantined = df[bad_mask]
    clean = df[~bad_mask]

    if len(quarantined) > 0:
        quarantined.to_parquet("data/staged/quarantine/nass.parquet", index=False)
        logger.info(
            f"  quarantined {len(quarantined)} rows → data/staged/quarantine/nass.parquet"
        )
    else:
        logger.info("  no rows quarantined — all checks passed at row level")

    clean.to_parquet("data/staged/validated/nass.parquet", index=False)
    logger.info(f"  validated {len(clean)} rows → data/staged/validated/nass.parquet")


def validate_fred_mapping():
    """
    Expectations:
      1. ingredient_id has no duplicate values
      2. ingredient_id is not NULL
    """
    logger.info("fred_mapping validation started")

    df = pd.read_parquet("data/staged/production/fred_mapping/")
    logger.info(f"loaded {len(df)} rows from data/staged/production/fred_mapping/")

    ds = context.data_sources.add_pandas("fred_map_ds")
    asset = ds.add_dataframe_asset("fred_map_asset")
    batch_def = asset.add_batch_definition_whole_dataframe("fred_map_batch")
    batch = batch_def.get_batch(batch_parameters={"dataframe": df})

    results = {}

    # 1 — ingredient_id is unique
    r = batch.validate(ExpectColumnValuesToBeUnique(column="ingredient_id"))
    results["ingredient_id_unique"] = r
    logger.info(f"ingredient_id_unique: success={r.success}")

    # 2 — ingredient_id not null
    r = batch.validate(ExpectColumnValuesToNotBeNull(column="ingredient_id"))
    results["ingredient_id_not_null"] = r
    logger.info(f"ingredient_id_not_null: success={r.success}")

    # --- quarantine failed rows ---
    quarantine_fred_mapping(df, results)

    all_passed = all(r.success for r in results.values())
    logger.info(f"fred_mapping validation {'PASSED' if all_passed else 'FAILED'}")
    return all_passed


def quarantine_fred_mapping(df, results):
    """Identify and move rows that failed any expectation to quarantine."""
    bad_mask = pd.Series(False, index=df.index)

    # duplicate ingredient_id
    if not results["ingredient_id_unique"].success:
        dupes = df.duplicated(subset=["ingredient_id"], keep=False)
        bad_mask |= dupes
        logger.info(f"  quarantine: {dupes.sum()} rows have duplicate ingredient_id")

    # null ingredient_id
    if not results["ingredient_id_not_null"].success:
        mask = df["ingredient_id"].isna()
        bad_mask |= mask
        logger.info(f"  quarantine: {mask.sum()} rows have null ingredient_id")

    quarantined = df[bad_mask]
    clean = df[~bad_mask]

    if len(quarantined) > 0:
        quarantined.to_parquet(
            "data/staged/quarantine/fred_mapping.parquet", index=False
        )
        logger.info(
            f"  quarantined {len(quarantined)} rows → data/staged/quarantine/fred_mapping.parquet"
        )
    else:
        logger.info("  no rows quarantined — all checks passed at row level")

    clean.to_parquet("data/staged/validated/fred_mapping.parquet", index=False)
    logger.info(
        f"  validated {len(clean)} rows → data/staged/validated/fred_mapping.parquet"
    )


# =========================================================================================================
# main
# =========================================================================================================
VALIDATORS = {
    "macroeconomic": validate_macroeconomic,
    "dollar_index": validate_dollar_index,
    "market_and_logistic": validate_market_and_logistic,
    "nass": validate_nass,
    "fred_mapping": validate_fred_mapping,
}

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Great Expectations validation")
    parser.add_argument(
        "--only",
        choices=list(VALIDATORS.keys()) + ["all"],
        default="all",
        help="Validate only a specific source (default: all)",
    )
    args = parser.parse_args()

    targets = VALIDATORS if args.only == "all" else {args.only: VALIDATORS[args.only]}

    logger.info(f"ge_checkpoint pipeline started (target={args.only})")
    try:
        for name, validate_fn in targets.items():
            ok = validate_fn()
            status = "PASS" if ok else "FAIL"
            logger.info(f"{name}: {status}")
            print(f"{name}: {status}")
    except Exception as e:
        logger.error(f"pipeline failed: {e}", exc_info=True)
        raise
    logger.info("ge_checkpoint pipeline completed")
