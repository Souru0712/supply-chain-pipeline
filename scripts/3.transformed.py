"""

transformed.py

Raw → Staged Transformations
--------------------------------------------------------------------------------------------------------
1. Structural / Schema

    Rename columns — snake_case, remove spaces, standardize abbreviations (e.g., Mkt Yr → market_year)
    Reorder columns — logical grouping (identifiers first, then measures, then metadata)
    Drop irrelevant columns — footnote columns, redundant duplicates, internal source artifacts
--------------------------------------------------------------------------------------------------------
2. Data Types

Cast strings to proper types — dates, integers, floats, booleans
Standardize date formats — everything to YYYY-MM-DD or ISO 8601
Normalize numeric precision — e.g., 4 decimal places for prices, 0 for counts
--------------------------------------------------------------------------------------------------------
3. String Cleaning

Trim whitespace — leading/trailing spaces are silent killers in joins
Normalize casing — UPPER, lower, or Title Case consistently per field
Remove special characters — non-printable chars, BOM markers, stray quotes
Standardize nulls — replace "N/A", "--", "(D)", "" with actual NULL
--------------------------------------------------------------------------------------------------------
4. Deduplication

Remove exact duplicates — same row ingested twice from the same source
Keep latest version — if a source re-publishes corrections, keep the most recent
--------------------------------------------------------------------------------------------------------
5. Validation & Flagging

Null checks — flag or reject rows missing required fields (e.g., no commodity name)
Range checks — flag implausible values (e.g., negative prices, future dates)
Referential checks — does the state_code exist in your state reference table?
Add a is_valid flag — rather than dropping bad rows, mark them so bad data is auditable
--------------------------------------------------------------------------------------------------------
6. Metadata / Lineage Columns

ingested_at — timestamp when the raw record was loaded
staged_at — timestamp of the staging transformation
source_file or source_url — where did this row come from?
source_row_id — original row number or API record ID for traceability
--------------------------------------------------------------------------------------------------------
7. Structural Normalization (if needed)

Unpivot / melt wide tables — USDA NASS loves wide year-column tables; these often need to go from (commodity, 2020, 2021, 2022) → (commodity, year, value)
Split concatenated fields — e.g., "Corn - Grain" → commodity = "Corn", class = "Grain"
Parse nested structures — flatten JSON blobs or delimited fields into proper columns
--------------------------------------------------------------------------------------------------------
Self Join

SELECT
    a.ingredient_id,
    a.ingredient_name,
    COUNT(*) AS duplicate_count
FROM read_csv('/mnt/c/Users/oscar/.vscode/supply-chain-pipeline/data/raw/macroeconomic/fred_mappings.csv') a
JOIN read_csv('/mnt/c/Users/oscar/.vscode/supply-chain-pipeline/data/raw/macroeconomic/fred_mappings.csv') b
    ON a.ingredient_name = b.ingredient_name
    AND a.ingredient_id <> b.ingredient_id
GROUP BY a.ingredient_id, a.ingredient_name
ORDER BY duplicate_count DESC;
--------------------------------------------------------------------------------------------------------
"""

# =========================================================================================================
# macroeconomic
# =========================================================================================================
import argparse
import logging
import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    upper,
    round as spark_round,
    regexp_replace,
    to_date,
    when,
    lit,
    trim,
)
from pyspark.sql.types import IntegerType, FloatType, DateType, StringType

# Parse args before Spark starts (Spark reads sys.argv and chokes on --only)
VALID_SOURCES = ["macroeconomic", "market_and_logistic", "nass", "fred_mapping", "all"]
_parser = argparse.ArgumentParser(description="Raw → Staged transformations")
_parser.add_argument(
    "--only",
    choices=VALID_SOURCES,
    default="all",
    help="Transform only a specific source (default: all)",
)
_args, _remaining = _parser.parse_known_args()
sys.argv = [sys.argv[0]] + _remaining  # pass only unrecognised args to Spark

os.makedirs("logs", exist_ok=True)
logger = logging.getLogger("transformed")
logger.setLevel(logging.INFO)
fh = logging.FileHandler("logs/transformed.log")
fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s  %(message)s"))
logger.addHandler(fh)

spark = (
    SparkSession.builder.appName("supply-chain-pipeline")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.shuffle.partitions", "64")
    .getOrCreate()
)


def transform_fred(spark):
    FRED_API_FILE = "data/raw/macroeconomic"

    fred_raw = (
        spark.read.option("header", True).option("inferSchema", True).csv(FRED_API_FILE)
    )

    fred_df = (
        fred_raw.select("ingredient_id", "ppi", "date", "frequency")
        .withColumnRenamed("date", "ppi_date")
        .withColumnRenamed("frequency", "ppi_frequency")
        .withColumn("ingredient_id", col("ingredient_id").cast(IntegerType()))
        .withColumn("ppi", col("ppi").cast(FloatType()))
        .withColumn("ppi_date", col("ppi_date").cast(DateType()))
        .withColumn("ppi_frequency", col("ppi_frequency").cast(StringType()))
    )

    logger.info("transform_fred completed — fred_api_df ready")
    return fred_df


def write_fred(fred_api_df):
    fred_api_df.write.mode("overwrite").parquet("data/staged/macroeconomic")
    logger.info("fred_api_df written to data/staged/macroeconomic")


# =========================================================================================================
# market & logistic
# =========================================================================================================
def transform_ams(spark):
    import glob as globmod

    AMS_DIR = os.path.join(os.getcwd(), "data", "raw", "market_and_logistic")
    ams_files = sorted(globmod.glob(os.path.join(AMS_DIR, "AMS_*", "ReportDetail.csv")))
    logger.info(f"Found {len(ams_files)} AMS CSV files")

    ams_raw = (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .option("quote", '"')
        .option("escape", '"')
        .csv(ams_files)
    )

    ams_df = (
        ams_raw.select(
            "slug_id",
            "report_title",
            "commodity",
            "cat",
            "grade",
            "`price Min`",
            "`price Max`",
            "avg_price",
            "price_unit",
            "`sale Type`",
            "delivery_point",
            "freight",
            "trans_mode",
            "market_location_state",
            "report_date",
        )
        # renames
        .withColumnRenamed("commodity", "ams_ingredient_name")
        .withColumnRenamed("cat", "ams_ingredient_group")
        .withColumnRenamed("grade", "ams_ingredient_grade")
        .withColumnRenamed("price Min", "price_min")
        .withColumnRenamed("price Max", "price_max")
        .withColumnRenamed("avg_price", "price_avg")
        .withColumnRenamed("sale Type", "sale_type")
        # cast slug_id
        .withColumn("slug_id", col("slug_id").cast(IntegerType()))
        # string columns → uppercased
        .withColumn("report_title", upper(col("report_title")))
        .withColumn("ams_ingredient_name", upper(col("ams_ingredient_name")))
        .withColumn("ams_ingredient_group", upper(col("ams_ingredient_group")))
        .withColumn("ams_ingredient_grade", upper(col("ams_ingredient_grade")))
        .withColumn("price_unit", upper(col("price_unit")))
        .withColumn("sale_type", upper(col("sale_type")))
        .withColumn("delivery_point", upper(col("delivery_point")))
        .withColumn("freight", upper(col("freight")))
        .withColumn("trans_mode", upper(col("trans_mode")))
        .withColumn("market_location_state", upper(col("market_location_state")))
        # Normalize AMS commodity names to match fred_mapping
        .withColumn(
            "ams_ingredient_name",
            regexp_replace(col("ams_ingredient_name"), "^SOYBEANS$", "SOYBEAN"),
        )
        .withColumn(
            "ams_ingredient_name",
            regexp_replace(
                col("ams_ingredient_name"), "^SUNFLOWER SEEDS$", "SUNFLOWER"
            ),
        )
        # WHITE OATS has no fred_mapping match — left as-is
        # price columns → NULL if empty, else float rounded to 4 decimals
        .withColumn(
            "price_min",
            when(trim(col("price_min")) == "", lit(None).cast(FloatType())).otherwise(
                spark_round(col("price_min").cast(FloatType()), 4)
            ),
        )
        .withColumn(
            "price_max",
            when(trim(col("price_max")) == "", lit(None).cast(FloatType())).otherwise(
                spark_round(col("price_max").cast(FloatType()), 4)
            ),
        )
        .withColumn(
            "price_avg",
            when(trim(col("price_avg")) == "", lit(None).cast(FloatType())).otherwise(
                spark_round(col("price_avg").cast(FloatType()), 4)
            ),
        )
        # report_date → date type (YYYY-MM-DD)
        .withColumn(
            "report_date",
            to_date(
                regexp_replace(
                    col("report_date"), r"(\d{2})/(\d{2})/(\d{4})", "$3-$1-$2"
                ),
                "yyyy-MM-dd",
            ),
        )
    )

    logger.info("transform_ams completed — ams_df ready")
    return ams_df


def write_ams(ams_df):
    ams_df.write.mode("overwrite").parquet("data/staged/market_and_logistic")
    logger.info("ams_df written to data/staged/market_and_logistic")


# =========================================================================================================
# production
# =========================================================================================================
def transform_nass(spark):
    NASS_FILE = "data/raw/production/qs.crops_20260327.txt"
    NASS_COLUMNS = [
        "CLASS_DESC",
        "UNIT_DESC",
        "VALUE",
        "CV_%",
        "YEAR",
        "FREQ_DESC",
        "REFERENCE_PERIOD_DESC",
        "LOAD_TIME",
        "STATE_ALPHA",
        "COUNTRY_NAME",
    ]

    nass_raw = (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .option("delimiter", "\t")
        .csv(NASS_FILE)
    )

    nass_df = (
        nass_raw.select([col(c).alias(c.lower()) for c in NASS_COLUMNS])
        # renames
        .withColumnRenamed("class_desc", "ingredient_name")
        .withColumnRenamed("unit_desc", "unit_of_measure")
        .withColumnRenamed("value", "amount")
        .withColumnRenamed("freq_desc", "frequency")
        .withColumnRenamed("reference_period_desc", "range")
        .withColumnRenamed("state_alpha", "state")
        .withColumnRenamed("country_name", "country")
        # string columns
        .withColumn("ingredient_name", col("ingredient_name").cast(StringType()))
        .withColumn("unit_of_measure", col("unit_of_measure").cast(StringType()))
        .withColumn("frequency", col("frequency").cast(StringType()))
        .withColumn("range", col("range").cast(StringType()))
        .withColumn("state", col("state").cast(StringType()))
        .withColumn("country", col("country").cast(StringType()))
        # float columns — strip commas/whitespace, cast numeric values, else NULL
        .withColumn("amount", regexp_replace(trim(col("amount")), ",", ""))
        .withColumn(
            "amount",
            when(col("amount").rlike("^-?\\d"), col("amount").cast(FloatType())),
        )
        .withColumn("cv_%", regexp_replace(trim(col("cv_%")), ",", ""))
        .withColumn(
            "cv_%", when(col("cv_%").rlike("^-?\\d"), col("cv_%").cast(FloatType()))
        )
        # int
        .withColumn("year", col("year").cast(IntegerType()))
        # date (strip timestamp)
        .withColumn("load_time", col("load_time").cast(DateType()))
    )

    logger.info("transform_nass completed — nass_df ready")
    return nass_df


def transform_fred_mapping(spark):
    FRED_MAPPING_FILE = "data/raw/production/fred_mapped.csv"

    fred_map_raw = (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .csv(FRED_MAPPING_FILE)
    )

    fred_map_df = (
        fred_map_raw.withColumn(
            "ingredient_id", col("ingredient_id").cast(IntegerType())
        )
        .withColumn("ingredient_group", col("ingredient_group").cast(StringType()))
        .withColumn(
            "ingredient_description", col("ingredient_description").cast(StringType())
        )
        .withColumn("ingredient_name", col("ingredient_name").cast(StringType()))
        .withColumn("unit_of_measure", col("unit_of_measure").cast(StringType()))
        .withColumn(
            "fred_series_id",
            when(
                upper(col("fred_series_id")) == "no ppi available",
                lit(None).cast(StringType()),
            ).otherwise(col("fred_series_id").cast(StringType())),
        )
    )

    logger.info("transform_fred_mapping completed — fred_map_df ready")
    return fred_map_df


def write_production(nass_df, fred_map_df):
    nass_df.write.mode("overwrite").parquet("data/staged/production/nass")
    logger.info("nass_df written to data/staged/production/nass")
    fred_map_df.write.mode("overwrite").parquet("data/staged/production/fred_mapping")
    logger.info("fred_map_df written to data/staged/production/fred_mapping")


# =========================================================================================================
# main
# =========================================================================================================
VALID_SOURCES = ["macroeconomic", "market_and_logistic", "nass", "fred_mapping", "all"]

if __name__ == "__main__":
    target = _args.only

    try:
        logger.info(f"pipeline started (target={target})")

        if target in ("macroeconomic", "all"):
            fred_api_df = transform_fred(spark)
            write_fred(fred_api_df)

        if target in ("market_and_logistic", "all"):
            ams_df = transform_ams(spark)
            write_ams(ams_df)

        if target in ("nass", "fred_mapping", "all"):
            if target in ("nass", "all"):
                nass_df = transform_nass(spark)
            if target in ("fred_mapping", "all"):
                fred_map_df = transform_fred_mapping(spark)
            # write whichever were transformed
            if target == "all":
                write_production(nass_df, fred_map_df)
            elif target == "nass":
                nass_df.write.mode("overwrite").parquet("data/staged/production/nass")
                logger.info("nass_df written to data/staged/production/nass")
            elif target == "fred_mapping":
                fred_map_df.write.mode("overwrite").parquet(
                    "data/staged/production/fred_mapping"
                )
                logger.info(
                    "fred_map_df written to data/staged/production/fred_mapping"
                )

        logger.info("pipeline completed successfully")
    except Exception as e:
        logger.error(f"pipeline failed: {e}", exc_info=True)
        raise
    finally:
        spark.stop()
        logger.info("spark session closed")
