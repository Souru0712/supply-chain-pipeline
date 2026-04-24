"""
create_product_master.py

Reads NASS export files (qs.crops_20260327.txt and qs.animals_products_20260327.txt)
using PySpark to load, normalise, and deduplicate.

Extracts unique combinations of COMMODITY_DESC, CLASS_DESC, GROUP_DESC, and UNIT_DESC,
applies a normalisation pipeline to each field, and writes a deduplicated
product_master.csv with columns:
    - ingredient_id        (auto-incremented integer, starts at 10001)
    - ingredient_group  (from GROUP_DESC)
    - ingredient_description      (from COMMODITY_DESC)
    - ingredient_name     (from CLASS_DESC)
    - unit_of_measure      (from UNIT_DESC, canonical singular form)

Normalisation pipeline (applied in this order per field):
    1. Whitespace collapse   — strips outer whitespace + collapses internal runs
    2. Uppercasing           — for consistent comparison
    3. Null-sentinel check   — discards placeholder strings like "(D)", "(NA)", etc.
    4. Punctuation standard  — commas/dashes/periods → single space (name & ingredient_group)
    5. Plural normalisation  — maps plural unit forms to their canonical singular
    6. Allowlist filter      — only whitelisted unit_of_measure values pass through

Claude AI was used to fetch series_id's from FRED to closely match commodity_names, their categories, and unit_of_measurements

"""

import logging
import os
import re
import sys
from functools import reduce

from pyspark.sql import SparkSession, functions as F, Window
from pyspark.sql.types import StringType

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/create_product_master.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SOURCE_FILES = [
    "data/raw/production/qs.crops_20260327.txt",
    "data/raw/production/qs.animals_products_20260327.txt",
]

OUTPUT_FILE = "data/raw/production/product_master.csv"

SOURCE_COLUMNS = {
    "ingredient_group": "GROUP_DESC",
    "ingredient_description": "COMMODITY_DESC",
    "ingredient_name": "CLASS_DESC",
    "unit_of_measure": "UNIT_DESC",
}

OUTPUT_COLUMNS = [
    "ingredient_id",
    "ingredient_group",
    "ingredient_description",
    "ingredient_name",
    "unit_of_measure",
]

# ingredient_id sequence start
ID_START = 10001

# ---------------------------------------------------------------------------
# 1. Null-sentinel values
# Any key field that resolves to one of these strings (after whitespace
# collapse + uppercasing) is treated as missing and the row is discarded.
# ---------------------------------------------------------------------------
NULL_SENTINELS: frozenset[str] = frozenset(
    {
        "(D)",  # Withheld to avoid disclosing individual operation data
        "(NA)",  # Not available
        "(Z)",  # Less than half the unit shown
        "NOT SPECIFIED",
        "NOT AVAILABLE",
        "N/A",
        "NA",
        "TOTAL",  # Aggregate roll-up rows, not a real ingredient
    }
)

# ---------------------------------------------------------------------------
# 2. Punctuation normalisation regex
# Matches one or more of: comma, hyphen, en-dash, em-dash, period.
# Replaced with a single space, then the result is whitespace-collapsed again.
# Applied to ingredient_description and ingredient_group ONLY — NOT to unit_of_measure,
# which uses slashes intentionally (e.g. "$ / LB").
# ---------------------------------------------------------------------------
_PUNCT_RE = re.compile(r"[\-\u2013\u2014,.]+")

# ---------------------------------------------------------------------------
# 3. Plural → singular unit mapping
# Keys are raw NASS spellings (uppercased + whitespace-collapsed).
# Values are the canonical singular form stored in product_master.csv.
# The lookup is applied before the allowlist check so only canonical
# forms need to appear in ALLOWED_UNITS.
# ---------------------------------------------------------------------------
UNIT_PLURAL_MAP: dict[str, str] = {
    "TONS": "TON",
    "BALES": "BALE",
    "BUSHELS": "BU",
    "GALLONS": "GALLON",
    "BARRELS": "BARREL",
    "DOZENS": "DOZEN",
}

# ---------------------------------------------------------------------------
# 4. Unit-of-measure allowlist
# Checked AFTER plural normalisation, so only canonical singular forms
# need to be listed here.
# ---------------------------------------------------------------------------
ALLOWED_UNITS: frozenset[str] = frozenset(
    {
        # Mass & Weight
        "LB",
        "CWT",
        "TON",
        "TONS, DRY BASIS",
        "TONS, FRESH BASIS",
        # Volume & Bulk
        "BU",
        "GALLON",
        "BARREL",
        "BALE",
        "480 LB BALES",
        # Quantity & Count
        "HEAD",
        "DOZEN",
        "THOUSANDS",
        "INDEX",
        # Dollar-denominated units
        "$ / LB",
        "$ / CWT",
        "$ / TON",
        "$ / BU",
        "$ / HEAD",
        "$ / DOZEN",
        "$ / GALLON",
    }
)


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------


def _collapse_whitespace(text: str) -> str:
    """Strip outer whitespace and collapse internal runs (including tabs,
    non-breaking spaces U+00A0, etc.) to a single ASCII space.

    " ".join(text.split()) handles all Unicode whitespace variants because
    str.split() with no argument splits on any whitespace character.
    """
    return " ".join(text.split())


def _normalize_text(text: str, *, strip_punct: bool = False) -> str:
    """Full normalisation pipeline for a single field value.

    Steps:
        1. Whitespace collapse
        2. Uppercase
        3. Optional punctuation standardisation (name / ingredient_group fields only)
        4. Second whitespace collapse (punct replacement may introduce spaces)

    Returns the normalised string, or "" if the result is a null sentinel.
    The caller is responsible for treating an empty return as a skip signal.
    """
    # Step 1 & 2: collapse + uppercase
    normalised = _collapse_whitespace(text).upper()

    # Step 3: punctuation (skip for unit_of_measure to preserve "$ / LB" etc.)
    if strip_punct:
        normalised = _PUNCT_RE.sub(" ", normalised)
        normalised = _collapse_whitespace(normalised)  # Step 4

    # Null-sentinel check — return empty string so caller can skip the row
    if normalised in NULL_SENTINELS:
        return ""

    return normalised


def _normalize_unit(raw: str) -> str:
    """Normalise a UNIT_DESC value and apply the plural → singular mapping.

    Returns "" if the value is empty or a null sentinel.
    """
    normalised = _normalize_text(raw, strip_punct=False)
    if not normalised:
        return ""

    # Plural → singular lookup (e.g. "TONS" → "TON", "BALES" → "BALE")
    return UNIT_PLURAL_MAP.get(normalised, normalised)


# ---------------------------------------------------------------------------
# PySpark UDF wrappers
# ---------------------------------------------------------------------------


def _normalize_text_spark(text):
    """UDF wrapper: normalise a text field with punctuation stripping."""
    return _normalize_text(text or "", strip_punct=True)


def _normalize_unit_spark(text):
    """UDF wrapper: normalise a unit field with plural mapping."""
    return _normalize_unit(text or "")


normalize_text_udf = F.udf(_normalize_text_spark, StringType())
normalize_unit_udf = F.udf(_normalize_unit_spark, StringType())


# ---------------------------------------------------------------------------
# Delimiter detection
# ---------------------------------------------------------------------------


def detect_delimiter(filepath: str) -> str:
    """Sniff the delimiter from the first line of the file."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            first_line = f.readline()
    except OSError as exc:
        logger.error("Could not open '%s' to detect delimiter: %s", filepath, exc)
        raise

    if first_line.count("\t") > first_line.count(",") and first_line.count(
        "\t"
    ) > first_line.count("|"):
        detected = "\t"
    elif first_line.count("|") > first_line.count(","):
        detected = "|"
    else:
        detected = ","

    logger.debug("Detected delimiter %r for file '%s'", detected, filepath)
    return detected


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logger.info("=" * 50)
    logger.info("NASS Product Master Builder — starting (Spark mode)")
    logger.info("=" * 50)

    spark = SparkSession.builder.appName("CreateProductMaster").getOrCreate()

    # ------------------------------------------------------------------
    # Read all source files and union into a single DataFrame
    # ------------------------------------------------------------------
    dfs = []
    for filepath in SOURCE_FILES:
        if not os.path.exists(filepath):
            logger.warning("Source file not found, skipping: '%s'", filepath)
            continue

        delimiter = detect_delimiter(filepath)
        df = spark.read.csv(filepath, header=True, sep=delimiter, inferSchema=True)
        dfs.append(df)
        logger.info("Loaded '%s'", filepath)

    if not dfs:
        logger.error(
            "No source files could be loaded. "
            "Ensure the .txt files exist and are accessible."
        )
        spark.stop()
        sys.exit(1)

    combined = reduce(lambda a, b: a.unionByName(b, allowMissingColumns=True), dfs)

    # ------------------------------------------------------------------
    # Select SOURCE_COLUMNS values and alias to keys
    # ------------------------------------------------------------------
    df = combined.select(
        [F.col(src).alias(alias) for alias, src in SOURCE_COLUMNS.items()]
    )

    # ------------------------------------------------------------------
    # Apply normalisation UDFs
    # ------------------------------------------------------------------
    df = (
        df.withColumn(
            "ingredient_description", normalize_text_udf("ingredient_description")
        )
        .withColumn("ingredient_name", normalize_text_udf("ingredient_name"))
        .withColumn("ingredient_group", normalize_text_udf("ingredient_group"))
        .withColumn("unit_of_measure", normalize_unit_udf("unit_of_measure"))
    )

    # ------------------------------------------------------------------
    # Filter out empty/sentinel fields and non-allowed units
    # ------------------------------------------------------------------
    df = df.filter(
        (F.col("ingredient_description") != "")
        & (F.col("ingredient_group") != "")
        & (F.col("unit_of_measure") != "")
        & F.col("unit_of_measure").isin(list(ALLOWED_UNITS))
    )

    # ------------------------------------------------------------------
    # Distinct on the four ingredient columns
    # ------------------------------------------------------------------
    df = df.dropDuplicates(
        [
            "ingredient_description",
            "ingredient_name",
            "ingredient_group",
            "unit_of_measure",
        ]
    )

    # ------------------------------------------------------------------
    # Add ingredient_id starting at ID_START (10001)
    # ------------------------------------------------------------------
    w = Window.orderBy(
        "ingredient_group",
        "ingredient_description",
        "ingredient_name",
        "unit_of_measure",
    )
    df = df.withColumn(
        "ingredient_id", F.row_number().over(w) + F.lit(ID_START - 1)
    ).select(OUTPUT_COLUMNS)

    # ------------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------------
    row_count = df.count()
    df.coalesce(1).write.option("header", True).mode("overwrite").csv(OUTPUT_FILE)

    logger.info("Successfully wrote %d row(s) to '%s'.", row_count, OUTPUT_FILE)
    logger.info("=" * 50)

    spark.stop()


if __name__ == "__main__":
    main()
