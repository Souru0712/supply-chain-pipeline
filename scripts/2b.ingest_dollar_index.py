"""
ingest_dollar_index.py

Ingests the Trade Weighted U.S. Dollar Index (DTWEXBGS) from the FRED API.
Fetches observations from 2015 onward, saves a raw CSV, then uses Spark to
cast types and write staged parquet.

output (raw):    data/raw/macroeconomic/dollar_index_{timestamp}.csv
output (staged): data/staged/dollar_index/
"""

import os
import logging
import time
import requests
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import FloatType, DateType, StringType

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename="logs/ingestion.log",
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FRED API config
# ---------------------------------------------------------------------------
load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY")

if not FRED_API_KEY:
    raise ValueError("FRED_API_KEY not found in .env file.")

FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
FRED_SERIES_URL = "https://api.stlouisfed.org/fred/series"
REQUEST_TIMEOUT = 10
SERIES_ID = "DTWEXBGS"

_FREQ_MAP = {
    "A":  "annual",
    "SA": "semiannual",
    "Q":  "quarterly",
    "M":  "monthly",
    "BW": "biweekly",
    "W":  "weekly",
    "D":  "daily",
}


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------
def _request_with_retry(url: str, params: dict, max_retries: int = 5) -> requests.Response:
    for attempt in range(max_retries):
        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        if resp.status_code in (429, 400):
            wait = 2 ** attempt
            print(f"    -> HTTP {resp.status_code} -- backing off {wait}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp
    resp.raise_for_status()
    return resp


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------
def fetch_dollar_index() -> pd.DataFrame:
    """Fetch DTWEXBGS observations from 2015-01-01 onward."""
    obs_params = {
        "series_id":         SERIES_ID,
        "api_key":           FRED_API_KEY,
        "file_type":         "json",
        "observation_start": "2015-01-01",
    }
    info_params = {
        "series_id": SERIES_ID,
        "api_key":   FRED_API_KEY,
        "file_type": "json",
    }

    obs_resp = _request_with_retry(FRED_BASE_URL, obs_params)
    info_resp = _request_with_retry(FRED_SERIES_URL, info_params)

    # parse frequency
    serieses = info_resp.json().get("seriess", [])
    freq_short = serieses[0].get("frequency_short", "") if serieses else ""
    frequency = _FREQ_MAP.get(freq_short, freq_short.lower() or "unknown")

    # parse observations
    observations = obs_resp.json().get("observations", [])
    if not observations:
        raise RuntimeError(f"Series {SERIES_ID}: no observations returned.")

    df = pd.DataFrame(observations)[["date", "value"]]
    df.columns = ["ppi_date", "ppi"]

    # drop FRED null sentinel "."
    df = df[df["ppi"] != "."].copy()
    df["ppi"] = df["ppi"].astype(float)

    if df.empty:
        raise RuntimeError(f"Series {SERIES_ID}: zero usable observations after filtering.")

    df["fred_series_id"] = SERIES_ID
    df["name"] = "dollar_index"
    df["ppi_frequency"] = frequency

    # reorder columns
    df = df[["fred_series_id", "name", "ppi", "ppi_date", "ppi_frequency"]]
    return df


# ---------------------------------------------------------------------------
# Spark transform & write
# ---------------------------------------------------------------------------
spark = (
    SparkSession.builder
    .appName("supply-chain-pipeline")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.shuffle.partitions", "64")
    .getOrCreate()
)


def transform_and_write(csv_path: str):
    """Read the raw CSV with Spark, cast types, and write staged parquet."""
    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(csv_path)
    )

    df = (
        df
        .withColumn("fred_series_id", col("fred_series_id").cast(StringType()))
        .withColumn("ppi", col("ppi").cast(FloatType()))
        .withColumn("ppi_date", col("ppi_date").cast(DateType()))
        .withColumn("ppi_frequency", col("ppi_frequency").cast(StringType()))
    )

    os.makedirs("data/staged/dollar_index", exist_ok=True)
    df.write.mode("overwrite").parquet("data/staged/dollar_index")
    logger.info("dollar_index parquet written to data/staged/dollar_index")
    print(f"Staged {df.count()} rows -> data/staged/dollar_index/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        logger.info("dollar_index ingestion started")
        print(f"Fetching {SERIES_ID} from FRED ...")

        df = fetch_dollar_index()
        print(f"  {len(df)} observations fetched.")

        # save raw CSV
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        csv_path = f"data/raw/macroeconomic/dollar_index_{timestamp}.csv"
        os.makedirs("data/raw/macroeconomic", exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"Wrote {len(df)} rows -> {csv_path}")
        logger.info("FILE_WRITTEN  source=FRED_dollar_index  rows=%d  path=%s", len(df), csv_path)

        # spark transform & staged parquet
        transform_and_write(csv_path)

        logger.info("dollar_index ingestion completed")
    except Exception as e:
        logger.error(f"dollar_index ingestion failed: {e}", exc_info=True)
        raise
    finally:
        spark.stop()
