"""
ingest_ams.py

Ingests USDA AMS Market News report data from the MARS API.
Fetches Report Detail rows for configured slug_ids, writes each
report to a CSV matching the same structure as the manual downloads.

API: https://marsapi.ams.usda.gov/services/v1.2/reports/{slug_id}
Auth: Basic auth with MARS_API_KEY from .env

output_path = data/raw/market_and_logistic/AMS_{slug_id}/ReportDetail.csv
--------------------------------------------------------------
To add more reports, add entries to REPORT_SLUGS below.
The script will fetch all available history (back to ~2020) in one call.
Each slug_id writes to a single canonical folder (overwrites on re-run).
"""
import os
import logging
from re import M
import requests
import pandas as pd
from dotenv import load_dotenv

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
# Config
# ---------------------------------------------------------------------------
load_dotenv()
MARS_API_KEY = os.getenv("MARS_API_KEY")

if not MARS_API_KEY:
    raise ValueError("MARS_API_KEY not found in .env file.")

BASE_URL = "https://marsapi.ams.usda.gov/services/v1.2/reports"
OUTPUT_DIR = "data/raw/market_and_logistic"

# Add more slug_ids here to pull additional reports.
# Each entry: slug_id -> description (for logging only)
REPORT_SLUGS = {
    2960: "Arkansas Grain Bids",
    3146: "California Grain Bids",
    2912: "Colorado Grain Bids",
    3192: "Illinois Grain Bids",
    3463: "Indiana Grain Bids",
    2850: "Iowa Grain Bids",
    3043: "Iowa Barge Terminal Grain Bids",
    2886: "Kansas Grain Bids",
    2892: "Kentucky Grain Bids",
    3147: "Louisiana & Texas Export Bids",
    2714: "Maryland Grain Bids",
    3046: "Minneapolis Grain Report",
    2928: "Mississippi Grain Bids",
    2932: "Missouri Grain Bids",
    # Excluded: different schema (inspect manually before adding)
    # 3512: "National Mill-Feeds and Miscellaneous Report",  # 43 cols, different column names
    # 3617: "National Ethanol Report",                       # 48 cols, different column names
    # 3618: "National Weekly Grain Co-Products Report",      # 24 cols, value-based not price-based
    # 3802: "National Organic Grain and Feedstuff Report",   # 38 cols, different column names
    2920: "National Weekly Non-GMO Grain Report",
    3225: "Nebraska Grain Bids",
    3156: "North Carolina Grain Bids",
    3878: "North Dakota Grain Bids",
    2851: "Ohio Grain Bids",
    3100: "Oklahoma Grain Bids",
    3091: "Pennsylvania Grain Bids",
    3148: "Portland Grain Bids",
    2787: "South Carolina Grain Bids",
    3186: "South Dakota Grain Bids",
    3049: "Southern Minnesota Grain Bids",
    3088: "Tennessee Grain Bids",
    2711: "Texas Grain Bids",
    3167: "Virginia Grain Bids",
    3239: "Wyoming Grain Bids"
}


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------
def fetch_report_detail(slug_id: int) -> pd.DataFrame:
    """Fetch all Report Detail rows for a given slug_id from the MARS API."""
    url = f"{BASE_URL}/{slug_id}"
    params = {"allSections": "true"}

    logger.info(f"Fetching slug_id={slug_id} from {url}")
    print(f"  Fetching slug_id={slug_id} ...")

    resp = requests.get(
        url,
        params=params,
        auth=("", MARS_API_KEY),
        verify=False,
        timeout=120,
    )
    resp.raise_for_status()

    data = resp.json()

    # allSections=true returns a list of section objects
    if not isinstance(data, list):
        logger.error(f"Unexpected response format for slug {slug_id}: {type(data)}")
        print(f"  ERROR: unexpected response — {data.get('message', 'unknown')}")
        return pd.DataFrame()

    for section in data:
        if section.get("reportSection") == "Report Detail":
            stats = section.get("stats", {})
            rows = section.get("results", [])
            total = stats.get("totalRows", len(rows))
            returned = stats.get("returnedRows", len(rows))
            logger.info(f"  slug_id={slug_id}: {returned}/{total} rows returned")
            print(f"  Got {returned} rows (total available: {total})")

            if returned < total:
                logger.warning(
                    f"  slug_id={slug_id}: only {returned} of {total} rows returned. "
                    "API row limit reached."
                )
                print(f"  WARNING: API returned {returned}/{total} — some rows truncated")

            return pd.DataFrame(rows)

    logger.warning(f"No Report Detail section found for slug_id={slug_id}")
    print(f"  WARNING: no Report Detail section in response")
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------
def write_report(df: pd.DataFrame, slug_id: int) -> str:
    """Write the fetched report to CSV.

    Uses a single canonical folder per slug_id (AMS_{slug_id}/) so
    re-runs overwrite rather than creating duplicate timestamped copies.
    """
    folder = os.path.join(OUTPUT_DIR, f"AMS_{slug_id}")
    os.makedirs(folder, exist_ok=True)

    path = os.path.join(folder, "ReportDetail.csv")
    df.to_csv(path, index=False)
    logger.info(f"Wrote {len(df)} rows to {path}")
    print(f"  Saved to {path}")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    print(f"AMS Market News Ingestion — {len(REPORT_SLUGS)} report(s)\n")
    logger.info(f"Starting AMS ingestion for slugs: {list(REPORT_SLUGS.keys())}")

    for slug_id, description in REPORT_SLUGS.items():
        print(f"=== {description} (slug {slug_id}) ===")

        df = fetch_report_detail(slug_id)

        if df.empty:
            print(f"  No data returned, skipping.\n")
            continue

        # Show summary
        if "report_date" in df.columns:
            dates = pd.to_datetime(df["report_date"], format="%m/%d/%Y")
            print(f"  Date range: {dates.min().date()} to {dates.max().date()}")

        if "commodity" in df.columns:
            counts = df["commodity"].value_counts()
            print(f"  Commodities: {dict(counts)}")

        write_report(df, slug_id)
        print()

    print("Done.")
    logger.info("AMS ingestion complete")
