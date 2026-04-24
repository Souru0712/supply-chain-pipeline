{% docs __overview__ %}

# Supply Chain Analytics Pipeline

Commodity price forecasting pipeline that ingests macroeconomic, market, and production data from federal APIs, transforms and validates it, then materializes feature-engineered datasets for XGBoost time-series models.

## Data Flow

```
  FRED API ─────► raw CSVs ─► stg_fred_ppi ──────────┐
  FRED API ─────► raw CSVs ─► stg_dollar_index ───────┤
  USDA MARS API ► raw CSVs ─► stg_ams_market ─────────┼──► mart models ──► Parquet ──► XGBoost + Dash
  NASS Export ──► raw TSV ──► stg_nass_production ─────┤
  product_master ► raw CSV ─► stg_product_master ──────┘
  fred_mapped ──► raw CSV ──► stg_fred_mapping ────────┘
```

## Layers

| Layer | Models | Materialization | Purpose |
|-------|--------|-----------------|---------|
| **Sources** | `raw_macroeconomic`, `raw_market`, `raw_production` | External CSV/TSV files | Raw data from API ingestion scripts |
| **Staging** | `stg_fred_ppi`, `stg_dollar_index`, `stg_ams_market`, `stg_nass_production`, `stg_product_master`, `stg_fred_mapping` | View | Schema normalization, type casting, string cleaning |
| **Marts** | `mart_dollar_index`, `mart_macroeconomic`, `mart_market_and_logistic`, `mart_nass` | External Parquet | Joins, forward rolling averages, percent changes |

## Data Sources

| Source | API | Refresh Frequency | Coverage |
|--------|-----|-------------------|----------|
| FRED PPI | Federal Reserve (FRED API) | Monthly | 2015 – present |
| Dollar Index | Federal Reserve (DTWEXBGS) | Daily (weekdays) | 2015 – present |
| USDA AMS | USDA MARS API (29 slugs) | Daily/weekly | ~2020 – present |
| USDA NASS | QuickStats bulk export | Periodic (annual crops) | 1972 – present |

## Ingredient Parameterization

Mart models filter by commodity name via the `ingredient_name` project variable (default: `CORN`).

```bash
# default — materializes CORN
dbt run --select marts

# override — materializes SOYBEAN
dbt run --select marts --vars '{ingredient_name: SOYBEAN}'
```

## Validation

- **Pre-ingestion:** Great Expectations validates API responses before writing to `data/raw/`
- **Post-ingestion:** dbt tests validate staging models (not_null, unique, no_lowercase, expression_is_true)

{% enddocs %}
