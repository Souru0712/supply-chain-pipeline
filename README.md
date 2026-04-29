# Supply Chain Intelligence Platform

End-to-end commodity price intelligence system for food manufacturers and procurement teams.  
Ingests data from USDA AMS, USDA NASS, and FRED, validates raw sources with Great Expectations **before** transformation, processes data through a dbt/DuckDB pipeline, trains **LightGBM quantile regression** models, and serves five decision-support tools through a **7-tab Dash dashboard**.

---

## What It Does

| Capability | What it answers |
|---|---|
| **Price Forecasting** | What will I pay for this commodity over the next 13 weeks? |
| **Margin & Cost Modeling** | What is my true all-in landed cost, and what happens under cost shocks? |
| **Supply Disruption Detection** | Is a supply crisis developing before the market has priced it in? |
| **Procurement Timing Optimization** | When should I buy, how much, and where are the opportunistic windows? |
| **Risk Quantification** | Where is my supply chain risk concentrated across four dimensions? |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                  Supply Chain Intelligence Platform                  │
└─────────────────────────────────────────────────────────────────────┘

  Data Sources               Ingestion
  ────────────               ─────────
  FRED API (PPI)  ────────► 2a.ingest_fred
  Dollar Index    ────────► 2b.ingest_di         data/raw/
  USDA AMS        ────────► 2c.ingest_ams    ──►  macroeconomic/
  NASS Exports    ────────► 1.create_product      market_and_logistic/
                                                       │
  Pre-Ingestion                                        ▼
  Validation      ◄──────────────────────── 3.ge_checkpoint.py
  ──────────────                              │  quarantine bad rows
                                              │  data/raw/quarantine/
                                              ▼  data/raw/validated/

  Transformation             Materialisation
  ──────────────             ───────────────
  4.transformed.py ────────► 5.materialized.py
  (PySpark)                  dbt mart models
                                  │
                                  ▼
                    data/materialized/CORN/training_weekly.parquet
                                  │
                    ┌─────────────┴──────────────────────────────┐
                    ▼                                            ▼
  Training    6.train_forecast.py                    Analytics scripts
  ────────    LightGBM 13 horizons × 3 quantiles     ──────────────────
              models/corn/lgb_h{1..13}.txt            8.price_forecast_v2.py
                    │                                 9.margin_model.py
  Inference         ▼                                 10.disruption_score.py
  ─────────   7.inference.py                          11.procurement_optimizer.py
              data/forecasts/CORN/                    12.risk_score.py
              weekly_forecast.{parquet,csv}                │
                    │                                      │
  Dashboard         └──────────────────┬──────────────────┘
  ─────────                            ▼
                               dashboard/app.py  (Dash · 7 tabs)
                               http://localhost:8050
```

---

## Model Design

| Property          | Value                                          |
|-------------------|------------------------------------------------|
| Algorithm         | LightGBM quantile regression (`objective: quantile`) |
| Horizons          | h = 1 … 13 weeks (direct multi-output)         |
| Quantiles         | p10 / p50 / p90                                |
| CV strategy       | 5-fold `TimeSeriesSplit`                        |
| Target            | `price_avg_weekly` (USD/bushel)                |
| Features          | 22 (price lags, rolling stats, macro, calendar) |
| Explainability    | SHAP feature attribution per horizon            |

---

## Metrics (LightGBM · CORN · 5-fold time-series CV)

| Horizon | MAE (p50) | RMSE (p50) | MAPE (p50) | Skill vs Naive |
|--------:|----------:|-----------:|-----------:|---------------:|
|  h = 1  |    0.0826 |     0.0999 |     1.62 % |      −136.53 % |
|  h = 2  |    0.0968 |     0.1149 |     1.90 % |       −39.21 % |
|  h = 3  |    0.1115 |     0.1319 |     2.20 % |        −7.69 % |
|  h = 4  |    0.1380 |     0.1649 |     2.72 % |        −0.58 % |
|  h = 5  |    0.1418 |     0.1694 |     2.79 % |       +14.54 % |
|  h = 6  |    0.1566 |     0.1820 |     3.07 % |       +20.50 % |
|  h = 7  |    0.1454 |     0.1739 |     2.83 % |       +35.90 % |
|  h = 8  |    0.1807 |     0.2132 |     3.50 % |       +29.37 % |
|  h = 9  |    0.1662 |     0.1936 |     3.24 % |       +41.39 % |
| h = 10  |    0.1815 |     0.2103 |     3.55 % |       +41.39 % |
| h = 11  |    0.1809 |     0.2091 |     3.55 % |       +45.17 % |
| h = 12  |    0.1817 |     0.2088 |     3.54 % |       +48.31 % |
| h = 13  |    0.1723 |     0.1944 |     3.38 % |       +53.69 % |

> Positive skill = beats the "last observed price" naive baseline. The model gains increasing skill from h=5 onward, reaching +54 % at 13 weeks.

---

## Repository Layout

```
supply-chain-pipeline/
├── scripts/
│   ├── 0.generate_sample_data.py      # synthetic data for dev/CI
│   ├── 1.create_product_master.py     # PySpark: ingredient reference
│   ├── 2a.ingest_fred.py              # FRED API → raw CSVs (PPI)
│   ├── 2b.ingest_dollar_index.py      # FRED API → dollar index
│   ├── 2c.ingest_ams.py               # USDA AMS market reports
│   ├── 3.ge_checkpoint.py             # Great Expectations: validate raw sources
│   ├── 4.transformed.py               # PySpark: raw → staged parquets
│   ├── 5.materialized.py              # DuckDB: materialise feature tables
│   ├── 6.train_forecast.py            # LightGBM 13-week quantile training
│   ├── 7.inference.py                 # load models → weekly_forecast.csv
│   ├── 8.price_forecast_v2.py         # Enhanced forecast: SHAP + directional accuracy
│   ├── 9.margin_model.py              # True landed cost stack + scenario analysis
│   ├── 10.disruption_score.py         # Composite disruption score (0–100)
│   ├── 11.procurement_optimizer.py    # LP-based 52-week purchase schedule
│   └── 12.risk_score.py              # Four-dimension risk quantification
├── dbt_pipeline/
│   ├── models/staging/                # 6 staging views (type-cast, clean)
│   └── models/marts/                  # 5 mart models → external parquets
├── dashboard/
│   ├── app.py                         # Dash: 7-tab dashboard
│   └── requirements.txt
├── models/corn/
│   ├── lgb_h{1..13}.txt              # p50 LightGBM boosters
│   ├── lgb_h{1..13}_q{10,90}.txt     # quantile boosters
│   └── metadata_v2.json              # features, metrics, SHAP summary path
├── tests/
│   ├── test_training.py               # training utility tests
│   ├── test_inference.py              # inference + dashboard tests
│   └── test_analytics.py             # scripts 9–12 unit tests
├── gx/                                # Great Expectations config
├── .github/workflows/ci.yml           # GitHub Actions: ruff + mypy + pytest
├── render.yaml                        # Render deployment config
├── fly.toml                           # Fly.io deployment config
├── Procfile                           # gunicorn start command
└── requirements.txt                   # full project dependencies
```

---

## Running the Full Pipeline (Real API Data)

### Step 1 — Prerequisites

```bash
git clone https://github.com/souru0712/supply-chain-pipeline.git
cd supply-chain-pipeline
pip install -r requirements.txt
pip install -r dashboard/requirements.txt
```

You need three API keys. All are free:

| Key | Where to get it | `.env` variable |
|-----|----------------|-----------------|
| FRED API key | [fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html) | `FRED_API_KEY` |
| USDA QuickStats key | [quickstats.nass.usda.gov/api](https://quickstats.nass.usda.gov/api) | `QUICKSTATS_KEY` |
| USDA AMS / MARS key | [mymarketnews.ams.usda.gov](https://mymarketnews.ams.usda.gov) | `MARS_KEY` |

```bash
cp .env.example .env
# edit .env and fill in the three keys above
```

### Step 2 — Ingest raw data

```bash
python scripts/1.create_product_master.py
python scripts/2a.ingest_fred.py        # PPI and macro series from FRED
python scripts/2b.ingest_dollar_index.py  # Dollar index from FRED
python scripts/2c.ingest_ams.py         # USDA AMS commodity prices
```

### Step 3 — Validate sources (pre-ingestion)

```bash
python scripts/3.ge_checkpoint.py
# Bad rows are quarantined to data/raw/quarantine/
# Clean data lands in data/raw/validated/
```

### Step 4 — Transform and materialise features

```bash
python scripts/4.transformed.py         # PySpark: raw → staged parquets
python scripts/5.materialized.py        # DuckDB: build feature tables

# Optional: run dbt mart models
cd dbt_pipeline
dbt run --select mart_corn_training_weekly
cd ..
```

### Step 5 — Train the forecast model

```bash
python scripts/6.train_forecast.py      # trains 13 horizons × 3 quantiles
# outputs: models/corn/lgb_h*.txt
```

### Step 6 — Run inference + analytics

```bash
python scripts/7.inference.py           # weekly price forecast
python scripts/8.price_forecast_v2.py   # SHAP attribution + directional accuracy
python scripts/9.margin_model.py        # landed cost stack + scenario waterfall
python scripts/10.disruption_score.py  # composite disruption score
python scripts/11.procurement_optimizer.py  # 52-week LP purchase schedule
python scripts/12.risk_score.py        # four-dimension risk score
```

All outputs land in `data/forecasts/CORN/`.

### Step 7 — Launch the dashboard

```bash
python dashboard/app.py
# Open http://localhost:8050
```

---

## Dashboard — 7 Tabs

| Tab | What you see | Business question |
|-----|-------------|-------------------|
| **📈 Price Forecast** | 13-week forward price line with p10–p90 confidence band; per-week detail table | What will I pay next quarter? |
| **⚠️ Risk Score** | Speedometer gauge (0–100); volatility, uncertainty, and momentum component bars | How exposed are we right now? |
| **🔮 SHAP Attribution** | Ranked feature importance bars; directional accuracy chart | Why is the model forecasting this? |
| **💰 Margin & Cost** | Stacked area chart (commodity + transport + energy + carry); cost-shock scenario bars | What is our true all-in cost? |
| **🚨 Disruption Alert** | Disruption gauge with four signal bars; 1-year trend with alert threshold lines | Is a supply crisis developing early? |
| **📅 Procurement Calendar** | 52-week price curve with opportunistic buy markers; recommended purchase bar chart | When and how much should we buy? |
| **🎯 Risk Dashboard** | Four-axis radar chart (supply/cost/logistics/demand); dimension time series | Where is risk concentrated? |

---

## Running Without API Keys (Synthetic Data)

```bash
python scripts/0.generate_sample_data.py
python scripts/6.train_forecast.py
python scripts/7.inference.py
python scripts/8.price_forecast_v2.py
python scripts/9.margin_model.py
python scripts/10.disruption_score.py
python scripts/11.procurement_optimizer.py
python scripts/12.risk_score.py
python dashboard/app.py
```

The dashboard falls back to synthetic demo data automatically for any tab whose output file is not yet present.

---

## Tests

```bash
pip install ruff mypy pytest lightgbm
ruff check scripts/ tests/ dashboard/
ruff format --check scripts/ tests/ dashboard/
mypy scripts/6.train_forecast.py scripts/7.inference.py --ignore-missing-imports
pytest tests/ -v
```

---

## CI — GitHub Actions

Every push triggers:

| Step | Tool | Scope |
|------|------|-------|
| Lint | `ruff` | `scripts/`, `tests/`, `dashboard/` |
| Format check | `ruff` | same |
| Type check | `mypy` | training + inference scripts |
| Unit tests | `pytest` | `tests/` |

---

## Deployment

### Render (recommended)

1. Push this repository to GitHub.
2. Create a new **Web Service** on [render.com](https://render.com) and connect the repo.
3. Render auto-detects `render.yaml` — no manual config needed.

### Fly.io

```bash
fly auth login
fly launch --config fly.toml
fly deploy
```

---

## Extending to Other Ingredients

```bash
python scripts/0.generate_sample_data.py --ingredient SOYBEAN
python scripts/6.train_forecast.py --ingredient SOYBEAN
python scripts/7.inference.py --ingredient SOYBEAN
python scripts/9.margin_model.py --ingredient SOYBEAN
python scripts/10.disruption_score.py --ingredient SOYBEAN
python scripts/11.procurement_optimizer.py --ingredient SOYBEAN
python scripts/12.risk_score.py --ingredient SOYBEAN
```

---

## MLflow Experiment Tracking

```bash
mlflow ui    # opens http://localhost:5000
```

Logs per-horizon MAE, RMSE, MAPE, baseline MAE, and skill-vs-naive for every training run.
