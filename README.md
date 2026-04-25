# Supply Chain Price Forecasting Pipeline

End-to-end commodity price forecasting system for corn (and extensible to any ingredient).  
Ingests macroeconomic and market data from federal APIs, validates and transforms it through a dbt/DuckDB pipeline, trains **XGBoost quantile regression** models across 13 weekly horizons, and serves predictions through a **Dash dashboard** with a composite risk score.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                  Supply Chain Forecasting Pipeline                   │
└─────────────────────────────────────────────────────────────────────┘

  Data Sources               Ingestion              Staging
  ────────────               ─────────              ───────
  FRED API (PPI)  ────────► 2a.ingest_fred    ───► stg_fred_ppi
  Dollar Index    ────────► 2b.ingest_di      ───► stg_dollar_index
  USDA AMS        ────────► 2c.ingest_ams     ───► stg_ams_market
  NASS Exports    ────────► 1.create_product  ───► stg_product_master

  Validation                 Materialisation
  ──────────                 ───────────────
  4.ge_checkpoint ─────────► quarantine /        5.materialized.py
                              validated parquets  dbt mart models
                                                       │
                                                       ▼
                              data/materialized/CORN/training_weekly.parquet
                                                       │
                              ┌────────────────────────┘
                              ▼
  Training                   6.train_forecast.py  (XGBoost, 13 horizons × 3 quantiles)
  ────────                        │
                            models/corn/xgb_h{1..13}.json
                                  │
  Inference                       ▼
  ─────────                  7.inference.py
                                  │
                            data/forecasts/CORN/weekly_forecast.{parquet,csv}
                                  │
  Dashboard                       ▼
  ─────────                  dashboard/app.py  (Dash + Plotly)
                             ├── 13-week forecast fan chart
                             └── Composite risk score gauge
                                  │
                                  ▼
                        https://supply-chain-dashboard.onrender.com
```

---

## Model Design

| Property          | Value                                         |
|-------------------|-----------------------------------------------|
| Algorithm         | XGBoost `reg:quantileerror`                   |
| Horizons          | h = 1 … 13 weeks (direct multi-output)        |
| Quantiles         | p10 / p50 / p90                               |
| CV strategy       | 5-fold `TimeSeriesSplit`                       |
| Target            | `price_avg_weekly` (USD/bushel)               |
| Features          | 22 (price lags, rolling stats, macro, calendar)|
| Training rows     | 310 weeks (2020-01-06 → 2025-12-08)           |

---

## Metrics (XGBoost · CORN · 5-fold time-series CV)

| Horizon | Rows | MAE (p50) | RMSE (p50) | MAPE (p50) | Skill vs Naive |
|--------:|-----:|----------:|-----------:|-----------:|---------------:|
|  h = 1  |  257 |    0.0785 |     0.0989 |     1.53 % |      −124.85 % |
|  h = 2  |  256 |    0.1036 |     0.1264 |     2.05 % |       −49.01 % |
|  h = 3  |  255 |    0.1260 |     0.1476 |     2.48 % |       −21.70 % |
|  h = 4  |  254 |    0.1475 |     0.1735 |     2.91 % |        −7.45 % |
|  h = 5  |  253 |    0.1495 |     0.1764 |     2.94 % |        +9.85 % |
|  h = 6  |  252 |    0.1591 |     0.1872 |     3.12 % |       +19.19 % |
|  h = 7  |  251 |    0.1538 |     0.1815 |     3.00 % |       +32.20 % |
|  h = 8  |  250 |    0.1706 |     0.1987 |     3.33 % |       +33.30 % |
|  h = 9  |  249 |    0.1818 |     0.2131 |     3.55 % |       +35.92 % |
| h = 10  |  248 |    0.1886 |     0.2166 |     3.69 % |       +39.09 % |
| h = 11  |  247 |    0.1929 |     0.2232 |     3.78 % |       +41.52 % |
| h = 12  |  246 |    0.1908 |     0.2176 |     3.73 % |       +45.72 % |
| h = 13  |  245 |    0.1826 |     0.2051 |     3.59 % |       +50.90 % |

> **Skill vs Naive**: positive = beats the "last observed price" baseline.  
> Short horizons underperform the naive baseline (price is very sticky at h=1–4).  
> The model gains increasing skill from h=5 onward, reaching +51 % at 13 weeks.

---

## Repository Layout

```
supply-chain-pipeline/
├── scripts/
│   ├── 0.generate_sample_data.py   # synthetic training data for dev/CI
│   ├── 1.create_product_master.py  # PySpark: build ingredient reference
│   ├── 2a.ingest_fred.py           # FRED API → raw CSVs (PPI)
│   ├── 2b.ingest_dollar_index.py   # FRED API → dollar index
│   ├── 2c.ingest_ams.py            # USDA AMS market reports
│   ├── 3.transformed.py            # PySpark: raw → staged parquets
│   ├── 4.ge_checkpoint.py          # Great Expectations validation
│   ├── 5.materialized.py           # DuckDB: materialise feature tables
│   ├── 6.train_forecast.py         # XGBoost 13-week quantile training ★
│   └── 7.inference.py              # load models → weekly_forecast.csv  ★
├── dbt_pipeline/
│   ├── models/staging/             # 6 staging views (type-cast, clean)
│   └── models/marts/               # 5 mart models → external parquets
├── dashboard/
│   ├── app.py                      # Dash: forecast + risk score tabs    ★
│   └── requirements.txt            # minimal dashboard deps
├── models/corn/
│   ├── xgb_h{1..13}.json          # point (p50) XGBoost boosters
│   ├── xgb_h{1..13}_q{10,90}.json # quantile boosters
│   └── metadata.json              # features, metrics, train_end_date
├── tests/
│   ├── test_training.py            # training utility unit tests
│   └── test_inference.py           # inference + dashboard unit tests
├── gx/                             # Great Expectations config
├── .github/workflows/ci.yml        # GitHub Actions: ruff + mypy + pytest
├── render.yaml                     # Render deployment config
├── fly.toml                        # Fly.io deployment config
├── Procfile                        # gunicorn start command
└── requirements.txt                # full project dependencies
```

---

## Reproduction

### Prerequisites

```bash
# Python 3.11+
git clone https://github.com/souru0712/supply-chain-pipeline.git
cd supply-chain-pipeline
pip install -r requirements.txt
```

### Option A — Synthetic data (no API keys required)

```bash
# 1. Generate synthetic training data
python scripts/0.generate_sample_data.py

# 2. Train XGBoost models (≈ 3–5 min)
python scripts/6.train_forecast.py

# 3. Run inference
python scripts/7.inference.py

# 4. Start dashboard
python dashboard/app.py       # http://localhost:8050
```

### Option B — Real data (requires API keys)

Copy `.env.example` → `.env` and fill in your keys, then run scripts 1–5 in order before step 6.

```bash
cp .env.example .env          # fill FRED_API_KEY, QUICKSTATS_KEY, MARS_KEY
python scripts/1.create_product_master.py
python scripts/2a.ingest_fred.py
python scripts/2b.ingest_dollar_index.py
python scripts/2c.ingest_ams.py
python scripts/3.transformed.py
python scripts/4.ge_checkpoint.py
python scripts/5.materialized.py
cd dbt_pipeline && dbt run --select mart_corn_training_weekly && cd ..
python scripts/6.train_forecast.py
python scripts/7.inference.py
python dashboard/app.py
```

### Run tests

```bash
pip install ruff mypy pytest
ruff check scripts/ tests/ dashboard/
mypy scripts/6.train_forecast.py scripts/7.inference.py --ignore-missing-imports
pytest tests/ -v
```

---

## Dashboard

The Dash dashboard has two tabs:

**📈 13-Week Forecast**
- Historical price line (last 2 years)
- XGBoost p50 forecast with p10–p90 shaded confidence band
- Per-horizon detail table (p10 / p50 / p90 / ±spread)

**⚠️ Composite Risk Score**
- Gauge (0–100) coloured green / amber / red
- Component breakdown:
  - **Price Volatility** — 12-week coefficient of variation (40 % weight)
  - **Forecast Uncertainty** — relative band width at h=13 (40 % weight)
  - **Price Momentum** — 12-week absolute price change rate (20 % weight)

The dashboard falls back to synthetic demo data if the pipeline hasn't been run yet.

---

## Deployment

### Render (recommended free tier)

1. Push this repository to GitHub.
2. Create a new **Web Service** on [render.com](https://render.com), connect the repo.
3. Render auto-detects `render.yaml` — no manual config needed.
4. The dashboard will be live at `https://<service-name>.onrender.com`.

```bash
# Manual CLI deploy
render up
```

### Fly.io

```bash
# Install flyctl, then:
fly auth login
fly launch --config fly.toml    # first deploy
fly deploy                       # subsequent deploys
```

The app will be available at `https://supply-chain-dashboard.fly.dev`.

---

## CI — GitHub Actions

Every push triggers:

| Step         | Tool    | Scope                              |
|--------------|---------|------------------------------------|
| Lint         | `ruff`  | `scripts/`, `tests/`, `dashboard/` |
| Format check | `ruff`  | same                               |
| Type check   | `mypy`  | training + inference scripts       |
| Unit tests   | `pytest`| `tests/`                           |

See `.github/workflows/ci.yml`.

---

## Extending to Other Ingredients

```bash
# Soybean example
python scripts/0.generate_sample_data.py --ingredient SOYBEAN
python scripts/6.train_forecast.py --ingredient SOYBEAN
python scripts/7.inference.py --ingredient SOYBEAN

# dbt (with real data)
dbt run --select mart_corn_training_weekly --vars '{ingredient_name: SOYBEAN}'
```

---

## MLflow Experiment Tracking

```bash
mlflow ui          # opens http://localhost:5000
```

Logs per-horizon MAE, RMSE, MAPE, baseline MAE, and skill-vs-naive for every training run.
