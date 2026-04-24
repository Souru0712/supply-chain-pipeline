# dbt + DuckDB Integration Plan

## Context

The supply chain pipeline currently runs as 6 numbered Python scripts executed manually in sequence: ingestion (APIs → CSV), transformation (PySpark → Parquet), validation (Great Expectations), materialization (DuckDB SQL → Parquet), and ML/visualization. There is no scheduler, no centralized documentation, and the transformation/materialization SQL logic lives embedded in Python scripts. Integrating dbt will formalize the SQL transformation layer, add built-in testing and documentation, and — paired with Dagster — provide automated scheduling with a monitoring UI.

**User decisions:**
- **Quarantine:** dbt handles post-ingestion quarantine via staging models. Great Expectations stays for pre-ingestion checks only.
- **Scheduler:** Dagster with `dagster-dbt` integration.
- **Install:** Plan includes full installation steps.

---

## Phase 1: Installation & Scaffold

### 1.1 Install packages

```bash
pip install dbt-core dbt-duckdb dagster dagster-dbt dagster-webserver
```

Note: `dbt_utils` is a **dbt package** (not a pip package) — it gets installed via `packages.yml` + `dbt deps` in step 1.4.

### 1.2 Create dbt project manually

Skip `dbt init` — it scaffolds boilerplate (sample models, example folders) that we'd immediately delete. Instead, create the exact structure we need:

```bash
mkdir -p dbt_pipeline/{models/staging,models/marts,macros,tests/generic,seeds}
```

### 1.3 Final dbt project structure

```
supply-chain-pipeline/
├── dbt_pipeline/                     # NEW — dbt project root
│   ├── dbt_project.yml
│   ├── profiles.yml
│   ├── packages.yml
│   ├── models/
│   │   ├── sources.yml               # raw CSV/Parquet source definitions
│   │   ├── overview.md               # {% docs __overview__ %} landing page
│   │   ├── macros.md                 # {% docs %} blocks for macro documentation
│   │   ├── _macros.yml               # wires doc blocks to macro descriptions
│   │   ├── staging/
│   │   │   ├── _stg_models.yml       # schema + tests for staging
│   │   │   ├── stg_fred_ppi.sql
│   │   │   ├── stg_dollar_index.sql
│   │   │   ├── stg_ams_market.sql
│   │   │   ├── stg_nass_production.sql
│   │   │   ├── stg_product_master.sql
│   │   │   └── stg_fred_mapping.sql
│   │   └── marts/
│   │       ├── _mart_models.yml      # schema + tests for marts
│   │       ├── mart_dollar_index.sql
│   │       ├── mart_macroeconomic.sql
│   │       ├── mart_market_and_logistic.sql
│   │       └── mart_nass.sql
│   ├── macros/
│   │   ├── rolling_avg_days.sql      # reusable window function macro
│   │   ├── rolling_avg_months.sql
│   │   └── fwd_pct_change.sql
│   ├── tests/
│   │   └── generic/
│   │       └── test_no_lowercase.sql # custom test: all-uppercase check
│   └── seeds/                        # empty for now
├── orchestration/                    # NEW — Dagster definitions
│   ├── __init__.py
│   ├── definitions.py                # Dagster code location
│   ├── assets.py                     # ingestion + dbt assets
│   └── schedules.py                  # daily schedule
├── gx/                               # KEPT — for pre-ingestion validation only
├── data/                             # UNCHANGED — dbt reads/writes here
├── 1.create_product_master.py        # KEPT
├── 2a.ingest_fred.py                 # KEPT
├── 2b.ingest_dollar_index.py         # KEPT
├── 2c.ingest_ams.py                  # KEPT
├── 3.transformed.py                  # DEPRECATED after Phase 3
├── 4.ge_checkpoint.py                # DEPRECATED after Phase 4 (pre-ingestion GE stays in gx/)
├── 5.materialized.py                 # DEPRECATED after Phase 4
├── 6.visualization.py                # KEPT — reads from same Parquet paths
└── .env
```

### 1.4 Configuration files

**`dbt_pipeline/profiles.yml`** — uses in-memory DuckDB (stateless; only produces external Parquet):
```yaml
supply_chain:
  outputs:
    dev:
      type: duckdb
      path: ":memory:"
      threads: 4
      extensions:
        - parquet
  target: dev
```

**`dbt_pipeline/dbt_project.yml`**:
```yaml
name: supply_chain
version: '1.0.0'
profile: supply_chain

model-paths: ["models"]
test-paths: ["tests"]
seed-paths: ["seeds"]
macro-paths: ["macros"]

vars:
  ingredient_name: "CORN"
  pipeline_root: ".."

models:
  supply_chain:
    staging:
      +materialized: view
    marts:
      +materialized: external
```

**`dbt_pipeline/packages.yml`**:
```yaml
packages:
  - package: dbt-labs/dbt_utils
    version: [">=1.0.0", "<2.0.0"]
```

Then run: `cd dbt_pipeline && dbt deps && dbt debug`

---

## Phase 2: Sources & Staging Models (replaces `3.transformed.py`)

### 2.1 Source definitions

**`models/sources.yml`** — DuckDB reads raw CSVs/Parquet via `read_csv()`/`read_parquet()` globs:

| Source | Points to | Notes |
|--------|-----------|-------|
| `raw_macroeconomic.fred_ppi` | `data/raw/macroeconomic/commodity_prices_*.csv` | `read_csv()` with glob |
| `raw_macroeconomic.dollar_index` | `data/raw/macroeconomic/dollar_index_*.csv` | `read_csv()` |
| `raw_market.ams_reports` | `data/raw/market_and_logistic/AMS_*/ReportDetail.csv` | `read_csv()` with glob |
| `raw_production.nass_crops` | `data/raw/production/qs.crops_20260327.txt` | `read_csv()` tab-delimited |
| `raw_production.product_master` | `data/raw/production/product_master.csv` | `read_csv()` |
| `raw_production.fred_mapping` | `data/raw/production/fred_mapped.csv` | `read_csv()` |

### 2.2 Staging models — what each one does

Each model translates the PySpark logic in `3.transformed.py` to pure DuckDB SQL:

| Model | Replaces | Key transforms |
|-------|----------|----------------|
| `stg_fred_ppi.sql` | `transform_fred()` | Cast ingredient_id→int, ppi→float, date→date; rename columns |
| `stg_dollar_index.sql` | `transform_dollar_index()` | Cast types, drop FRED null sentinel "." |
| `stg_ams_market.sql` | `transform_ams()` | UPPER all strings, normalize SOYBEANS→SOYBEAN, parse MM/DD/YYYY→date, cast prices→float |
| `stg_nass_production.sql` | `transform_nass()` | Strip commas from VALUE, cast amount→float, rename NASS columns |
| `stg_product_master.sql` | (reference) | Simple select from product_master.csv with type casts |

**Materialization:** All staging models are `view` (lightweight, no Parquet written).

**Validation split:**
- **Pre-ingestion:** Great Expectations validates API response quality before writing to `data/raw/` (kept in `gx/`)
- **Post-ingestion:** dbt tests (`not_null`, `unique`, `unique_combination_of_columns`, custom `no_lowercase`) validate staged models — see Phase 4

**Critical files to port from:**
- [3.transformed.py](scripts/3.transformed.py) — all `transform_*()` functions (lines ~50–336)

---

## Phase 3: Mart Models (replaces `5.materialized.py`)

### 3.1 Reusable Jinja macros

Extract the repetitive rolling-average window functions into macros:

| Macro | Purpose |
|-------|---------|
| `rolling_avg_days(col, date_col, partition_col, [7,14,30,90,180,365])` | Forward rolling AVG with day intervals |
| `rolling_avg_months(col, date_col, partition_col, [1,3,6,12])` | Forward rolling AVG with month intervals |
| `fwd_pct_change(col, intervals)` | `(fwd - current) / current` for each interval |

### 3.2 Mart models

| Model | Replaces | Output path | Key logic |
|-------|----------|-------------|-----------|
| `mart_dollar_index.sql` | `materialize_dollar_index()` | `data/materialized/CORN/dollar_index.parquet` | Standalone: rolling avg 7d–365d + pct change |
| `mart_macroeconomic.sql` | `join_macroeconomic()` | `data/materialized/CORN/macroeconomic.parquet` | JOIN product_master + fred_ppi ON ingredient_id; rolling avg 1mo–12mo |
| `mart_market_and_logistic.sql` | `join_market_and_logistic()` | `data/materialized/CORN/market_and_logistic.parquet` | JOIN product_master + ams ON ingredient_name; daily agg + rolling avg 7d–365d |
| `mart_nass.sql` | `join_nass()` | `data/materialized/CORN/nass.parquet` | JOIN product_master + nass ON (ingredient_name, unit); rolling avg |

**Materialization:** All mart models use `external` (dbt-duckdb writes Parquet directly). Output paths match the existing `data/materialized/CORN/` structure so `6.visualization.py` works unchanged.

**Ingredient parameterization:** `WHERE ingredient_name = '{{ var("ingredient_name") }}'` in each mart. Run other ingredients via: `dbt run --select marts --vars '{ingredient_name: SOYBEAN}'`

**Critical file to port from:**
- [5.materialized.py](5.materialized.py) — all SQL CTEs and window functions (lines ~50–344)

---

## Phase 4: Data Tests (replaces post-ingestion GE checks)

### 4.1 Test mapping from Great Expectations → dbt

| GE Expectation | dbt Test | Applied to |
|----------------|----------|------------|
| `expect_compound_columns_to_be_unique` | `dbt_utils.unique_combination_of_columns` | `stg_fred_ppi` (ingredient_id + ppi_date) |
| `expect_column_values_to_not_be_null` | `not_null` | All key columns |
| `expect_column_values_to_be_of_type(float)` | Custom singular test or `dbt_utils.expression_is_true` | ppi, price_avg, amount |
| `expect_column_values_to_match_regex(YYYY-MM-DD)` | `dbt_utils.expression_is_true` with `regexp_matches()` | date columns |
| All-uppercase check | Custom generic test `no_lowercase` | All string columns in stg_ams_market, stg_nass |

### 4.2 Custom generic test

**`tests/generic/test_no_lowercase.sql`:**
```sql
{% test no_lowercase(model, column_name) %}
select {{ column_name }}
from {{ model }}
where {{ column_name }} is not null
  and {{ column_name }} != upper({{ column_name }})
{% endtest %}
```

### 4.3 Schema YAML files

Tests are defined in `_stg_models.yml` and `_mart_models.yml` alongside column descriptions. Example structure:

```yaml
models:
  - name: stg_fred_ppi
    columns:
      - name: ingredient_id
        tests: [not_null]
      - name: ppi
        tests: [not_null]
      - name: ppi_date
        tests: [not_null]
    tests:
      - dbt_utils.unique_combination_of_columns:
          combination_of_columns: [ingredient_id, ppi_date]
```

### 4.4 Great Expectations scope change

GE stays in `gx/` but is scoped to **pre-ingestion validation only** (checking API response quality before writing to `data/raw/`). All post-ingestion validation moves to dbt tests.

---

## Phase 5: Documentation

### 5.1 What gets documented

- **Model descriptions** in `_stg_models.yml` and `_mart_models.yml` — every model and column gets a description
- **Source descriptions** in `sources.yml` — documents each API source and its refresh frequency
- **Project overview** in `models/overview.md` using dbt's `{% docs __overview__ %}` block — explains the full pipeline architecture, data sources, and how to run
- **Macro descriptions** — document the rolling average and pct change macros

### 5.2 Generate and serve

```bash
cd dbt_pipeline
dbt docs generate
dbt docs serve --port 8080
```

This produces an interactive documentation site with the full DAG visualizer showing staging → mart lineage.

---

## Phase 6: Scheduling with Dagster

### 6.1 Install and structure

```bash
pip install dagster dagster-dbt dagster-webserver
```

**`orchestration/assets.py`** defines:

| Asset | Type | What it runs |
|-------|------|-------------|
| `ingest_fred` | Python subprocess | `python 2a.ingest_fred.py` |
| `ingest_dollar_index` | Python subprocess | `python 2b.ingest_dollar_index.py` |
| `ingest_ams` | Python subprocess | `python 2c.ingest_ams.py` |
| `dbt_assets` | dagster-dbt | `dbt run` (staging + marts) |
| `dbt_tests` | dagster-dbt | `dbt test` |

### 6.2 Asset dependency graph

```
ingest_fred ──────┐
ingest_dollar ────┤
ingest_ams ───────┼──→ dbt_assets (run) ──→ dbt_tests (test)
                  │
product_master ───┘  (manual/infrequent — not scheduled)
```

Ingestion assets run in parallel, then dbt runs sequentially.

### 6.3 Schedule

```python
# Daily at 6 AM UTC — captures all data source update frequencies
daily_pipeline = ScheduleDefinition(
    job=full_pipeline_job,
    cron_schedule="0 6 * * *",
)
```

### 6.4 Launch

```bash
cd orchestration
dagster dev
# Opens UI at http://localhost:3000
```

---

## Migration Order & Verification

Execute phases sequentially. Each phase is independently verifiable:

| Phase | What to build | How to verify | Rollback |
|-------|--------------|---------------|----------|
| 1. Scaffold | Config files, `dbt debug` | `dbt debug` returns "All checks passed" | Delete `dbt_pipeline/` |
| 2. Staging | 5 staging models + quarantine models | Compare row counts and spot-check values: `dbt run --select staging` then DuckDB query diffing against existing `data/staged/` Parquet | Run `3.transformed.py` to regenerate |
| 3. Marts | 4 mart models + 3 macros | `dbt run --select marts` then verify `6.visualization.py` produces identical charts | Run `5.materialized.py CORN` to regenerate |
| 4. Tests | Schema YAML + custom tests | `dbt test` — all pass; intentionally corrupt a row and verify test catches it | GE checkpoint still works |
| 5. Docs | Descriptions + overview.md | `dbt docs serve` — browse DAG and verify all descriptions render | N/A |
| 6. Dagster | Assets + schedule | `dagster dev` — trigger full pipeline from UI, verify end-to-end | Run scripts manually |

**After full migration:** deprecate `3.transformed.py`, `4.ge_checkpoint.py` (post-ingestion parts), and `5.materialized.py` by moving them to a `_deprecated/` folder.
