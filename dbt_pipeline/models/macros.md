{% docs rolling_avg_days %}

Generates forward rolling average columns using **day-based** date-range window functions.

Uses DuckDB's `RANGE BETWEEN CURRENT ROW AND INTERVAL 'N days' FOLLOWING` syntax, which calculates averages based on calendar distance rather than row count. This correctly handles gaps in reporting (weekends, holidays, irregular frequencies).

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `value_col` | string | Column to average (e.g. `'ppi'`, `'n.amount'`) |
| `date_col` | string | Date column for ordering (e.g. `'ppi_date'`) |
| `partition_col` | string | Column to partition by (e.g. `'fred_series_id'`) |
| `days_list` | list | Day intervals (e.g. `[7, 14, 30, 90, 180, 365]`) |

### Output Columns

For `value_col = 'ppi'` and `days_list = [7, 14, 30]`:
- `ppi_fwd_7d` — average ppi over next 7 calendar days
- `ppi_fwd_14d` — average ppi over next 14 calendar days
- `ppi_fwd_30d` — average ppi over next 30 calendar days

### Usage

{% raw %}
```sql
select
    {{ rolling_avg_days('ppi', 'ppi_date', 'fred_series_id', [7, 14, 30, 90, 180, 365]) }}
from {{ ref('stg_dollar_index') }}
```
{% endraw %}

{% enddocs %}


{% docs rolling_avg_months %}

Generates forward rolling average columns using **month-based** date-range window functions.

Designed for data with monthly reporting frequency (e.g. FRED PPI). Uses `INTERVAL 'N months' FOLLOWING` to align with calendar months rather than fixed day counts.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `value_col` | string | Column to average (e.g. `'m.ppi'`) |
| `date_col` | string | Date column for ordering (e.g. `'m.ppi_date'`) |
| `partition_col` | string | Column to partition by (e.g. `'fm.ingredient_id'`) |
| `months_list` | list | Month intervals (e.g. `[1, 3, 6, 12]`) |

### Output Columns

For `value_col = 'ppi'` and `months_list = [1, 3, 6, 12]`:
- `ppi_fwd_1mo` — average ppi over next 1 month
- `ppi_fwd_3mo` — average ppi over next 3 months
- `ppi_fwd_6mo` — average ppi over next 6 months
- `ppi_fwd_12mo` — average ppi over next 12 months

### Usage

{% raw %}
```sql
select
    {{ rolling_avg_months('m.ppi', 'm.ppi_date', 'fm.ingredient_id', [1, 3, 6, 12]) }}
from {{ ref('stg_fred_ppi') }} m
```
{% endraw %}

{% enddocs %}


{% docs fwd_pct_change %}

Generates percent-change columns between each forward rolling average and the current value. Must be called in an outer SELECT that wraps the rolling average CTE, since it references the `_fwd_Xd` / `_fwd_Xmo` columns produced by `rolling_avg_days` or `rolling_avg_months`.

### Formula

```
(forward_avg - current_value) / current_value
```

Returns NULL when the current value is 0 (division by zero guard).

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `value_col` | string | Base column name (e.g. `'ppi'`, `'price_avg'`) |
| `interval_type` | string | `'days'` or `'months'` — determines column suffix |
| `intervals` | list | Matching intervals from the rolling macro |

### Output Columns

For `value_col = 'ppi'`, `interval_type = 'days'`, `intervals = [7, 30, 90]`:
- `ppi_fwd_7d_pct` — percent change from ppi to 7-day forward avg
- `ppi_fwd_30d_pct` — percent change from ppi to 30-day forward avg
- `ppi_fwd_90d_pct` — percent change from ppi to 90-day forward avg

### Usage

{% raw %}
```sql
-- must wrap a CTE that already has the rolling avg columns
with base as (
    select ppi, {{ rolling_avg_days('ppi', 'ppi_date', 'id', [7, 30, 90]) }}
    from {{ ref('stg_dollar_index') }}
)
select *, {{ fwd_pct_change('ppi', 'days', [7, 30, 90]) }}
from base
```
{% endraw %}

{% enddocs %}
