-- mart_dollar_index.sql
-- Replaces: materialize_dollar_index() in 5.materialized.py (lines 176-191)
-- Standalone dollar_index with forward rolling averages (7d-365d) and pct changes on ppi.
-- No ingredient filter — dollar index applies globally across all commodities.

{{ config(
    materialized='external',
    location='../data/materialized/' ~ var('ingredient_name') ~ '/dollar_index.parquet',
    format='parquet'
) }}

with base as (

    select
        fred_series_id,
        name,
        ppi,
        ppi_date,
        ppi_frequency,
        {{ rolling_avg_days('ppi', 'ppi_date', 'fred_series_id', [7, 14, 30, 90, 180, 365]) }}

    from {{ ref('stg_dollar_index') }}

)

select
    *,
    {{ fwd_pct_change('ppi', 'days', [7, 14, 30, 90, 180, 365]) }}

from base
order by ppi_date
