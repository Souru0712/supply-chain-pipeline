-- mart_macroeconomic.sql
-- Replaces: join_macroeconomic() in 5.materialized.py (lines 194-211)
-- fred_mapping JOIN macroeconomic ON ingredient_id, filtered by ingredient_name.
-- Forward rolling averages use month intervals (1mo-12mo) since PPI is reported monthly.

{{ config(
    materialized='external',
    location='../data/materialized/' ~ var('ingredient_name') ~ '/macroeconomic.parquet',
    format='parquet'
) }}

with base as (

    select
        fm.ingredient_id,
        fm.ingredient_group,
        fm.ingredient_description,
        fm.ingredient_name,
        fm.unit_of_measure,
        fm.fred_series_id,
        m.ppi,
        m.ppi_date,
        m.ppi_frequency,
        {{ rolling_avg_months('ppi', 'm.ppi_date', 'fm.ingredient_id', [1, 3, 6, 12]) }}

    from {{ ref('stg_fred_mapping') }} fm
    inner join {{ ref('stg_fred_ppi') }} m
        on fm.ingredient_id = m.ingredient_id
    where fm.ingredient_name = '{{ var("ingredient_name") }}'

)

select
    *,
    {{ fwd_pct_change('ppi', 'months', [1, 3, 6, 12]) }}

from base
order by ingredient_id, ppi_date
