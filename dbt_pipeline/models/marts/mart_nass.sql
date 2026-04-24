-- mart_nass.sql
-- Replaces: join_nass() in 5.materialized.py (lines 252-272)
-- fred_mapping JOIN nass ON ingredient_name AND unit_of_measure, filtered by ingredient_name.
-- Forward rolling averages use day intervals (7d-365d) on amount.

{{ config(
    materialized='external',
    location='../data/materialized/' ~ var('ingredient_name') ~ '/nass.parquet',
    format='parquet'
) }}

with joined as (

    select
        fm.ingredient_id,
        fm.ingredient_group,
        fm.ingredient_description,
        fm.ingredient_name,
        fm.unit_of_measure,
        fm.fred_series_id,
        n.amount,
        n.year,
        n.frequency,
        n.range,
        n.load_time,
        n.state,
        n.country

    from {{ ref('stg_fred_mapping') }} fm
    inner join {{ ref('stg_nass_production') }} n
        on fm.ingredient_name = n.ingredient_name
        and fm.unit_of_measure = n.unit_of_measure
    where fm.ingredient_name like '%{{ var("ingredient_name") }}%'

),

with_rolling as (

    select
        *,
        {{ rolling_avg_days('amount', 'load_time', 'ingredient_id', [7, 14, 30, 90, 180, 365]) }}

    from joined

)

select
    *,
    {{ fwd_pct_change('amount', 'days', [7, 14, 30, 90, 180, 365]) }}

from with_rolling
order by ingredient_id, load_time
