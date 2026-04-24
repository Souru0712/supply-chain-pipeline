-- mart_market_and_logistic.sql
-- Replaces: join_market_and_logistic() in 5.materialized.py (lines 214-249)
-- fred_mapping JOIN market_and_logistic ON ingredient_name, filtered by ingredient_name.
-- Daily aggregation: one row per (ingredient_id, report_date) using
-- AVG(price_avg), MIN(price_min), MAX(price_max).
-- Forward rolling averages use day intervals (7d-365d) on price_avg.

{{ config(
    materialized='external',
    location='../data/materialized/' ~ var('ingredient_name') ~ '/market_and_logistic.parquet',
    format='parquet'
) }}

with daily_agg as (

    select
        fm.ingredient_id,
        fm.ingredient_group,
        fm.ingredient_description,
        fm.ingredient_name,
        fm.unit_of_measure,
        fm.fred_series_id,
        ml.report_title,
        ml.ams_ingredient_name,
        min(ml.price_min)   as price_min,
        max(ml.price_max)   as price_max,
        avg(ml.price_avg)   as price_avg,
        ml.price_unit,
        ml.sale_type,
        ml.report_date

    from {{ ref('stg_fred_mapping') }} fm
    inner join {{ ref('stg_ams_market') }} ml
        on fm.ingredient_name = ml.ams_ingredient_name
    where fm.ingredient_name like '%{{ var("ingredient_name") }}%'
    group by
        fm.ingredient_id,
        fm.ingredient_group,
        fm.ingredient_description,
        fm.ingredient_name,
        fm.unit_of_measure,
        fm.fred_series_id,
        ml.report_title,
        ml.ams_ingredient_name,
        ml.price_unit,
        ml.sale_type,
        ml.report_date

),

with_rolling as (

    select
        *,
        {{ rolling_avg_days('price_avg', 'report_date', 'ingredient_id', [7, 14, 30, 90, 180, 365]) }}

    from daily_agg

)

select
    *,
    {{ fwd_pct_change('price_avg', 'days', [7, 14, 30, 90, 180, 365]) }}

from with_rolling
order by ingredient_id, report_date
