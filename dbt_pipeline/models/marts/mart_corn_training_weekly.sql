-- mart_corn_training_weekly.sql
-- Unified weekly training table for the 13-week direct multi-output forecast.
--
-- Leakage rule: every feature on row t is computed from data available at or
-- before the start of week t. No forward-looking windows. The training script
-- builds y_{t+h} (h = 1..13) by shifting price_avg_weekly backward.
--
-- Cross features (dollar index, macro PPI) are resampled to the weekly grid;
-- the training script forward-fills the remaining NULLs in Python.

{{ config(
    materialized='external',
    location='../data/materialized/' ~ var('ingredient_name') ~ '/training_weekly.parquet',
    format='parquet'
) }}

with weekly_price as (

    select
        date_trunc('week', ml.report_date)     as week_start,
        avg(ml.price_avg)                      as price_avg_weekly,
        min(ml.price_min)                      as price_min_weekly,
        max(ml.price_max)                      as price_max_weekly,
        count(*)                               as n_reports_in_week

    from {{ ref('stg_ams_market') }} ml
    inner join {{ ref('stg_fred_mapping') }} fm
        on fm.ingredient_name = ml.ams_ingredient_name
    where fm.ingredient_name like '%{{ var("ingredient_name") }}%'
    group by 1

),

with_backward_features as (

    select
        *,
        -- backward lags (in weeks)
        lag(price_avg_weekly,  1) over (order by week_start) as price_lag_1w,
        lag(price_avg_weekly,  2) over (order by week_start) as price_lag_2w,
        lag(price_avg_weekly,  4) over (order by week_start) as price_lag_4w,
        lag(price_avg_weekly,  8) over (order by week_start) as price_lag_8w,
        lag(price_avg_weekly, 13) over (order by week_start) as price_lag_13w,
        lag(price_avg_weekly, 26) over (order by week_start) as price_lag_26w,
        lag(price_avg_weekly, 52) over (order by week_start) as price_lag_52w,

        -- trailing rolling means (exclude current row — strictly backward)
        avg(price_avg_weekly) over (
            order by week_start rows between  4 preceding and 1 preceding
        ) as price_hist_avg_4w,
        avg(price_avg_weekly) over (
            order by week_start rows between 12 preceding and 1 preceding
        ) as price_hist_avg_12w,
        avg(price_avg_weekly) over (
            order by week_start rows between 26 preceding and 1 preceding
        ) as price_hist_avg_26w,
        avg(price_avg_weekly) over (
            order by week_start rows between 52 preceding and 1 preceding
        ) as price_hist_avg_52w,

        -- trailing volatility
        stddev_pop(price_avg_weekly) over (
            order by week_start rows between 12 preceding and 1 preceding
        ) as price_hist_std_12w

    from weekly_price

),

-- Dollar index: daily → weekly (last observation within the week)
di_weekly as (

    select
        date_trunc('week', ppi_date) as week_start,
        last(ppi order by ppi_date)  as di_ppi

    from {{ ref('stg_dollar_index') }}
    group by 1

),

-- Macro PPI for this ingredient: monthly → weekly (stamped on the week
-- containing the monthly observation; Python ffills the rest)
macro_weekly as (

    select
        date_trunc('week', m.ppi_date) as week_start,
        avg(m.ppi)                     as macro_ppi

    from {{ ref('stg_fred_ppi') }} m
    where m.ingredient_id in (
        select distinct ingredient_id
        from {{ ref('stg_fred_mapping') }}
        where ingredient_name = '{{ var("ingredient_name") }}'
    )
    group by 1

)

select
    wbf.week_start,
    wbf.price_avg_weekly,
    wbf.price_min_weekly,
    wbf.price_max_weekly,
    wbf.n_reports_in_week,

    wbf.price_lag_1w,
    wbf.price_lag_2w,
    wbf.price_lag_4w,
    wbf.price_lag_8w,
    wbf.price_lag_13w,
    wbf.price_lag_26w,
    wbf.price_lag_52w,

    wbf.price_hist_avg_4w,
    wbf.price_hist_avg_12w,
    wbf.price_hist_avg_26w,
    wbf.price_hist_avg_52w,
    wbf.price_hist_std_12w,

    di.di_ppi,
    mw.macro_ppi,

    extract(year    from wbf.week_start) as year,
    extract(week    from wbf.week_start) as week_of_year,
    extract(month   from wbf.week_start) as month,
    extract(quarter from wbf.week_start) as quarter

from with_backward_features wbf
left join di_weekly    di on wbf.week_start = di.week_start
left join macro_weekly mw on wbf.week_start = mw.week_start
order by wbf.week_start
