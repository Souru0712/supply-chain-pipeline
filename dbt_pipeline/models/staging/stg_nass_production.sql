-- stg_nass_production.sql
-- Replaces: transform_nass() in 3.transformed.py (lines 211-259)
-- Source:   NASS QuickStats tab-delimited crop export
-- Selects 10 columns, renames to snake_case, strips commas from VALUE,
-- casts amount and cv_pct to float (non-numeric → NULL), casts year to int,
-- casts load_time to date.

with source as (

    select * from {{ source('raw_production', 'nass_crops') }}

),

renamed as (

    select
        cast(CLASS_DESC as varchar)              as ingredient_name,
        cast(UNIT_DESC as varchar)               as unit_of_measure,
        VALUE                                    as amount_raw,
        "CV_%"                                   as cv_pct_raw,
        YEAR                                     as year,
        cast(FREQ_DESC as varchar)               as frequency,
        cast(REFERENCE_PERIOD_DESC as varchar)   as range,
        LOAD_TIME                                as load_time_raw,
        cast(STATE_ALPHA as varchar)             as state,
        cast(COUNTRY_NAME as varchar)            as country

    from source

),

cleaned as (

    select
        ingredient_name,
        unit_of_measure,

        -- strip commas, cast to float; non-numeric values become NULL
        try_cast(
            replace(trim(cast(amount_raw as varchar)), ',', '')
            as float
        ) as amount,

        try_cast(
            replace(trim(cast(cv_pct_raw as varchar)), ',', '')
            as float
        ) as cv_pct,

        cast(year as integer)   as year,
        frequency,
        range,
        cast(load_time_raw as date) as load_time,
        state,
        country

    from renamed

)

select * from cleaned
