-- stg_ams_market.sql
-- Replaces: transform_ams() in 3.transformed.py (lines 133-200)
-- Source:   USDA AMS ReportDetail CSVs across all AMS_*/ReportDetail.csv slugs
-- Renames columns, uppercases all strings, normalizes commodity names
-- (SOYBEANS→SOYBEAN, SUNFLOWER SEEDS→SUNFLOWER), casts prices to float(4),
-- parses report_date from MM/DD/YYYY to date.

with source as (

    select * from {{ source('raw_market', 'ams_reports') }}

),

renamed as (

    select
        cast(slug_id as integer)                     as slug_id,
        upper(report_title)                          as report_title,
        upper(trim(commodity))                       as ams_ingredient_name,
        upper(cat)                                   as ams_ingredient_group,
        upper(grade)                                 as ams_ingredient_grade,
        "price Min"                                  as price_min_raw,
        "price Max"                                  as price_max_raw,
        avg_price                                    as price_avg_raw,
        upper(price_unit)                            as price_unit,
        upper("sale Type")                           as sale_type,
        upper(delivery_point)                        as delivery_point,
        upper(freight)                               as freight,
        upper(trans_mode)                             as trans_mode,
        upper(market_location_state)                 as market_location_state,
        report_date                                  as report_date_raw

    from source

),

cleaned as (

    select
        slug_id,
        report_title,

        -- normalize commodity names to match fred_mapping
        case
            when ams_ingredient_name = 'SOYBEANS'        then 'SOYBEAN'
            when ams_ingredient_name = 'SUNFLOWER SEEDS' then 'SUNFLOWER'
            else ams_ingredient_name
        end as ams_ingredient_name,

        ams_ingredient_group,
        ams_ingredient_grade,

        -- price columns: NULL if empty string, else float rounded to 4 decimals
        case
            when trim(cast(price_min_raw as varchar)) = '' then null
            else round(cast(price_min_raw as float), 4)
        end as price_min,

        case
            when trim(cast(price_max_raw as varchar)) = '' then null
            else round(cast(price_max_raw as float), 4)
        end as price_max,

        case
            when trim(cast(price_avg_raw as varchar)) = '' then null
            else round(cast(price_avg_raw as float), 4)
        end as price_avg,

        price_unit,
        sale_type,
        delivery_point,
        freight,
        trans_mode,
        market_location_state,

        -- DuckDB's CSV reader auto-detects this column as DATE, so a plain
        -- cast is enough. Using try_cast → varchar → strptime as a fallback
        -- makes this resilient even if a future CSV arrives as a string.
        coalesce(
            try_cast(report_date_raw as date),
            try_cast(strptime(cast(report_date_raw as varchar), '%m/%d/%Y') as date)
        ) as report_date

    from renamed

)

select * from cleaned
