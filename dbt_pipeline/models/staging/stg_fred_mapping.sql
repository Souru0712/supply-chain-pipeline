-- stg_fred_mapping.sql
-- Replaces: transform_fred_mapping() in 3.transformed.py (lines 262-287)
-- Source:   fred_mapped.csv — maps ingredient_id to FRED series_id
-- Casts types. Nullifies fred_series_id where value is "no ppi available".

with source as (

    select * from {{ source('raw_production', 'fred_mapping') }}

)

select
    cast(ingredient_id as integer)             as ingredient_id,
    cast(ingredient_group as varchar)          as ingredient_group,
    cast(ingredient_description as varchar)    as ingredient_description,
    cast(ingredient_name as varchar)           as ingredient_name,
    cast(unit_of_measure as varchar)           as unit_of_measure,
    case
        when upper(fred_series_id) = 'NO PPI AVAILABLE' then null
        else cast(fred_series_id as varchar)
    end as fred_series_id

from source
