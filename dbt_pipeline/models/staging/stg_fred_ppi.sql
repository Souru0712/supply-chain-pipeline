-- stg_fred_ppi.sql
-- Replaces: transform_fred() in 3.transformed.py (lines 100-122)
-- Source:   FRED API commodity price CSVs (glob reads all timestamped files)
-- Selects 4 columns, renames date→ppi_date and frequency→ppi_frequency, casts types.

with source as (

    select * from {{ source('raw_macroeconomic', 'fred_ppi') }}

)

-- nullstr=['.', ''] on the source read turns FRED sentinels into NULLs,
-- so we only need to drop the resulting NULLs here.
select
    cast(ingredient_id as integer)  as ingredient_id,
    try_cast(ppi as float)          as ppi,
    cast(date as date)              as ppi_date,
    cast(frequency as varchar)      as ppi_frequency

from source
where ppi is not null
