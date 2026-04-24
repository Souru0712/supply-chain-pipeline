-- stg_dollar_index.sql
-- Replaces: transform_and_write() in 2b.ingest_dollar_index.py (lines 139-158)
-- Source:   FRED dollar index CSVs (glob reads all timestamped files)
-- Drops FRED null sentinel ".", casts types.

with source as (

    select * from {{ source('raw_macroeconomic', 'dollar_index') }}

)

-- nullstr=['.', ''] on the source read turns FRED sentinels into NULLs,
-- so we only need to drop the resulting NULLs here.
select
    cast(fred_series_id as varchar)  as fred_series_id,
    cast(name as varchar)            as name,
    try_cast(ppi as float)           as ppi,
    cast(ppi_date as date)           as ppi_date,
    cast(ppi_frequency as varchar)   as ppi_frequency

from source
where ppi is not null
