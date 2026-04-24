-- stg_product_master.sql
-- Source: product_master.csv (Spark output directory, reads part-*.csv)
-- Reference table: normalized commodity definitions with ingredient_id as PK.
-- Simple type casts only — data was already normalized by 1.create_product_master.py.

with source as (

    select * from {{ source('raw_production', 'product_master') }}

)

select
    cast(ingredient_id as integer)             as ingredient_id,
    cast(ingredient_group as varchar)          as ingredient_group,
    cast(ingredient_description as varchar)    as ingredient_description,
    cast(ingredient_name as varchar)           as ingredient_name,
    cast(unit_of_measure as varchar)           as unit_of_measure

from source
