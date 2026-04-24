-- rolling_avg_months.sql
-- Replaces: _rolling_months() in 5.materialized.py (lines 136-147)
--
-- Generates forward rolling AVG columns using month intervals (for monthly data like PPI).
-- Produces columns named: {value_col}_fwd_{m}mo for each month in months_list.
--
-- Usage:
--   {{ rolling_avg_months('ppi', 'ppi_date', 'ingredient_id', [1, 3, 6, 12]) }}

{% macro rolling_avg_months(value_col, date_col, partition_col, months_list) %}

    {% for m in months_list %}
        avg({{ value_col }}) over (
            partition by {{ partition_col }}
            order by {{ date_col }}
            range between current row and interval '{{ m }} months' following
        ) as {{ value_col }}_fwd_{{ m }}mo
        {%- if not loop.last %},{% endif %}
    {% endfor %}

{% endmacro %}
