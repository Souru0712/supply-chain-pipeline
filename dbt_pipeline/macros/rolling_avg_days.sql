-- rolling_avg_days.sql
-- Replaces: _rolling_days() in 5.materialized.py (lines 122-133)
--
-- Generates forward rolling AVG columns using date-range intervals (days).
-- Produces columns named: {value_col}_fwd_{d}d for each day in days_list.
--
-- Usage:
--   {{ rolling_avg_days('ppi', 'ppi_date', 'fred_series_id', [7, 14, 30, 90, 180, 365]) }}

{% macro rolling_avg_days(value_col, date_col, partition_col, days_list) %}

    {% for d in days_list %}
        avg({{ value_col }}) over (
            partition by {{ partition_col }}
            order by {{ date_col }}
            range between current row and interval '{{ d }} days' following
        ) as {{ value_col }}_fwd_{{ d }}d
        {%- if not loop.last %},{% endif %}
    {% endfor %}

{% endmacro %}
