-- fwd_pct_change.sql
-- Replaces: _fwd_pct_days() and _fwd_pct_months() in 5.materialized.py (lines 150-173)
--
-- Generates percent-change columns between each forward rolling average and the current value.
-- Works with both day-based and month-based rolling columns.
--
-- For day-based rolling columns (suffix _fwd_{n}d):
--   {{ fwd_pct_change('ppi', 'days', [7, 14, 30, 90, 180, 365]) }}
--   Produces: ppi_fwd_7d_pct, ppi_fwd_14d_pct, ...
--
-- For month-based rolling columns (suffix _fwd_{n}mo):
--   {{ fwd_pct_change('ppi', 'months', [1, 3, 6, 12]) }}
--   Produces: ppi_fwd_1mo_pct, ppi_fwd_3mo_pct, ...

{% macro fwd_pct_change(value_col, interval_type, intervals) %}

    {% if interval_type == 'days' %}
        {% set suffix = 'd' %}
    {% elif interval_type == 'months' %}
        {% set suffix = 'mo' %}
    {% endif %}

    {% for n in intervals %}
        case
            when {{ value_col }} != 0
            then ({{ value_col }}_fwd_{{ n }}{{ suffix }} - {{ value_col }}) / {{ value_col }}
        end as {{ value_col }}_fwd_{{ n }}{{ suffix }}_pct
        {%- if not loop.last %},{% endif %}
    {% endfor %}

{% endmacro %}
