-- Replaces: ExpectColumnValuesToNotMatchRegex(regex=r"[a-z]") from 4.ge_checkpoint.py
-- Returns rows where a string column contains lowercase characters.
-- A non-empty result set means the test fails.

{% test no_lowercase(model, column_name) %}

select {{ column_name }}
from {{ model }}
where {{ column_name }} is not null
  and {{ column_name }} != upper({{ column_name }})

{% endtest %}
