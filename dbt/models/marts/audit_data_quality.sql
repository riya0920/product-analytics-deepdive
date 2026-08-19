-- audit_data_quality: what the cleaning removed, and what remains suspicious.
--
-- Reported alongside every analysis. An analysis that cleans data without
-- quantifying what it cleaned is asking the reader to trust an invisible step.

WITH raw_counts AS (
    SELECT
        COUNT(*) AS raw_rows,
        COUNT(*) FILTER (WHERE user_id IS NULL) AS null_user_rows,
        COUNT(DISTINCT user_id) AS raw_users
    FROM {{ source('raw', 'raw_events') }}
),

dup_counts AS (
    SELECT COALESCE(SUM(n - 1), 0) AS duplicate_rows
    FROM (
        SELECT COUNT(*) AS n
        FROM {{ source('raw', 'raw_events') }}
        WHERE user_id IS NOT NULL
        GROUP BY user_id, event_name, event_ts
        HAVING COUNT(*) > 1
    )
),

clean_counts AS (
    SELECT COUNT(*) AS clean_rows, COUNT(DISTINCT user_id) AS clean_users FROM {{ ref('stg_events') }}
)

SELECT
    r.raw_rows,
    c.clean_rows,
    r.raw_rows - c.clean_rows AS rows_removed,
    d.duplicate_rows,
    ROUND(d.duplicate_rows::DOUBLE / NULLIF(r.raw_rows, 0), 5) AS duplicate_rate,
    r.null_user_rows,
    ROUND(r.null_user_rows::DOUBLE / NULLIF(r.raw_rows, 0), 5) AS null_user_rate,
    c.clean_users
FROM raw_counts r, dup_counts d, clean_counts c
