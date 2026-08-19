-- mart_sessions: sessionise the event stream with a 30-minute inactivity gap.
--
-- This is the canonical window-function exercise and it is the one that shows up
-- in a live SQL round: LAG to find the previous event, a boolean for "gap
-- exceeded", then a running SUM over that boolean to allocate session ids.
-- Written out rather than hidden in pandas, because the point is the SQL.

WITH ordered AS (
    SELECT
        user_id,
        event_id,
        event_name,
        event_ts,
        revenue,
        LAG(event_ts) OVER (PARTITION BY user_id ORDER BY event_ts, event_id) AS prev_ts
    FROM {{ ref('stg_events') }}
),

flagged AS (
    SELECT
        *,
        CASE
            WHEN prev_ts IS NULL THEN 1
            WHEN DATE_DIFF('minute', prev_ts, event_ts) > 30 THEN 1
            ELSE 0
        END AS is_new_session
    FROM ordered
),

numbered AS (
    SELECT
        *,
        -- Running sum of the new-session flag = session index within the user.
        SUM(is_new_session) OVER (
            PARTITION BY user_id
            ORDER BY event_ts, event_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS session_index
    FROM flagged
)

SELECT
    user_id,
    user_id || '_s' || session_index AS session_id,
    session_index,
    MIN(event_ts) AS session_start,
    MAX(event_ts) AS session_end,
    DATE_DIFF('second', MIN(event_ts), MAX(event_ts)) AS session_seconds,
    COUNT(*) AS event_count,
    SUM(revenue) AS session_revenue,
    MAX(CASE WHEN event_name = 'purchase' THEN 1 ELSE 0 END) AS converted
FROM numbered
GROUP BY user_id, session_index
