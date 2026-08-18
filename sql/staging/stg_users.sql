-- stg_users: one row per user, with the acquisition attributes fixed at signup.
--
-- Channel is taken from the FIRST event rather than from any later event. A user
-- whose later events carry a different channel value has been re-attributed, and
-- using the latest value would silently move users between cohorts over time --
-- which makes historical retention curves change every time you re-run them.

SELECT
    user_id,
    MIN(event_ts) AS signup_ts,
    CAST(MIN(event_ts) AS DATE) AS signup_date,
    DATE_TRUNC('week', MIN(event_ts)) AS signup_week,
    FIRST(channel ORDER BY event_ts) AS channel,
    FIRST(platform ORDER BY event_ts) AS platform
FROM stg_events
GROUP BY user_id
