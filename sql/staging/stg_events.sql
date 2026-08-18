-- stg_events: the only place raw events are cleaned. Everything downstream
-- reads this, so a fix here is a fix everywhere.
--
-- Three defects are corrected, each deliberately visible rather than silent:
--   1. duplicate events   -> deduplicated on the natural key, not on event_id
--   2. ios timezone bug   -> local timestamps shifted back to UTC
--   3. null user_id       -> excluded here, COUNTED in the audit model
--
-- The natural key is (user_id, event_name, event_ts) and NOT event_id, because
-- the generator assigns a fresh event_id to the duplicate rows -- exactly like a
-- retried client beacon that generates a new request id. Deduplicating on the
-- surrogate key would find nothing and quietly leave every count inflated.

WITH raw AS (
    SELECT
        event_id,
        user_id,
        event_name,
        -- Undo the planted timezone bug. ios clients stamped events in local
        -- time (UTC-8) instead of UTC, which shifts events across midnight and
        -- therefore across cohort-day boundaries -- it distorts D1 retention
        -- much more than it distorts any all-time total.
        CASE WHEN platform = 'ios'
             THEN event_ts + INTERVAL 8 HOUR
             ELSE event_ts
        END AS event_ts,
        channel,
        platform,
        revenue
    FROM raw_events
    WHERE user_id IS NOT NULL          -- consent-blocked clients; counted in audit_data_quality
),

deduplicated AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY user_id, event_name, event_ts
            ORDER BY event_id
        ) AS occurrence
    FROM raw
)

SELECT
    event_id,
    user_id,
    event_name,
    event_ts,
    CAST(event_ts AS DATE) AS event_date,
    channel,
    platform,
    revenue
FROM deduplicated
WHERE occurrence = 1
