-- FAILS if any (user_id, event_name, event_ts) appears twice in staging.
-- This is the test that catches the planted duplicate-beacon defect.
SELECT user_id, event_name, event_ts, COUNT(*) AS n
FROM stg_events
GROUP BY user_id, event_name, event_ts
HAVING COUNT(*) > 1
