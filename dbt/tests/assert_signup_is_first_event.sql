-- FAILS if any user has an event before their signup event: a timezone bug that
-- was only PARTIALLY corrected shows up here, because the ios shift would move
-- some events before the (unshifted) signup of the same user.
WITH first_events AS (
    SELECT user_id, MIN(event_ts) AS first_ts FROM {{ ref('stg_events') }} GROUP BY user_id
),
signups AS (
    SELECT user_id, MIN(event_ts) AS signup_ts
    FROM {{ ref('stg_events') }} WHERE event_name = 'signup' GROUP BY user_id
)
SELECT s.user_id, s.signup_ts, f.first_ts
FROM signups s JOIN first_events f USING (user_id)
WHERE f.first_ts < s.signup_ts
