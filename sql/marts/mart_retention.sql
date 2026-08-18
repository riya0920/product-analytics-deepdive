-- mart_retention: D1/D7/D30 retention by signup cohort.
--
-- The definition used here is UNBOUNDED ("active on day N or later") for the
-- classic curve, plus an EXACT-DAY variant, because the two disagree and papers
-- over that disagreement are how retention numbers stop being comparable
-- between teams. Both are emitted; the memo says which one it quotes.
--
-- The double-counting trap: a user active three times on day 7 must count once.
-- COUNT(DISTINCT user_id) inside the day filter is what prevents that, and the
-- test suite asserts retention never exceeds cohort size.

WITH activity AS (
    SELECT DISTINCT
        e.user_id,
        u.signup_date,
        u.channel,
        u.platform,
        CAST(e.event_ts AS DATE) AS active_date,
        DATE_DIFF('day', u.signup_date, CAST(e.event_ts AS DATE)) AS day_number
    FROM stg_events e
    JOIN stg_users u USING (user_id)
    WHERE CAST(e.event_ts AS DATE) >= u.signup_date
),

cohort_size AS (
    SELECT signup_date, channel, COUNT(DISTINCT user_id) AS cohort_users
    FROM stg_users
    GROUP BY signup_date, channel
)

SELECT
    c.signup_date,
    c.channel,
    c.cohort_users,
    COUNT(DISTINCT CASE WHEN a.day_number = 1 THEN a.user_id END) AS d1_exact,
    COUNT(DISTINCT CASE WHEN a.day_number = 7 THEN a.user_id END) AS d7_exact,
    COUNT(DISTINCT CASE WHEN a.day_number >= 1 THEN a.user_id END) AS d1_plus,
    COUNT(DISTINCT CASE WHEN a.day_number >= 7 THEN a.user_id END) AS d7_plus,
    COUNT(DISTINCT CASE WHEN a.day_number >= 30 THEN a.user_id END) AS d30_plus,
    ROUND(COUNT(DISTINCT CASE WHEN a.day_number = 1 THEN a.user_id END)::DOUBLE
          / NULLIF(c.cohort_users, 0), 4) AS d1_exact_rate,
    ROUND(COUNT(DISTINCT CASE WHEN a.day_number >= 7 THEN a.user_id END)::DOUBLE
          / NULLIF(c.cohort_users, 0), 4) AS d7_plus_rate,
    ROUND(COUNT(DISTINCT CASE WHEN a.day_number >= 30 THEN a.user_id END)::DOUBLE
          / NULLIF(c.cohort_users, 0), 4) AS d30_plus_rate
FROM cohort_size c
LEFT JOIN activity a
       ON a.signup_date = c.signup_date
      AND a.channel = c.channel
GROUP BY c.signup_date, c.channel, c.cohort_users
ORDER BY c.signup_date, c.channel
