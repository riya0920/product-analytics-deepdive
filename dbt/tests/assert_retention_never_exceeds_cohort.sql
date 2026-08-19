-- FAILS if a retention count exceeds its cohort size, which is the signature of
-- double-counting users who were active multiple times on the same day.
SELECT signup_date, channel, cohort_users, d1_plus, d7_plus, d30_plus
FROM {{ ref('mart_retention') }}
WHERE d1_plus > cohort_users
   OR d7_plus > cohort_users
   OR d30_plus > cohort_users
