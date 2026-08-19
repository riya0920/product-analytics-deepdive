-- FAILS if two platforms' events cluster at materially different times of day.
-- This is the test that catches the planted timezone bug.
--
-- Why this shape of test: a uniform per-platform timestamp shift preserves event
-- ORDER, so every ordering assertion (signup-is-first, monotonic funnel) passes
-- while each cohort-day boundary is silently wrong. The defect is only visible
-- in the DISTRIBUTION of time-of-day.
--
-- Why a CIRCULAR mean and not AVG(hour): hour-of-day wraps at midnight, so the
-- arithmetic mean is not a valid summary of it. With an evening-peaked traffic
-- profile, an 8-hour shift moves late-night hours across the midnight boundary
-- and the arithmetic means differ by only ~2.4 hours -- enough to hide the bug
-- under any sensible threshold. The circular mean (atan2 of the mean unit
-- vector) recovers the true 8-hour separation.
--
-- Threshold: 3 hours between any two platforms. Genuine platform differences in
-- usage time are real (web skews to working hours) but are well inside that.

WITH platform_vectors AS (
    SELECT
        platform,
        COUNT(*) AS n,
        AVG(SIN(2 * PI() * EXTRACT(hour FROM event_ts) / 24.0)) AS mean_sin,
        AVG(COS(2 * PI() * EXTRACT(hour FROM event_ts) / 24.0)) AS mean_cos
    FROM {{ ref('stg_events') }}
    GROUP BY platform
    HAVING COUNT(*) > 1000
),

circular_means AS (
    SELECT
        platform,
        n,
        -- atan2 -> radians -> hours, normalised into [0, 24)
        MOD(ATAN2(mean_sin, mean_cos) * 24.0 / (2 * PI()) + 24.0, 24.0) AS mean_hour
    FROM platform_vectors
),

pairs AS (
    SELECT
        a.platform AS platform_a,
        b.platform AS platform_b,
        ROUND(a.mean_hour, 2) AS mean_hour_a,
        ROUND(b.mean_hour, 2) AS mean_hour_b,
        -- circular distance: never more than 12 hours apart
        ROUND(
            LEAST(
                MOD(ABS(a.mean_hour - b.mean_hour), 24.0),
                24.0 - MOD(ABS(a.mean_hour - b.mean_hour), 24.0)
            ), 2
        ) AS hours_apart
    FROM circular_means a
    JOIN circular_means b ON a.platform < b.platform
)

SELECT *
FROM pairs
WHERE hours_apart > 3.0
