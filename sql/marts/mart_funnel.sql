-- mart_funnel: per-step conversion with drop-off quantified in users AND dollars.
--
-- Two decisions that change the answer:
--
-- 1. Steps are counted as "ever reached", not "reached in one session". For an
--    onboarding funnel measured over 120 days, a session-scoped funnel would
--    report a drop-off that is really just users coming back tomorrow.
--
-- 2. The revenue attributed to a step is the revenue of users who reached it,
--    so "value lost at step N" is a real forgone amount, not a rate multiplied
--    by an average. The memo's dollar figure comes from here.

WITH step_order AS (
    SELECT * FROM (VALUES
        ('signup', 1), ('activate', 2), ('first_search', 3), ('add_to_cart', 4), ('purchase', 5)
    ) AS t(event_name, step_index)
),

user_steps AS (
    SELECT
        e.user_id,
        s.event_name,
        s.step_index,
        MIN(e.event_ts) AS first_reached_at
    FROM stg_events e
    JOIN step_order s USING (event_name)
    GROUP BY e.user_id, s.event_name, s.step_index
),

revenue_per_user AS (
    SELECT user_id, SUM(revenue) AS user_revenue
    FROM stg_events
    GROUP BY user_id
),

step_totals AS (
    SELECT
        us.step_index,
        us.event_name,
        COUNT(DISTINCT us.user_id) AS users_reached,
        SUM(COALESCE(r.user_revenue, 0)) AS revenue_from_reachers
    FROM user_steps us
    LEFT JOIN revenue_per_user r USING (user_id)
    GROUP BY us.step_index, us.event_name
)

SELECT
    step_index,
    event_name AS step,
    users_reached,
    -- LAG gives the previous step's population: the denominator for step-through.
    LAG(users_reached) OVER (ORDER BY step_index) AS users_previous_step,
    ROUND(
        users_reached::DOUBLE
        / NULLIF(LAG(users_reached) OVER (ORDER BY step_index), 0), 4
    ) AS step_conversion,
    ROUND(
        users_reached::DOUBLE
        / NULLIF(FIRST_VALUE(users_reached) OVER (ORDER BY step_index), 0), 4
    ) AS cumulative_conversion,
    LAG(users_reached) OVER (ORDER BY step_index) - users_reached AS users_lost,
    -- Average revenue per user who reached the FINAL step, applied to the users
    -- lost here. Stated as an upper bound in the memo: it assumes a recovered
    -- user converts like a user who already made it, which they will not.
    ROUND(
        (LAG(users_reached) OVER (ORDER BY step_index) - users_reached)
        * (SELECT revenue_from_reachers / NULLIF(users_reached, 0)
           FROM step_totals WHERE event_name = 'purchase'), 2
    ) AS max_recoverable_revenue
FROM step_totals
ORDER BY step_index
