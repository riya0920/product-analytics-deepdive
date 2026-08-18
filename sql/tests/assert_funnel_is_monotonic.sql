-- FAILS if a later funnel step has MORE users than an earlier one, which would
-- mean users are reaching purchase without signing up -- i.e. the join is wrong.
SELECT step_index, step, users_reached, users_previous_step
FROM mart_funnel
WHERE users_previous_step IS NOT NULL
  AND users_reached > users_previous_step
