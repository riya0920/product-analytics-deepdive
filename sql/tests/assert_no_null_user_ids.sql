-- FAILS if a null user_id survived into staging.
SELECT COUNT(*) AS null_rows FROM stg_events WHERE user_id IS NULL HAVING COUNT(*) > 0
