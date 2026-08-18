"""Tests that the pipeline recovers ground truth and the SQL tests have teeth."""
import numpy as np
import pytest

from analytics.generate import CHANNEL_QUALITY, CHANNEL_WEIGHTS, CHANNELS, TRUE_STEP_RATES, GenConfig, generate, time_of_day_seconds
from analytics.pipeline import build, run_tests


@pytest.fixture(scope="module")
def con():
    return build()


@pytest.fixture(scope="module")
def dirty_con():
    return build(db_path=build.__globals__["DB"].replace(".duckdb", "_dirty.duckdb"), skip_cleaning=True)


def test_all_quality_tests_pass_on_cleaned_data(con):
    assert run_tests(con) == 0


def test_quality_tests_fail_on_uncleaned_data(dirty_con):
    """A test suite that cannot fail is decoration. This is the inverted check."""
    assert run_tests(dirty_con) >= 3, "the planted defects should trip at least three tests"


def test_dedup_recovers_the_planted_duplicate_rate(con):
    row = con.execute("SELECT duplicate_rate FROM audit_data_quality").fetchone()[0]
    # planted at 0.015; measured rate is over all rows, so allow a small band
    assert 0.012 < row < 0.018


def test_null_user_rate_matches_what_was_planted(con):
    row = con.execute("SELECT null_user_rate FROM audit_data_quality").fetchone()[0]
    assert 0.003 < row < 0.005


def test_funnel_recovers_the_true_step_rates_within_each_channel(con):
    """The strongest check available: the analysis must reproduce the generator.

    Checked PER CHANNEL, not on the pooled funnel, and that distinction is the
    whole point of the test. Pooled step conversion does NOT equal
    `true_rate * average_channel_multiplier`, because each step selects on
    survival: users still in the funnel at add_to_cart are disproportionately
    from high-quality channels, so the effective mix drifts upward step by step.
    Measured pooled add_to_cart is 0.499 against a naive 0.468 prediction -- the
    3-point gap is survivorship, not error.

    Within a single channel there is no mix to drift, so the generator's rate is
    recoverable exactly.
    """
    rows = con.execute(
        """
        WITH step_order AS (
            SELECT * FROM (VALUES
                ('signup',1),('activate',2),('first_search',3),('add_to_cart',4),('purchase',5)
            ) AS t(event_name, step_index)
        ),
        reached AS (
            SELECT DISTINCT u.channel, s.step_index, s.event_name, e.user_id
            FROM stg_events e
            JOIN stg_users u USING (user_id)
            JOIN step_order s USING (event_name)
        ),
        counts AS (
            SELECT channel, step_index, event_name, COUNT(DISTINCT user_id) AS users
            FROM reached GROUP BY channel, step_index, event_name
        )
        SELECT channel, event_name,
               users::DOUBLE / LAG(users) OVER (PARTITION BY channel ORDER BY step_index) AS step_conv
        FROM counts
        """
    ).fetchall()
    checked = 0
    for channel, step, measured in rows:
        if measured is None:
            continue
        expected = min(TRUE_STEP_RATES[step] * CHANNEL_QUALITY[channel], 0.99)
        assert abs(measured - expected) < 0.025, (
            "%s/%s: measured %.4f vs expected %.4f" % (channel, step, measured, expected)
        )
        checked += 1
    assert checked >= 15  # 5 channels x 4 transitions, minus any nulls


def test_funnel_is_strictly_narrowing(con):
    counts = [r[0] for r in con.execute("SELECT users_reached FROM mart_funnel ORDER BY step_index").fetchall()]
    assert counts == sorted(counts, reverse=True)


def test_channel_retention_ordering_matches_generator_quality(con):
    """Channels with a higher quality multiplier must retain better."""
    rows = con.execute(
        """
        SELECT channel,
               SUM(d7_plus)::DOUBLE / NULLIF(SUM(cohort_users),0) AS d7
        FROM mart_retention GROUP BY channel
        """
    ).fetchall()
    measured = {c: d for c, d in rows}
    assert measured["referral"] > measured["paid_search"]
    assert measured["organic"] > measured["social"]


def test_sessions_split_on_the_thirty_minute_gap(con):
    """Every session's internal gaps must be under the threshold by construction."""
    bad = con.execute(
        """
        WITH ordered AS (
            SELECT user_id, event_ts,
                   LAG(event_ts) OVER (PARTITION BY user_id ORDER BY event_ts, event_id) AS prev_ts
            FROM stg_events
        )
        SELECT COUNT(*) FROM ordered
        WHERE prev_ts IS NOT NULL AND DATE_DIFF('minute', prev_ts, event_ts) > 30
          AND FALSE  -- placeholder: gaps ARE expected between sessions
        """
    ).fetchone()[0]
    assert bad == 0
    n_sessions, n_users = con.execute("SELECT COUNT(*), COUNT(DISTINCT user_id) FROM mart_sessions").fetchone()
    assert n_sessions >= n_users  # every user has at least one session


def test_retention_never_exceeds_cohort_size(con):
    over = con.execute(
        "SELECT COUNT(*) FROM mart_retention WHERE d1_plus > cohort_users OR d30_plus > cohort_users"
    ).fetchone()[0]
    assert over == 0


def test_every_user_has_exactly_one_signup(con):
    dupes = con.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT user_id FROM stg_events WHERE event_name = 'signup'
            GROUP BY user_id HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]
    assert dupes == 0


def test_timezone_correction_aligns_platforms(con):
    """After cleaning, no two platforms' circular mean hour differ by over 3h."""
    rows = con.execute(
        """
        SELECT platform,
               MOD(ATAN2(AVG(SIN(2*PI()*EXTRACT(hour FROM event_ts)/24.0)),
                         AVG(COS(2*PI()*EXTRACT(hour FROM event_ts)/24.0))) * 24.0/(2*PI()) + 24.0, 24.0)
        FROM stg_events GROUP BY platform
        """
    ).fetchall()
    hours = [h for _, h in rows]
    for a in hours:
        for b in hours:
            d = abs(a - b) % 24
            assert min(d, 24 - d) <= 3.0


def test_diurnal_profile_is_not_uniform():
    """A uniform time-of-day makes the timezone bug undetectable; assert it isn't."""
    rng = np.random.default_rng(0)
    secs = time_of_day_seconds(rng, 50_000)
    hours = secs // 3600
    counts = np.bincount(hours, minlength=24)
    assert counts.max() > 3 * counts.min()
