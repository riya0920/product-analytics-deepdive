"""Tests for segmentation and the incrementality study.

The theme is the same as the rest of this repo: assert against something known.
For segmentation that means the planted timezone bug — the tests check it
manufactures a platform effect on a fragile metric and not on a robust one. For
incrementality it means the simulated world's `tau`, which lets the estimators be
scored rather than admired.
"""
import math
import os

import numpy as np
import pytest

from analytics import incrementality as inc
from analytics import segments as seg

DB = seg.DB
DIRTY = seg.DIRTY_DB

pytestmark = pytest.mark.skipif(
    not os.path.exists(DB), reason="warehouse not built; run `make generate build` first")


@pytest.fixture(scope="module")
def con():
    import duckdb
    c = duckdb.connect(DB, read_only=True)
    yield c
    c.close()


# --- Wilson ----------------------------------------------------------------

def test_wilson_stays_inside_zero_one_at_the_extremes():
    """The reason this is not the normal approximation."""
    lo, hi = seg.wilson(0, 40)
    assert lo == 0.0 and 0 < hi < 1
    lo, hi = seg.wilson(40, 40)
    assert hi == 1.0 and 0 < lo < 1


def test_wilson_narrows_as_n_grows():
    w_small = seg.wilson(50, 100)
    w_big = seg.wilson(5000, 10_000)
    assert (w_big[1] - w_big[0]) < (w_small[1] - w_small[0]) / 5


# --- segmentation ----------------------------------------------------------

def test_thin_cells_are_flagged_not_silently_reported(con):
    cells = seg._cells(con, ["channel", "platform"])
    assert cells
    for c in cells:
        assert c["thin"] == (c["n"] < seg.MIN_CELL)


def test_paid_search_gap_holds_inside_every_platform(con):
    """The generator makes channel quality independent of platform, so a gap that
    appeared only in aggregate would be a bug in this analysis rather than a
    finding about the product."""
    r = seg.gap_within_platforms(con)
    assert r["overall_gap"]["diff"] < 0
    usable = [p for p in r["per_platform"] if "diff" in p]
    assert len(usable) == 3
    assert all(p["diff"] < 0 and p["significant"] for p in usable)
    assert r["gap_holds_within_every_platform"]


def test_no_simpson_reversal_on_this_data(con):
    """Asserted rather than assumed. If a future generator change introduces one,
    this test is how it gets noticed."""
    assert seg.simpson_check(con)["reversals_found"] == 0


def test_cohort_trend_is_tested_not_eyeballed(con):
    """The generator plants no cohort trend, so a significant one would be a false
    finding. The first version of this analysis used an unweighted slope against a
    fixed 2-point threshold and reported a trend that was not there."""
    r = seg.cohort_over_cohort(con)
    assert r["trend"] is not None
    assert "se_per_week" in r["trend"]
    assert not r["trend"]["significant"], "reported a cohort trend the generator never planted"


@pytest.mark.skipif(not os.path.exists(DIRTY), reason="dirty warehouse not built")
def test_the_timezone_bug_manufactures_a_platform_effect_on_a_fragile_metric():
    """The headline of the module, asserted both ways.

    On an exact-day metric the uncleaned warehouse invents a multi-point iOS
    deficit. On an unbounded metric the same bug is invisible. Both are checked,
    because "the dashboard looks fine" is only reassuring if you know which
    dashboard.
    """
    fragile = seg.dirty_vs_clean(metric="d1_exact")
    robust = seg.dirty_vs_clean(metric="d7_plus")

    assert abs(fragile["manufactured_effect"]) > 0.02
    assert fragile["uncleaned"]["focus_vs_rest"]["significant"]
    assert not fragile["clean"]["focus_vs_rest"]["significant"]

    assert abs(robust["manufactured_effect"]) < 0.01


# --- incrementality --------------------------------------------------------

def test_the_simulated_world_has_the_confounding_it_claims():
    w = inc.simulate_world(inc.WorldConfig(tau=0.0, n_users=20_000))
    # Intent must drive BOTH channel and retention, or there is no confounding to
    # study and every estimator below would trivially be unbiased.
    assert np.corrcoef(w["intent"], w["paid"])[0, 1] < -0.3
    assert np.corrcoef(w["intent"], w["retained"])[0, 1] > 0.2


def test_zero_tau_means_zero_true_effect():
    assert abs(inc.true_effect(inc.WorldConfig(tau=0.0), n=50_000)["risk_difference"]) < 1e-9


def test_the_raw_gap_is_large_even_when_the_true_effect_is_zero():
    """The whole problem, in one assertion."""
    w = inc.simulate_world(inc.WorldConfig(tau=0.0, n_users=40_000))
    assert inc.naive_estimate(w)["risk_difference"] < -0.10


def test_adjustment_shrinks_the_bias_but_does_not_remove_it():
    cfg = inc.WorldConfig(tau=0.0, n_users=40_000)
    w = inc.simulate_world(cfg)
    raw = abs(inc.naive_estimate(w)["risk_difference"])
    adj = abs(inc.regression_adjusted(w)["risk_difference"])
    assert adj < raw * 0.85           # it helps
    assert adj > 0.05                 # and it is nowhere near enough


def test_ipw_and_regression_agree_here():
    """Two estimators with different assumptions landing in the same place is
    weak evidence the outcome model is not wildly misspecified. It is not
    evidence about unmeasured confounding, which is what actually matters."""
    w = inc.simulate_world(inc.WorldConfig(tau=-0.55, n_users=40_000))
    a = inc.regression_adjusted(w)["risk_difference"]
    b = inc.ipw_estimate(w)["risk_difference"]
    assert abs(a - b) < 0.02


def test_g_computation_is_not_the_log_odds_coefficient():
    """Quoting the coefficient as a risk difference is a unit error that inflates
    the effect several-fold; the standardised contrast is the reportable one."""
    w = inc.simulate_world(inc.WorldConfig(tau=-0.55, n_users=30_000))
    r = inc.regression_adjusted(w)
    assert abs(r["coefficient_log_odds"]) > 2 * abs(r["risk_difference"])


def test_e_value_is_one_when_there_is_no_association():
    assert inc.e_value(1.0) == pytest.approx(1.0)


def test_e_value_is_symmetric_in_direction():
    assert inc.e_value(2.0) == pytest.approx(inc.e_value(0.5))


def test_e_value_grows_with_the_association():
    assert inc.e_value(3.0) > inc.e_value(2.0) > inc.e_value(1.5)


def test_bundled_benchmark_is_stronger_than_any_single_proxy():
    """The reason the single-proxy benchmark is the wrong comparison."""
    w = inc.simulate_world(inc.WorldConfig(tau=0.0, n_users=40_000))
    m = inc.measured_confounder_strengths(w)
    assert m["bundled_measured_rr"] > m["strongest_single_proxy_rr"]


def test_the_sensitivity_analysis_fails_to_separate_the_two_worlds():
    """The module's negative result, pinned so it cannot quietly be dropped.

    In the world where the true causal effect is exactly zero, the E-value still
    clears its benchmark. A threshold rule on it calls pure selection causal.
    """
    pure = inc.assess(inc.WorldConfig(tau=0.0, n_users=60_000))
    real = inc.assess(inc.WorldConfig(tau=-0.55, n_users=60_000))
    assert pure["ground_truth"]["risk_difference"] == pytest.approx(0.0, abs=1e-9)
    assert pure["e_value_over_bundled_benchmark"] > 1.0, "the failure this module reports"
    assert real["e_value_over_bundled_benchmark"] > pure["e_value_over_bundled_benchmark"]


# --- the geo design --------------------------------------------------------

def test_geo_power_depends_on_geo_count_not_user_count():
    """The number people get wrong. Ten times the users, identical MDE."""
    a = inc.geo_holdout_design(n_geos=60, users_per_geo_per_week=900)
    b = inc.geo_holdout_design(n_geos=60, users_per_geo_per_week=9_000)
    assert a["mde_at_current_geo_count"] == pytest.approx(b["mde_at_current_geo_count"])
    c = inc.geo_holdout_design(n_geos=240, users_per_geo_per_week=900)
    assert c["mde_at_current_geo_count"] < a["mde_at_current_geo_count"] / 1.9


def test_the_design_admits_it_is_underpowered():
    d = inc.geo_holdout_design()
    assert not d["feasible_with_available_geos"]
    assert d["geos_needed_for_target_lift"] > d["n_geos_available"]


def test_pre_period_adjustment_reduces_the_geos_needed_monotonically():
    d = inc.geo_holdout_design()
    needed = [d["cuped_on_pre_period"][k]["geos_needed_for_target"]
              for k in sorted(d["cuped_on_pre_period"])]
    assert needed == sorted(needed, reverse=True)


def test_the_solved_rho_actually_makes_the_study_feasible():
    d = inc.geo_holdout_design()
    rho = d["min_pre_period_rho_to_hit_target"]
    z = 1.959963985 + 0.8416212336
    needed = math.ceil(((z * d["between_geo_sd"] * math.sqrt(1 - rho ** 2)) / d["target_lift"]) ** 2 * 4)
    assert needed <= d["n_geos_available"] + 1
