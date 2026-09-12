"""DiD grading harness, pinned to its planted truth.

The point of this module is that it grades DiD against a known effect, so these
tests pin the two behaviours that matter: DiD recovers the effect when parallel
trends holds, and it is biased (and the placebo catches it) when they do not.
"""
from analytics import did_lab as D


def test_recovers_tau_under_parallel_trends():
    df = D.simulate_panel(tau=3.0, cutoff=4, pretrend_gap=0.0, seed=0)
    est = D.grade(D.twfe_did(df), 3.0)
    assert abs(est["bias"]) < 0.5, est
    assert est["covers_truth"] is True, est
    placebo = D.parallel_trends_placebo(df, 4)["estimate"]
    assert abs(placebo) < 0.6, ("no pre-trend expected when parallel trends holds", placebo)


def test_biased_and_placebo_fires_under_violation():
    df = D.simulate_panel(tau=3.0, cutoff=4, pretrend_gap=0.6, seed=0)
    est = D.grade(D.twfe_did(df), 3.0)
    # a real pre-trend pushes the estimate away from tau; the CI should miss it
    assert est["bias"] > 1.0, est
    assert est["covers_truth"] is False, est
    placebo = D.parallel_trends_placebo(df, 4)["estimate"]
    assert abs(placebo) > 0.5, ("the placebo must detect the pre-trend", placebo)


def test_2x2_and_twfe_agree_on_balanced_panel():
    df = D.simulate_panel(tau=3.0, cutoff=4, pretrend_gap=0.0, seed=1)
    a = D.did_2x2(df)["estimate"]
    b = D.twfe_did(df)["estimate"]
    assert abs(a - b) < 1e-6, (a, b)


def test_cluster_robust_se_is_positive_and_finite():
    df = D.simulate_panel(seed=2)
    est = D.twfe_did(df)
    assert est["se"] > 0 and est["se"] == est["se"]
