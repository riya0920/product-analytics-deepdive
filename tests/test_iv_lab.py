"""IV grading harness, pinned to its planted truth.

Pins the four behaviours that matter: OLS is biased by the confounder, 2SLS
recovers beta with a strong valid instrument, a weak instrument is flagged by a
low first-stage F, and a violated exclusion restriction biases 2SLS even when the
first-stage F is strong.
"""
from analytics import iv_lab as IV


def test_ols_is_biased_by_the_confounder():
    d = IV.simulate(beta=2.0, confounding=1.5, seed=0)
    est = IV.grade(IV.ols(d), 2.0)
    assert est["bias"] > 0.2, est          # confounding pushes OLS up
    assert est["covers_truth"] is False, est


def test_2sls_recovers_beta_with_strong_valid_instrument():
    d = IV.simulate(beta=2.0, instrument_strength=1.0, exclusion_violation=0.0, seed=0)
    est = IV.grade(IV.tsls(d), 2.0)
    assert abs(est["bias"]) < 0.2, est
    assert est["covers_truth"] is True, est
    assert est["first_stage_F"] > 10, est   # strong instrument


def test_weak_instrument_is_flagged_by_low_first_stage_F():
    d = IV.simulate(beta=2.0, instrument_strength=0.05, seed=0)
    F = IV.first_stage_F(d)
    assert F < 10, F                        # Staiger-Stock: weak below ~10
    # and 2SLS is unreliable -- a large SE relative to the point estimate
    est = IV.tsls(d)
    assert est["se"] > 0.3, est


def test_exclusion_violation_biases_2sls_despite_strong_F():
    d = IV.simulate(beta=2.0, instrument_strength=1.0, exclusion_violation=1.5, seed=0)
    est = IV.grade(IV.tsls(d), 2.0)
    assert est["first_stage_F"] > 10, est   # the first stage still looks strong
    assert est["bias"] > 0.5, est           # but 2SLS is biased by the direct path
    assert est["covers_truth"] is False, est
