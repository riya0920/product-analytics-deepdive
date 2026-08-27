"""The LaLonde grading harness, pinned against published values.

These tests exist because the whole point of this module is that it grades other
things. A grader whose own numbers drift is worse than no grader, so the two
anchors here are values that appear in the literature: the experimental effect of
$1,794 and the fact that the naive PSID comparison gets the sign wrong.
"""
import os

import pytest

from analytics import lalonde_lab as L

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(L.DATA, L.FILES["treated"])),
    reason="LaLonde data not downloaded; run make lalonde-data",
)


def test_sample_sizes_match_dehejia_wahba():
    assert len(L.load("treated")) == 185
    assert len(L.load("experimental_control")) == 260
    assert len(L.load("psid")) == 2490
    assert len(L.load("cps")) == 15992


def test_experimental_effect_matches_published_value():
    truth = L.experimental_truth()
    assert 1700 < truth["estimate"] < 1900, truth["estimate"]
    assert truth["ci_low"] > 0, "the experimental effect should be distinguishable from zero"


def test_naive_psid_comparison_gets_the_sign_wrong():
    """The headline failure. Not merely imprecise: confidently negative."""
    t, c = L.load("treated"), L.load("psid")
    est = L.naive(t, c)
    assert est["estimate"] < -10000
    assert est["ci_high"] < 0, "and its interval does not even include zero"


def test_harness_recovers_truth_on_the_experimental_control():
    """Sanity check on the grader itself.

    Against the real randomised control every estimator should land near the
    truth. If one does not, the estimator is broken rather than the data being
    hard, and that has to fail here before any result on PSID or CPS is believed.
    """
    truth = L.experimental_truth()
    t, c = L.load("treated"), L.load("experimental_control")
    for f in L.ESTIMATORS:
        got = L.grade(f(t, c), truth)
        assert got["sign_correct"], got["method"]
        assert got["covers_truth"], (got["method"], got["estimate"])


def test_overlap_is_much_worse_for_the_survey_controls():
    t = L.load("treated")
    good = L.overlap_report(t, L.load("experimental_control"))
    bad = L.overlap_report(t, L.load("psid"))
    assert bad["n_covariates_imbalanced"] > good["n_covariates_imbalanced"]
    assert bad["controls_above_treated_min"] < good["controls_above_treated_min"]
    assert abs(bad["worst_smd"]) > abs(good["worst_smd"])


def test_matching_reports_who_it_could_not_match():
    """A matched estimate without an unmatched count is unreadable: it silently
    changes which population the number is about."""
    t, c = L.load("treated"), L.load("psid")
    est = L.psm(t, c)
    assert "unmatched_treated" in est and "matched_pairs" in est
    assert est["matched_pairs"] + est["unmatched_treated"] == len(t)
