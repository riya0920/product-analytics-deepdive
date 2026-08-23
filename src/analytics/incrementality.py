"""Incrementality: is paid search *causing* the retention gap, or selecting it?

    python -m analytics.incrementality validate
    python -m analytics.incrementality design

This is the memo's largest open question and the honest answer in the memo is
"this data cannot settle it". That answer is correct and it is also not good
enough on its own, because the next question is always *"so how wrong could we
be, and what would it take to find out?"* This module answers both.

## Why the warehouse cannot answer it

Channel is not assigned; it is chosen. Users who arrive through a paid ad differ
from users who type the name into a search bar in ways that were true **before**
they ever saw the ad, and those ways also drive retention. So the observed gap is

    observed gap  =  causal effect of the channel  +  selection

and no amount of SQL over this table separates the two terms. Adjusting for
everything observable shrinks the second term. It never proves the second term
reached zero, because the thing doing the confounding is user intent and nobody
logs intent.

## What this module does instead

1. Simulates the situation with a **known** causal effect `tau`, so both terms are
   separable by construction and every estimator below can be scored.
2. Runs the three things an analyst would actually run: raw difference, regression
   adjustment on observed proxies, and inverse-propensity weighting.
3. Computes an **E-value** (VanderWeele & Ding 2017): the minimum strength, on the
   risk-ratio scale, that an unmeasured confounder would need with *both* channel
   and retention to explain the adjusted estimate away entirely.
4. Benchmarks that E-value against the confounders actually measured — both the
   strongest single one (the comparison people reach for, and too weak by
   construction) and all of them bundled (the defensible one).
5. Prices the experiment that would settle it — a geo holdout — including how long
   it has to run.

## The result that matters, and it is a negative one

The plan was that the E-value would separate a pure-selection world from a
real-effect one. **It does not.** Run with `tau = 0`, where the true causal effect
is exactly zero, and adjustment still leaves -12.0 points; the E-value for that
residual is 1.69 against a bundled benchmark of 1.65, so it *clears* the benchmark
in a world with no causal effect at all. Only the margin differs — 1.02x against
1.23x — and a margin with no sampling distribution is not something to set a
budget by.

That failure is the useful output. It is the argument for running the geo holdout
rather than the argument for a cleverer regression, and a validation that only ran
in the world where the method works would never have surfaced it.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS = os.path.join(ROOT, "results")


# ---------------------------------------------------------------------------
# a world where selection and causation are separable by construction
# ---------------------------------------------------------------------------

@dataclass
class WorldConfig:
    n_users: int = 60_000
    # True causal effect of arriving via paid search, on the log-odds of D7
    # retention. Zero means the whole observed gap is selection.
    tau: float = 0.0
    # How strongly latent intent drives retention, and how strongly it drives
    # ending up in the paid channel. Both are needed for confounding; either one
    # alone produces none.
    intent_on_retention: float = 1.05
    intent_on_channel: float = -1.20
    # How much of intent each observed proxy actually captures. The gap between
    # these and 1.0 is the unmeasured confounding, and it is the only reason the
    # question is hard.
    proxy_loadings: tuple = (0.62, 0.44, 0.30)
    proxy_noise: float = 1.0
    base_retention_logit: float = 0.8
    seed: int = 11


def simulate_world(cfg: WorldConfig) -> dict:
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_users

    intent = rng.normal(0, 1, n)

    # Observed proxies: noisy, partial views of intent. A real warehouse has
    # exactly this -- session depth, time-of-day, device age -- and none of them
    # is intent.
    X = np.column_stack([
        load * intent + cfg.proxy_noise * rng.normal(0, 1, n)
        for load in cfg.proxy_loadings
    ])

    p_paid = 1.0 / (1.0 + np.exp(-(cfg.intent_on_channel * intent)))
    paid = (rng.random(n) < p_paid).astype(float)

    logit = (cfg.base_retention_logit
             + cfg.intent_on_retention * intent
             + cfg.tau * paid)
    retained = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(float)

    return {"intent": intent, "X": X, "paid": paid, "retained": retained, "cfg": cfg}


def true_effect(cfg: WorldConfig, n: int = 400_000) -> dict:
    """The counterfactual contrast, computed by running the world both ways.

    This is what an experiment would recover, and it is not the same object as
    `tau`: tau is on the log-odds scale, the risk difference depends on where on
    the curve the population sits. Reporting tau as if it were the risk
    difference is a quiet unit error that makes every estimator below look biased.
    """
    rng = np.random.default_rng(cfg.seed + 999)
    intent = rng.normal(0, 1, n)
    base = cfg.base_retention_logit + cfg.intent_on_retention * intent
    p0 = 1.0 / (1.0 + np.exp(-base))
    p1 = 1.0 / (1.0 + np.exp(-(base + cfg.tau)))
    return {"risk_difference": float(p1.mean() - p0.mean()),
            "risk_ratio": float(p1.mean() / p0.mean()),
            "tau_log_odds": cfg.tau}


# ---------------------------------------------------------------------------
# estimators
# ---------------------------------------------------------------------------

def _logistic_irls(X: np.ndarray, y: np.ndarray, iters: int = 40, ridge: float = 1e-6):
    """Logistic regression by IRLS. Written out rather than imported.

    Fifteen lines, no new dependency, and the ridge term is the only thing that
    stops it exploding when a covariate is nearly separating -- which is worth
    seeing, because that failure is common in propensity models on real data and
    an off-the-shelf fit hides it behind a convergence warning nobody reads.
    """
    X = np.column_stack([np.ones(len(X)), X])
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = X @ beta
        p = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
        w = np.clip(p * (1 - p), 1e-8, None)
        z = eta + (y - p) / w
        XtW = X.T * w
        beta_new = np.linalg.solve(XtW @ X + ridge * np.eye(X.shape[1]), XtW @ z)
        if np.max(np.abs(beta_new - beta)) < 1e-9:
            beta = beta_new
            break
        beta = beta_new
    return beta


def naive_estimate(w: dict) -> dict:
    paid, ret = w["paid"], w["retained"]
    p1, p0 = ret[paid == 1].mean(), ret[paid == 0].mean()
    n1, n0 = int(paid.sum()), int((1 - paid).sum())
    se = math.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
    return {"method": "raw difference", "risk_difference": float(p1 - p0),
            "risk_ratio": float(p1 / p0), "se": se, "n_paid": n1, "n_other": n0}


def regression_adjusted(w: dict) -> dict:
    """G-computation on a logistic outcome model with the observed proxies.

    Standardised over the whole population rather than reported as a coefficient:
    a log-odds coefficient is not a risk difference, and quoting one as the other
    is how an adjusted 'effect' ends up two to three times its real size.
    """
    X, paid, ret = w["X"], w["paid"], w["retained"]
    design = np.column_stack([paid, X])
    beta = _logistic_irls(design, ret)

    def predict(t):
        d = np.column_stack([np.ones(len(X)), np.full(len(X), t), X])
        return 1.0 / (1.0 + np.exp(-np.clip(d @ beta, -30, 30)))

    p1, p0 = predict(1.0).mean(), predict(0.0).mean()
    return {"method": "regression adjustment (g-computation)",
            "risk_difference": float(p1 - p0), "risk_ratio": float(p1 / p0),
            "coefficient_log_odds": float(beta[1])}


def ipw_estimate(w: dict, trim: float = 0.02) -> dict:
    """Inverse-propensity weighting, with trimming and the overlap check.

    Trimming is not a nicety. A propensity of 0.004 produces a weight of 250 and
    one user then carries more of the estimate than a thousand others; the
    estimate becomes a report on that user. The trimmed fraction is returned so
    the cost of the fix is visible.
    """
    X, paid, ret = w["X"], w["paid"], w["retained"]
    ps = 1.0 / (1.0 + np.exp(-np.clip(np.column_stack([np.ones(len(X)), X])
                                      @ _logistic_irls(X, paid), -30, 30)))
    keep = (ps > trim) & (ps < 1 - trim)
    ps_k, paid_k, ret_k = ps[keep], paid[keep], ret[keep]

    wt = np.where(paid_k == 1, 1.0 / ps_k, 1.0 / (1.0 - ps_k))
    p1 = np.average(ret_k[paid_k == 1], weights=wt[paid_k == 1])
    p0 = np.average(ret_k[paid_k == 0], weights=wt[paid_k == 0])
    return {"method": "inverse-propensity weighting",
            "risk_difference": float(p1 - p0), "risk_ratio": float(p1 / p0),
            "trimmed_fraction": float(1 - keep.mean()),
            "max_weight": float(wt.max()),
            "propensity_range": [float(ps.min()), float(ps.max())]}


# ---------------------------------------------------------------------------
# sensitivity
# ---------------------------------------------------------------------------

def e_value(risk_ratio: float) -> float:
    """VanderWeele & Ding's E-value.

    The minimum strength of association, on the risk-ratio scale, that an
    unmeasured confounder would need with **both** the exposure and the outcome —
    above and beyond the measured covariates — to fully explain away an observed
    association.

    It is a lower bound on required confounding, not a p-value and not a
    probability that confounding exists. Its only job is to move the argument from
    "there might be confounders" (unfalsifiable) to "a confounder would have to be
    at least this strong, and here is the strongest one we can actually measure"
    (checkable).
    """
    rr = risk_ratio if risk_ratio >= 1 else 1.0 / risk_ratio
    return rr + math.sqrt(rr * (rr - 1))


def measured_confounder_strengths(w: dict) -> dict:
    """How strong are the confounders we DID measure?

    This is the benchmark that makes an E-value interpretable, and *which*
    benchmark you pick changes the answer:

      * **Strongest single proxy.** The obvious choice and the wrong one. A latent
        confounder is generally stronger than any individual noisy measurement of
        it, so a single covariate systematically understates what is plausible and
        every E-value clears it.
      * **The measured covariates bundled.** Fit the outcome model and the
        propensity model, take each fitted index, and compare its top quartile to
        its bottom quartile. That is the strength of everything measured acting
        together, which is the right thing to hold an unmeasured confounder up
        against.

    Both are returned, because the gap between them is the point.
    """
    X, paid, ret = w["X"], w["paid"], w["retained"]

    per_proxy = []
    for j in range(X.shape[1]):
        hi = X[:, j] > np.median(X[:, j])
        rr_outcome = ret[hi].mean() / max(ret[~hi].mean(), 1e-9)
        rr_exposure = paid[hi].mean() / max(paid[~hi].mean(), 1e-9)
        per_proxy.append({"proxy": "x%d" % j,
                          "rr_with_retention": float(max(rr_outcome, 1 / rr_outcome)),
                          "rr_with_channel": float(max(rr_exposure, 1 / rr_exposure))})
    strongest_single = max(min(o["rr_with_retention"], o["rr_with_channel"]) for o in per_proxy)

    # Bundled: the fitted indices, top quartile against bottom quartile.
    beta_out = _logistic_irls(np.column_stack([paid, X]), ret)
    idx_out = X @ beta_out[2:]
    beta_ps = _logistic_irls(X, paid)
    idx_ps = X @ beta_ps[1:]

    def quartile_rr(index, y):
        lo, hi = np.quantile(index, [0.25, 0.75])
        a, b = y[index > hi].mean(), y[index < lo].mean()
        rr = a / max(b, 1e-9)
        return float(max(rr, 1 / rr))

    bundled = min(quartile_rr(idx_out, ret), quartile_rr(idx_ps, paid))

    return {"per_proxy": per_proxy,
            "strongest_single_proxy_rr": float(strongest_single),
            "bundled_measured_rr": float(bundled),
            "note": ("the bundled figure is the one to compare an E-value against; the single-proxy "
                     "figure is included because it is the comparison people reach for and it is "
                     "too weak by construction")}


def assess(cfg: WorldConfig) -> dict:
    w = simulate_world(cfg)
    truth = true_effect(cfg)
    naive = naive_estimate(w)
    adj = regression_adjusted(w)
    ipw = ipw_estimate(w)
    meas = measured_confounder_strengths(w)

    ev = e_value(adj["risk_ratio"])
    margin = ev / meas["bundled_measured_rr"]

    return {
        "config": {"tau_log_odds": cfg.tau, "n_users": cfg.n_users},
        "ground_truth": truth,
        "estimates": [naive, adj, ipw],
        "residual_bias_after_adjustment": adj["risk_difference"] - truth["risk_difference"],
        "selection_share_of_raw_gap": (
            float(1 - (truth["risk_difference"] / naive["risk_difference"]))
            if abs(naive["risk_difference"]) > 1e-9 else float("nan")),
        "e_value_for_adjusted_estimate": ev,
        "measured_confounders": meas,
        # Deliberately a MARGIN and not a verdict. See validate() for why a
        # threshold rule on this number is not defensible.
        "e_value_over_bundled_benchmark": margin,
        "reading": ("an unmeasured confounder would need RR>=%.2f with both channel and retention "
                    "to explain the adjusted estimate away. Everything measured here, bundled, "
                    "reaches %.2f -- a margin of %.2fx."
                    % (ev, meas["bundled_measured_rr"], margin)),
    }


def validate() -> dict:
    """Score the whole apparatus in two worlds where the answer is known.

    Without this the E-value is a formula with no demonstrated behaviour. With
    it, there is a measured negative result, and it is the most useful thing in
    this module.

    ## What was expected

    That the E-value would clear the benchmark in the real-effect world and fail
    to clear it in the pure-selection world, giving a usable decision rule.

    ## What actually happened

    It clears the benchmark in **both**. In the pure-selection world -- where the
    true causal effect is exactly zero and the entire -18.4 point raw gap is
    selection -- adjustment still leaves -12.0 points, and the E-value for that
    residual is 1.69 against a bundled benchmark of 1.65. A rule of "E-value beats
    what we measured, therefore causal" calls a purely selected effect causal.

    What survives is the **margin**, not the verdict: 1.02x in the pure-selection
    world against 1.23x in the real one. That is a difference in degree, on a
    quantity with no sampling distribution attached, and it is not enough to
    decide a budget on.

    ## What that is worth knowing

    This is the argument for running the geo holdout rather than the argument for
    a cleverer regression. Sensitivity analysis bounds how wrong an observational
    claim could be; it does not convert one into a causal claim, and a validation
    that only ever ran in the world where the method works would never have shown
    that.
    """
    pure = assess(WorldConfig(tau=0.0))
    real = assess(WorldConfig(tau=-0.55))
    return {
        "pure_selection_world": pure,
        "real_effect_world": real,
        "comparison": {
            "true_rd": [pure["ground_truth"]["risk_difference"], real["ground_truth"]["risk_difference"]],
            "raw_rd": [pure["estimates"][0]["risk_difference"], real["estimates"][0]["risk_difference"]],
            "adjusted_rd": [pure["estimates"][1]["risk_difference"], real["estimates"][1]["risk_difference"]],
            "e_value": [pure["e_value_for_adjusted_estimate"], real["e_value_for_adjusted_estimate"]],
            "bundled_benchmark": [pure["measured_confounders"]["bundled_measured_rr"],
                                  real["measured_confounders"]["bundled_measured_rr"]],
            "margin": [pure["e_value_over_bundled_benchmark"], real["e_value_over_bundled_benchmark"]],
        },
        "negative_result": (
            "the E-value clears its benchmark in BOTH worlds, including the one where the true "
            "causal effect is exactly zero. A threshold rule on it would call pure selection "
            "causal. Only the margin differs (%.2fx vs %.2fx), and a margin with no sampling "
            "distribution is not a decision rule."
            % (pure["e_value_over_bundled_benchmark"], real["e_value_over_bundled_benchmark"])),
        "implication": ("run the geo holdout. Sensitivity analysis bounds how wrong an "
                        "observational claim could be; it does not turn one into a causal claim."),
    }


# ---------------------------------------------------------------------------
# the experiment that would settle it
# ---------------------------------------------------------------------------

def geo_holdout_design(n_geos: int = 60, users_per_geo_per_week: int = 900,
                       baseline_retention: float = 0.70,
                       between_geo_sd: float = 0.045,
                       target_lift: float = 0.02, alpha: float = 0.05,
                       power: float = 0.80) -> dict:
    """Price the paid-search geo holdout, honestly.

    The unit of randomisation is a **geo**, not a user, because that is where the
    intervention can be applied -- you cannot switch ads off for one user. That
    single fact costs almost all the power, and it is the number people get wrong:
    54,000 users a week sounds enormous, but with 60 geos the effective sample
    size is 60.

    Between-geo variance dominates. A geo's baseline retention varies by roughly
    4.5 points for reasons that have nothing to do with ads, and that variance --
    not the binomial noise inside a geo -- sets the MDE.
    """
    z_a = 1.959963985
    z_b = 0.8416212336

    # Per-arm geo count needed for the target lift, cluster-randomised.
    per_arm = n_geos / 2.0
    se_geo = between_geo_sd * math.sqrt(2.0 / per_arm)
    mde_now = (z_a + z_b) * se_geo

    needed_per_arm = ((z_a + z_b) * between_geo_sd / target_lift) ** 2 * 2.0
    needed_geos = math.ceil(needed_per_arm * 2)

    # Pre-period covariate adjustment (CUPED on the geo's own history) is the one
    # lever that helps without buying more geos.
    cuped = {("rho_%.2f" % rho): {
        "effective_sd": between_geo_sd * math.sqrt(1 - rho ** 2),
        "mde": (z_a + z_b) * between_geo_sd * math.sqrt(1 - rho ** 2) * math.sqrt(2.0 / per_arm),
        "geos_needed_for_target": math.ceil(
            ((z_a + z_b) * between_geo_sd * math.sqrt(1 - rho ** 2) / target_lift) ** 2 * 4)
    } for rho in (0.0, 0.5, 0.7, 0.85)}

    # The pre-period correlation that would make the available geos sufficient.
    # Solved rather than eyeballed off the table, because the table's rows are
    # round numbers and the answer never is.
    ratio = target_lift * math.sqrt(n_geos / 4.0) / ((z_a + z_b) * between_geo_sd)
    rho_needed = float(math.sqrt(max(0.0, 1.0 - min(ratio, 1.0) ** 2))) if ratio < 1 else 0.0

    return {
        "randomisation_unit": "geo",
        "n_geos_available": n_geos,
        "users_per_geo_per_week": users_per_geo_per_week,
        "users_per_week_total": n_geos * users_per_geo_per_week,
        "between_geo_sd": between_geo_sd,
        "mde_at_current_geo_count": mde_now,
        "target_lift": target_lift,
        "geos_needed_for_target_lift": needed_geos,
        "feasible_with_available_geos": needed_geos <= n_geos,
        "cuped_on_pre_period": cuped,
        "min_pre_period_rho_to_hit_target": rho_needed,
        "note": ("with %d geos the study can only detect a %.1f point change; detecting the %.1f "
                 "point target needs %d geos, or pre-period adjustment at rho>=%.2f -- and a "
                 "geo's own retention history correlating that well with its next four weeks is "
                 "an assumption to check, not to assume. The %s users per week are irrelevant to "
                 "the power; the effective n is the geo count."
                 % (n_geos, 100 * mde_now, 100 * target_lift, needed_geos, rho_needed,
                    "{:,}".format(n_geos * users_per_geo_per_week))),
        "what_it_cannot_do": ("a geo holdout measures the effect of TURNING PAID SEARCH OFF, which "
                              "is the decision-relevant quantity but is not the same as the effect "
                              "of a paid-search user being a paid-search user. Organic pickup means "
                              "the two differ, and the holdout measures the one worth knowing."),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["validate", "design", "assess"])
    ap.add_argument("--tau", type=float, default=0.0)
    args = ap.parse_args()

    if args.command == "validate":
        out = validate()
    elif args.command == "design":
        out = geo_holdout_design()
    else:
        out = assess(WorldConfig(tau=args.tau))

    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, "incrementality_%s.json" % args.command)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print(json.dumps(out, indent=2, default=float))
    print("\nwritten:", os.path.relpath(path, ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
