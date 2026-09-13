"""Instrumental variables (2SLS), graded against a planted effect.

    python -m analytics.iv_lab            # grade 2SLS on data with known beta

The same plant-then-score discipline as ``did_lab`` and ``lalonde_lab``. An
**unobserved confounder** U drives both the treatment D and the outcome Y, so a
plain OLS of Y on D is biased; it cannot separate the causal effect from the
selection U induces. An **instrument** Z is a variable that moves D but affects Y
*only through D* (the exclusion restriction) and is independent of U. Two-stage
least squares uses only the part of D that Z explains, which is U-free, so it
recovers the causal effect.

Three runs on one machine:

1. **Strong, valid instrument.** OLS is biased; 2SLS recovers beta; the
   first-stage F is well above the Staiger–Stock rule of thumb of 10.
2. **Weak instrument.** Z barely moves D, the first-stage F collapses below 10,
   and 2SLS becomes high-variance and unreliable; the diagnostic that says "do
   not trust this estimate" fires.
3. **Exclusion restriction violated.** Z affects Y directly (not only through D),
   so it is not a valid instrument; 2SLS is biased by the direct path even though
   the first-stage F looks strong. Reported with its bias, not tuned away.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# data with a known effect, an endogenous treatment, and an instrument
# ---------------------------------------------------------------------------
def simulate(
    n: int = 4000,
    beta: float = 2.0,
    instrument_strength: float = 1.0,   # Z -> D  (first-stage strength)
    confounding: float = 1.5,           # U -> D and U -> Y (endogeneity)
    exclusion_violation: float = 0.0,   # Z -> Y direct path (breaks IV validity)
    seed: int = 0,
) -> dict:
    """One instrument Z, one unobserved confounder U, treatment D, outcome Y.

    D = instrument_strength * Z + confounding * U + noise
    Y = beta * D + confounding * U + exclusion_violation * Z + noise

    U is returned for reference but is NOT given to the estimators (it is the
    thing that makes OLS biased).
    """
    rng = np.random.default_rng(seed)
    Z = rng.normal(0.0, 1.0, n)
    U = rng.normal(0.0, 1.0, n)                       # unobserved confounder
    D = instrument_strength * Z + confounding * U + rng.normal(0.0, 1.0, n)
    Y = beta * D + confounding * U + exclusion_violation * Z + rng.normal(0.0, 1.0, n)
    return {"Y": Y, "D": D, "Z": Z, "U": U}


# ---------------------------------------------------------------------------
# estimators
# ---------------------------------------------------------------------------
def _ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (beta, cov, rss) for y = X beta."""
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    rss = float(resid @ resid)
    dof = max(len(y) - X.shape[1], 1)
    cov = (rss / dof) * XtX_inv
    return beta, cov, rss


def ols(d: dict) -> dict:
    """Naive OLS of Y on D; biased when a confounder drives both."""
    X = np.column_stack([np.ones_like(d["D"]), d["D"]])
    beta, cov, _ = _ols(X, d["Y"])
    return {"method": "OLS (naive)", "estimate": float(beta[1]),
            "se": float(np.sqrt(cov[1, 1]))}


def first_stage_F(d: dict) -> float:
    """F-statistic for the instrument in the first stage D ~ 1 + Z.

    Below ~10 the instrument is 'weak' (Staiger–Stock) and 2SLS is unreliable.
    """
    n = len(d["D"])
    Z1 = np.column_stack([np.ones(n), d["Z"]])
    _, _, rss_u = _ols(Z1, d["D"])
    _, _, rss_r = _ols(np.ones((n, 1)), d["D"])   # restricted: intercept only
    q, k = 1, 2
    return float(((rss_r - rss_u) / q) / (rss_u / (n - k)))


def tsls(d: dict) -> dict:
    """Two-stage least squares of Y on D, instrumented by Z."""
    n = len(d["D"])
    Z1 = np.column_stack([np.ones(n), d["Z"]])      # instruments (+ intercept)
    X = np.column_stack([np.ones(n), d["D"]])       # regressors
    # First stage: project X onto the instrument space.
    Pi = np.linalg.pinv(Z1.T @ Z1) @ (Z1.T @ X)
    Xhat = Z1 @ Pi
    XhatX_inv = np.linalg.pinv(Xhat.T @ X)
    beta = XhatX_inv @ (Xhat.T @ d["Y"])
    # 2SLS residuals use the ACTUAL D, not the fitted D.
    resid = d["Y"] - X @ beta
    sigma2 = float(resid @ resid) / max(n - 2, 1)
    cov = sigma2 * np.linalg.pinv(Xhat.T @ Xhat)
    return {"method": "2SLS (IV)", "estimate": float(beta[1]),
            "se": float(np.sqrt(cov[1, 1])), "first_stage_F": first_stage_F(d)}


# ---------------------------------------------------------------------------
# grading
# ---------------------------------------------------------------------------
def grade(est: dict, beta: float) -> dict:
    out = dict(est)
    e = est["estimate"]
    out["truth"] = beta
    out["bias"] = float(e - beta)
    se = est.get("se")
    if se:
        out["covers_truth"] = bool(e - 1.96 * se <= beta <= e + 1.96 * se)
    return out


def run(beta: float = 2.0, seed: int = 0) -> dict:
    scenarios = {
        "strong_valid_instrument": simulate(beta=beta, instrument_strength=1.0,
                                            exclusion_violation=0.0, seed=seed),
        "weak_instrument": simulate(beta=beta, instrument_strength=0.05,
                                    exclusion_violation=0.0, seed=seed),
        "exclusion_violated": simulate(beta=beta, instrument_strength=1.0,
                                       exclusion_violation=1.5, seed=seed),
    }
    report = {"truth": beta, "scenarios": {}}
    for name, d in scenarios.items():
        report["scenarios"][name] = {
            "ols": grade(ols(d), beta),
            "tsls": grade(tsls(d), beta),
        }
    return report


def to_markdown(rep: dict) -> str:
    beta = rep["truth"]
    lines = [f"### Instrumental variables vs a planted effect (beta = {beta})", ""]
    for name, s in rep["scenarios"].items():
        pretty = name.replace("_", " ")
        tsls_est = s["tsls"]
        lines.append(f"**{pretty}** (first-stage F = {tsls_est['first_stage_F']:.1f})")
        lines.append("")
        lines.append("| estimator | estimate | bias | 95% CI covers beta |")
        lines.append("| --- | --- | --- | --- |")
        for key in ("ols", "tsls"):
            e = s[key]
            covers = e.get("covers_truth")
            lines.append(
                f"| {e['method']} | {e['estimate']:+.2f} (se {e['se']:.2f}) "
                f"| {e['bias']:+.2f} | {'yes' if covers else 'no'} |"
            )
        lines.append("")
    lines.append(
        "With a strong valid instrument OLS is biased by the confounder and 2SLS "
        "recovers beta. A weak instrument (first-stage F below 10) makes 2SLS "
        "high-variance and unreliable. When the exclusion restriction is violated "
        "the instrument is invalid and 2SLS is biased by the direct path, a "
        "strong first-stage F does not rescue it."
    )
    return "\n".join(lines)


def main() -> int:
    print(to_markdown(run()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
