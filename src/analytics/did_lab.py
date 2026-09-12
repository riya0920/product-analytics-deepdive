"""Difference-in-differences, graded against a planted effect.

    python -m analytics.did_lab            # grade DiD on a panel with known tau

Like ``incrementality.py`` and ``lalonde_lab.py``, this plants a **known**
treatment effect so the estimator can be scored rather than admired. DiD's entire
identifying assumption is **parallel trends**: absent treatment, the treated and
control groups would have moved in parallel, so the control group's change is a
valid counterfactual for the treated group's change.

This module does two runs on the same machinery:

1. **Parallel trends holds.** Two-way fixed-effects DiD recovers tau, and a
   placebo test on the pre-period finds no pre-trend.
2. **Parallel trends is violated** — the treated units are on a different
   underlying trajectory before anyone is treated. DiD is then biased by
   *exactly* that trend gap, and the placebo test catches it. The estimate is
   reported with its bias rather than tuned until it looks right, because a
   method is only as trustworthy as the assumption it rests on.

The estimator is the textbook two-way fixed-effects regression
``y ~ unit FE + period FE + beta * (treated x post)`` fit by OLS, with
**cluster-robust standard errors by unit** (the DiD error term is correlated
within a unit over time, and ignoring that understates the SE).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# data with a known effect
# ---------------------------------------------------------------------------
def simulate_panel(
    n_units: int = 120,
    n_periods: int = 8,
    cutoff: int = 4,
    treat_frac: float = 0.5,
    tau: float = 3.0,
    pretrend_gap: float = 0.0,
    noise: float = 1.0,
    seed: int = 0,
) -> pd.DataFrame:
    """A balanced panel with unit fixed effects, a common time path, and a
    treatment that switches on for the treated units at ``cutoff``.

    ``pretrend_gap`` adds a per-period drift to the treated units ONLY. At 0 the
    parallel-trends assumption holds by construction; non-zero violates it (the
    treated group was already diverging before treatment), which is the case DiD
    cannot save you from.
    """
    rng = np.random.default_rng(seed)
    units = np.arange(n_units)
    treated = units < int(round(n_units * treat_frac))
    alpha = rng.normal(0.0, 2.0, n_units)                 # unit fixed effects
    time_fe = np.cumsum(rng.normal(0.2, 0.5, n_periods))  # common shocks/trend

    rows = []
    for t in range(n_periods):
        post = 1 if t >= cutoff else 0
        for i in units:
            ti = bool(treated[i])
            d = 1.0 if (ti and post) else 0.0
            y = (
                alpha[i]
                + time_fe[t]
                + pretrend_gap * t * (1.0 if ti else 0.0)   # violates parallel trends when != 0
                + tau * d
                + rng.normal(0.0, noise)
            )
            rows.append((int(i), int(t), int(ti), post, d, float(y)))
    return pd.DataFrame(rows, columns=["unit", "period", "treated", "post", "treat_post", "y"])


# ---------------------------------------------------------------------------
# estimators
# ---------------------------------------------------------------------------
def did_2x2(df: pd.DataFrame) -> dict:
    """Canonical 2x2 DiD: the difference of the treated and control pre/post
    changes in group means. Exact when the panel collapses to two periods."""
    g = df.groupby(["treated", "post"])["y"].mean().unstack()
    treated_change = g.loc[1, 1] - g.loc[1, 0]
    control_change = g.loc[0, 1] - g.loc[0, 0]
    att = float(treated_change - control_change)
    return {"method": "2x2 DiD (group means)", "estimate": att}


def _ols_cluster(X: np.ndarray, y: np.ndarray, target: int, clusters: np.ndarray) -> tuple[float, float]:
    """OLS coefficient on column ``target`` with cluster-robust (by cluster) SE."""
    XtX = X.T @ X
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    # cluster-robust meat: sum over clusters of (X_g' u_g)(X_g' u_g)'
    meat = np.zeros_like(XtX)
    for c in np.unique(clusters):
        m = clusters == c
        Xg = X[m]
        ug = resid[m]
        s = Xg.T @ ug
        meat += np.outer(s, s)
    G = len(np.unique(clusters))
    k = X.shape[1]
    n = X.shape[0]
    adj = (G / (G - 1)) * ((n - 1) / (n - k)) if G > 1 and n > k else 1.0
    cov = XtX_inv @ meat @ XtX_inv * adj
    return float(beta[target]), float(np.sqrt(cov[target, target]))


def twfe_did(df: pd.DataFrame) -> dict:
    """Two-way fixed-effects DiD: y ~ unit FE + period FE + beta*(treated x post),
    fit by OLS with cluster-robust SE by unit."""
    unit_d = pd.get_dummies(df["unit"], prefix="u", drop_first=True).to_numpy(float)
    period_d = pd.get_dummies(df["period"], prefix="t", drop_first=True).to_numpy(float)
    d = df["treat_post"].to_numpy(float).reshape(-1, 1)
    intercept = np.ones((len(df), 1))
    X = np.hstack([intercept, d, unit_d, period_d])
    y = df["y"].to_numpy(float)
    beta, se = _ols_cluster(X, y, target=1, clusters=df["unit"].to_numpy())
    z = beta / se if se > 0 else float("nan")
    return {
        "method": "TWFE DiD (OLS, cluster-robust SE)",
        "estimate": beta,
        "se": se,
        "ci_low": beta - 1.96 * se,
        "ci_high": beta + 1.96 * se,
        "z": z,
    }


def parallel_trends_placebo(df: pd.DataFrame, cutoff: int) -> dict:
    """Placebo DiD on the pre-period only: split the pre-treatment periods in
    half with a fake cutoff and run DiD. A real effect here is impossible (nobody
    is treated yet), so a non-zero estimate is a pre-trend -- the signature of a
    parallel-trends violation."""
    pre = df[df["period"] < cutoff].copy()
    fake = cutoff // 2
    if fake < 1 or pre["period"].nunique() < 2:
        return {"method": "pre-trend placebo", "estimate": float("nan"),
                "note": "not enough pre-periods"}
    pre["post"] = (pre["period"] >= fake).astype(int)
    pre["treat_post"] = ((pre["treated"] == 1) & (pre["post"] == 1)).astype(int)
    est = did_2x2(pre)
    return {"method": "pre-trend placebo (should be ~0)", "estimate": est["estimate"]}


# ---------------------------------------------------------------------------
# grading
# ---------------------------------------------------------------------------
def grade(est: dict, tau: float) -> dict:
    out = dict(est)
    e = est.get("estimate")
    out["truth"] = tau
    out["bias"] = None if e is None or e != e else float(e - tau)
    if "ci_low" in est and est["ci_low"] == est["ci_low"]:
        out["covers_truth"] = bool(est["ci_low"] <= tau <= est["ci_high"])
    return out


def run(tau: float = 3.0, cutoff: int = 4, seed: int = 0) -> dict:
    """Grade DiD under parallel trends and under a violation of it."""
    scenarios = {
        "parallel_trends_holds": simulate_panel(tau=tau, cutoff=cutoff, pretrend_gap=0.0, seed=seed),
        "parallel_trends_violated": simulate_panel(tau=tau, cutoff=cutoff, pretrend_gap=0.6, seed=seed),
    }
    report = {"truth": tau, "cutoff": cutoff, "scenarios": {}}
    for name, df in scenarios.items():
        report["scenarios"][name] = {
            "twfe": grade(twfe_did(df), tau),
            "did_2x2": grade(did_2x2(df), tau),
            "placebo": parallel_trends_placebo(df, cutoff),
        }
    return report


def to_markdown(rep: dict) -> str:
    tau = rep["truth"]
    lines = [f"### Difference-in-differences vs a planted effect (tau = {tau})", ""]
    for name, s in rep["scenarios"].items():
        pretty = name.replace("_", " ")
        lines.append(f"**{pretty}**")
        lines.append("")
        lines.append("| estimator | estimate | bias | 95% CI covers tau | placebo pre-trend |")
        lines.append("| --- | --- | --- | --- | --- |")
        tw = s["twfe"]
        placebo = s["placebo"]["estimate"]
        placebo_s = "—" if placebo != placebo else f"{placebo:+.2f}"
        covers = tw.get("covers_truth")
        lines.append(
            f"| TWFE DiD | {tw['estimate']:+.2f} (se {tw['se']:.2f}) | {tw['bias']:+.2f} "
            f"| {'yes' if covers else 'no'} | {placebo_s} |"
        )
        lines.append(
            f"| 2x2 DiD | {s['did_2x2']['estimate']:+.2f} | {s['did_2x2']['bias']:+.2f} | — | |"
        )
        lines.append("")
    lines.append(
        "Under parallel trends DiD recovers tau and the placebo pre-trend is ~0. "
        "Under a violation the estimate is biased by the trend gap and the placebo "
        "pre-trend is non-zero — the diagnostic that would stop you trusting it."
    )
    return "\n".join(lines)


def main() -> int:
    rep = run()
    print(to_markdown(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
