"""Grading causal methods against a real experiment, on real data.

Everything else in this repository is measured on a generator I wrote, which
means I chose the answer before I measured it. This module does the same exercise
on data where somebody else chose the answer, and chose it by running an actual
randomised trial on actual people.

## The data

The National Supported Work Demonstration (LaLonde 1986; subset and observational
comparison groups from Dehejia and Wahba 1999). Unemployed men were **randomly
assigned** to a subsidised training programme or not, and their 1978 earnings were
recorded. Random assignment means the difference in earnings between the two arms
is the causal effect of the programme, with no adjustment required.

That is the ground truth: **$1,794**.

The same release also ships two *observational* comparison groups, PSID and CPS,
pulled from national surveys. These people were never in the experiment. Swapping
the experimental control arm for one of them is exactly what an analyst does when
there is no experiment: find people who look comparable and compare.

So the setup grades itself. Take the treated men, compare them to a survey
population instead of their real control group, run the methods an analyst would
actually run, and check each answer against a number that is known.

## Why this is the right benchmark rather than a synthetic one

A simulation can only be as adversarial as its author. This one is harder than
anything I would have thought to write: the naive comparison against PSID does not
merely get the magnitude wrong, it gets the **sign** wrong by seventeen thousand
dollars, and it does so with a tight confidence interval. It is confidently,
precisely wrong, which is the failure mode that actually costs money.

## What the module reports

For each comparison group and each estimator: the estimate, its interval, the bias
against the experimental benchmark, and whether the interval covers the truth.

Before any of that it reports **overlap**, because overlap is the part that
decides whether adjustment can work at all, and it is the step that gets skipped.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "lalonde"))
RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results"))

COLS = ["treat", "age", "educ", "black", "hispan", "married", "nodegree", "re74", "re75", "re78"]
COVARIATES = ["age", "educ", "black", "hispan", "married", "nodegree", "re74", "re75"]

FILES = {
    "treated": "nswre74_treated.txt",
    "experimental_control": "nswre74_control.txt",
    "psid": "psid_controls.txt",
    "cps": "cps_controls.txt",
}


BASE_URL = "https://users.nber.org/~rdehejia/data/"


def download(data_dir: str = DATA) -> list[str]:
    """Fetch the Dehejia and Wahba release.

    The files are not committed. They are somebody else's research data, and a
    repository that vendors three megabytes of it to save one command is making
    the wrong trade.
    """
    import urllib.request

    os.makedirs(data_dir, exist_ok=True)
    got = []
    for name in FILES.values():
        dest = os.path.join(data_dir, name)
        if not os.path.exists(dest):
            urllib.request.urlretrieve(BASE_URL + name, dest)
        got.append(name)
    return got


def load(name: str, data_dir: str = DATA) -> pd.DataFrame:
    path = os.path.join(data_dir, FILES[name])
    return pd.read_csv(path, sep=r"\s+", header=None, names=COLS)


def _mean_diff(a: np.ndarray, b: np.ndarray) -> dict:
    d = a.mean() - b.mean()
    se = float(np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))
    return {"estimate": float(d), "se": se,
            "ci_low": float(d - 1.96 * se), "ci_high": float(d + 1.96 * se),
            "n_treated": int(len(a)), "n_control": int(len(b))}


def experimental_truth(data_dir: str = DATA) -> dict:
    """The benchmark. Randomisation is what makes this a causal effect."""
    t, c = load("treated", data_dir), load("experimental_control", data_dir)
    out = _mean_diff(t.re78.values, c.re78.values)
    out["method"] = "randomised experiment"
    return out


# ---------------------------------------------------------------------------
# overlap, which decides whether any of the rest can work
# ---------------------------------------------------------------------------

def standardised_differences(t: pd.DataFrame, c: pd.DataFrame) -> dict:
    """How far apart the two groups are, per covariate, in pooled SD units.

    The usual rule of thumb is that anything past 0.1 is an imbalance worth
    fixing. This is reported before any estimate because a comparison group that
    is nothing like the treated group cannot be rescued by adjustment: there is
    no one to adjust *to*.
    """
    out = {}
    for v in COVARIATES:
        a, b = t[v].values, c[v].values
        pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        out[v] = float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0
    return out


def propensity(t: pd.DataFrame, c: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """P(treated | covariates), fitted on the pooled sample."""
    df = pd.concat([t.assign(_z=1), c.assign(_z=0)], ignore_index=True)
    X = df[COVARIATES].values.astype(float)
    X = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1.0)
    z = df["_z"].values
    model = LogisticRegression(max_iter=2000, C=1.0)
    model.fit(X, z)
    ps = model.predict_proba(X)[:, 1]
    return ps, z, df["re78"].values


def overlap_report(t: pd.DataFrame, c: pd.DataFrame) -> dict:
    ps, z, _ = propensity(t, c)
    ps_t, ps_c = ps[z == 1], ps[z == 0]
    lo, hi = max(ps_t.min(), ps_c.min()), min(ps_t.max(), ps_c.max())
    in_support = float(((ps >= lo) & (ps <= hi)).mean())
    smd = standardised_differences(t, c)
    worst = max(smd, key=lambda k: abs(smd[k]))
    return {
        "smd": smd,
        "worst_covariate": worst,
        "worst_smd": smd[worst],
        "n_covariates_imbalanced": int(sum(abs(v) > 0.1 for v in smd.values())),
        "propensity_treated_median": float(np.median(ps_t)),
        "propensity_control_median": float(np.median(ps_c)),
        "controls_above_treated_min": float((ps_c >= ps_t.min()).mean()),
        "fraction_in_common_support": in_support,
    }


# ---------------------------------------------------------------------------
# the estimators an analyst would actually reach for
# ---------------------------------------------------------------------------

def naive(t: pd.DataFrame, c: pd.DataFrame) -> dict:
    out = _mean_diff(t.re78.values, c.re78.values)
    out["method"] = "naive difference"
    return out


def ols_adjusted(t: pd.DataFrame, c: pd.DataFrame) -> dict:
    """Regression on the covariates. The reflex move, and it is not enough."""
    df = pd.concat([t.assign(_z=1), c.assign(_z=0)], ignore_index=True)
    X = np.column_stack([np.ones(len(df)), df["_z"].values,
                         df[COVARIATES].values.astype(float)])
    y = df["re78"].values
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    cov = np.linalg.pinv(X.T @ X) * (resid @ resid) / dof
    se = float(np.sqrt(cov[1, 1]))
    d = float(beta[1])
    return {"method": "OLS with covariates", "estimate": d, "se": se,
            "ci_low": d - 1.96 * se, "ci_high": d + 1.96 * se,
            "n_treated": len(t), "n_control": len(c)}


def psm(t: pd.DataFrame, c: pd.DataFrame, caliper: float = 0.05, seed: int = 0) -> dict:
    """One-to-one nearest neighbour on the propensity score, with a caliper.

    The caliper is the part that matters. Without it every treated unit is matched
    to *something*, however unlike it, and the method reports an answer for a
    comparison that does not exist in the data.
    """
    ps, z, y = propensity(t, c)
    ps_t, y_t = ps[z == 1], y[z == 1]
    ps_c, y_c = ps[z == 0], y[z == 0]
    order = np.argsort(ps_c)
    ps_c_sorted, y_c_sorted = ps_c[order], y_c[order]

    pairs = []
    for p, yt in zip(ps_t, y_t):
        j = int(np.searchsorted(ps_c_sorted, p))
        best, bestd = None, np.inf
        for k in (j - 1, j, j + 1):
            if 0 <= k < len(ps_c_sorted):
                d = abs(ps_c_sorted[k] - p)
                if d < bestd:
                    best, bestd = k, d
        if best is not None and bestd <= caliper:
            pairs.append((yt, y_c_sorted[best]))

    if not pairs:
        return {"method": "propensity matching", "estimate": float("nan"),
                "se": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "matched_pairs": 0, "unmatched_treated": len(ps_t),
                "n_treated": len(t), "n_control": len(c)}

    arr = np.array(pairs)
    d = float((arr[:, 0] - arr[:, 1]).mean())
    rng = np.random.default_rng(seed)
    boots = [float((arr[i, 0] - arr[i, 1]).mean())
             for i in (rng.integers(0, len(arr), len(arr)) for _ in range(400))]
    se = float(np.std(boots, ddof=1))
    return {"method": "propensity matching", "estimate": d, "se": se,
            "ci_low": d - 1.96 * se, "ci_high": d + 1.96 * se,
            "matched_pairs": len(pairs),
            "unmatched_treated": int(len(ps_t) - len(pairs)),
            "n_treated": len(t), "n_control": len(c)}


def ipw(t: pd.DataFrame, c: pd.DataFrame, trim: float = 0.01, seed: int = 0) -> dict:
    """Inverse propensity weighting, trimmed.

    Trimming is not tidying. A control with propensity 0.001 receives a weight of
    a thousand, and one such row can move the estimate more than the other
    fifteen thousand combined.
    """
    ps, z, y = propensity(t, c)
    keep = (ps > trim) & (ps < 1 - trim)
    ps, z, y = ps[keep], z[keep], y[keep]
    w = np.where(z == 1, 1.0 / ps, 1.0 / (1.0 - ps))

    def est(idx):
        zz, yy, ww = z[idx], y[idx], w[idx]
        a = (ww[zz == 1] * yy[zz == 1]).sum() / ww[zz == 1].sum()
        b = (ww[zz == 0] * yy[zz == 0]).sum() / ww[zz == 0].sum()
        return float(a - b)

    d = est(np.arange(len(z)))
    rng = np.random.default_rng(seed)
    boots = [est(rng.integers(0, len(z), len(z))) for _ in range(400)]
    se = float(np.std(boots, ddof=1))
    return {"method": "IPW (trimmed)", "estimate": d, "se": se,
            "ci_low": d - 1.96 * se, "ci_high": d + 1.96 * se,
            "dropped_by_trim": int((~keep).sum()),
            "max_weight": float(w.max()),
            "n_treated": len(t), "n_control": len(c)}


def aipw(t: pd.DataFrame, c: pd.DataFrame, trim: float = 0.01, seed: int = 0) -> dict:
    """Doubly robust: outcome model plus weighting, so either one may be wrong."""
    df = pd.concat([t.assign(_z=1), c.assign(_z=0)], ignore_index=True)
    ps, z, y = propensity(t, c)
    keep = (ps > trim) & (ps < 1 - trim)
    df, ps, z, y = df[keep].reset_index(drop=True), ps[keep], z[keep], y[keep]

    X = df[COVARIATES].values.astype(float)
    Xd = np.column_stack([np.ones(len(df)), X])
    mu = {}
    for arm in (0, 1):
        m = z == arm
        beta, *_ = np.linalg.lstsq(Xd[m], y[m], rcond=None)
        mu[arm] = Xd @ beta

    def est(idx):
        zz, yy, pp = z[idx], y[idx], ps[idx]
        m1, m0 = mu[1][idx], mu[0][idx]
        a = m1 + zz * (yy - m1) / pp
        b = m0 + (1 - zz) * (yy - m0) / (1 - pp)
        return float((a - b).mean())

    d = est(np.arange(len(z)))
    rng = np.random.default_rng(seed)
    boots = [est(rng.integers(0, len(z), len(z))) for _ in range(400)]
    se = float(np.std(boots, ddof=1))
    return {"method": "AIPW (doubly robust)", "estimate": d, "se": se,
            "ci_low": d - 1.96 * se, "ci_high": d + 1.96 * se,
            "n_treated": len(t), "n_control": len(c)}


ESTIMATORS = [naive, ols_adjusted, psm, ipw, aipw]


def grade(est: dict, truth: dict) -> dict:
    """Bias against the experiment, and whether the interval covers the truth."""
    e, tr = est["estimate"], truth["estimate"]
    covers = (est["ci_low"] <= tr <= est["ci_high"]) if e == e else False
    return {**est,
            "bias": float(e - tr) if e == e else float("nan"),
            "sign_correct": bool(e > 0) if e == e else False,
            "covers_truth": bool(covers)}


def run(data_dir: str = DATA) -> dict:
    truth = experimental_truth(data_dir)
    t = load("treated", data_dir)

    report = {"truth": truth, "comparisons": []}
    for group in ("experimental_control", "psid", "cps"):
        c = load(group, data_dir)
        rows = [grade(f(t, c), truth) for f in ESTIMATORS]
        report["comparisons"].append({
            "control_group": group,
            "overlap": overlap_report(t, c),
            "estimates": rows,
        })
    return report


def to_markdown(rep: dict) -> str:
    tr = rep["truth"]
    out = ["# Grading causal methods against a real experiment", "",
           "**Ground truth (randomised): ${:,.0f}** "
           "(95% CI ${:,.0f} to ${:,.0f}, n={} treated vs {} control)".format(
               tr["estimate"], tr["ci_low"], tr["ci_high"],
               tr["n_treated"], tr["n_control"]), ""]
    for comp in rep["comparisons"]:
        o = comp["overlap"]
        out += ["## Control group: `%s`" % comp["control_group"], "",
                "Overlap: **{}/{} covariates imbalanced** (worst `{}` at {:+.2f} SD), "
                "only **{:.1%}** of controls reach the lowest treated propensity.".format(
                    o["n_covariates_imbalanced"], len(o["smd"]),
                    o["worst_covariate"], o["worst_smd"],
                    o["controls_above_treated_min"]), "",
                "| method | estimate | 95% CI | bias vs truth | sign | covers truth |",
                "|---|---|---|---|---|---|"]
        for e in comp["estimates"]:
            if e["estimate"] != e["estimate"]:
                out.append("| %s | no valid matches | | | | |" % e["method"])
                continue
            out.append("| {} | ${:,.0f} | ${:,.0f} to ${:,.0f} | ${:+,.0f} | {} | {} |".format(
                e["method"], e["estimate"], e["ci_low"], e["ci_high"], e["bias"],
                "ok" if e["sign_correct"] else "WRONG",
                "yes" if e["covers_truth"] else "no"))
        out.append("")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", default=DATA)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--download", action="store_true",
                    help="fetch the data and exit")
    args = ap.parse_args()
    if args.download:
        for f in download(args.data_dir):
            print("have", f)
        return 0
    rep = run(args.data_dir)
    print(json.dumps(rep, indent=2, default=float) if args.json else to_markdown(rep))
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "lalonde_grading.json"), "w") as fh:
        json.dump(rep, fh, indent=2, default=float)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
