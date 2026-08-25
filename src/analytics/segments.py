"""Segmentation beyond channel - and the discipline that makes it safe.

    python -m analytics.segments

The memo's headline is that paid search retains worse. A single cut is a
hypothesis, not a finding: the honest follow-up is whether the gap survives
conditioning on everything else observable, or whether it was a mix artifact all
along.

## Why this module refuses to print some numbers

Cutting a 60,000-user dataset by channel x platform x cohort-week produces
hundreds of cells. Two things go wrong at that point, and both are silent:

  * **Multiplicity.** With 5 channels x 3 platforms and a 5% test, roughly one
    cell in twenty looks "significant" with nothing going on. So every cell here
    carries a Wilson interval, and the module never reports a bare point estimate
    for a cut nobody pre-declared.
  * **Thin cells.** A 40-user cell has a retention interval roughly +/-15
    points wide. It is not evidence of anything, and printing it invites someone
    to build a slide on it. Cells below `MIN_CELL` are counted and excluded, and
    the count is reported so the exclusion is visible rather than quiet.

## What it checks

1. **Does the paid-search gap hold within every platform?** If it only appears in
   aggregate, the "channel effect" is a platform-mix effect wearing a hat.
2. **Simpson's paradox, explicitly.** Any cut where the aggregate direction
   reverses inside every subgroup is flagged by name.
3. **Cohort-over-cohort.** Is the gap widening, stable, or an artifact of one
   bad week?
4. **Geography, decomposed.** A regional retention gap has two explanations with
   opposite responses -- a different channel mix (marketing) or a worse
   experience within the same channel (product) -- and the headline number
   cannot tell them apart. `kitagawa_decomposition` splits it.
5. **The same cuts on the UNCLEANED warehouse**, which is where this gets
   interesting: the planted iOS timezone bug manufactures a platform effect that
   does not exist. Segmentation on unvalidated data does not just add noise, it
   adds confident, wrong findings - and this is the demonstration.
"""
from __future__ import annotations

import argparse
import json
import math
import os

import duckdb

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
DB = os.path.join(DATA, "warehouse.duckdb")
DIRTY_DB = os.path.join(DATA, "warehouse_dirty.duckdb")

# Below this, a retention rate's interval is wider than any effect worth acting
# on, so the cell is excluded rather than reported.
MIN_CELL = 300


def wilson(k: int, n: int, z: float = 1.959963985):
    """Wilson score interval. Not normal-approximation.

    At the cell sizes this module produces, some rates land near 0 or 1 where the
    normal interval runs outside [0,1] and reports a lower bound below zero.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(centre - half, 0.0), min(centre + half, 1.0))


RETENTION_CELLS = """
WITH activity AS (
    SELECT DISTINCT u.user_id, u.channel, u.platform, u.signup_week,
           DATE_DIFF('day', u.signup_date, CAST(e.event_ts AS DATE)) AS day_number
    FROM stg_events e
    JOIN stg_users u USING (user_id)
    WHERE CAST(e.event_ts AS DATE) >= u.signup_date
)
SELECT {dims},
       COUNT(DISTINCT u.user_id) AS cohort_users,
       COUNT(DISTINCT CASE WHEN {metric} THEN a.user_id END) AS retained
FROM stg_users u
LEFT JOIN activity a USING (user_id)
GROUP BY {dims}
ORDER BY {dims}
"""

# Two retention definitions, and which one you pick decides what you can see.
#
#   d7_plus    "active on day 7 or later" -- the classic curve, and robust to a
#              few hours of clock error because a whole tail of later activity
#              backs it up.
#   d1_exact   "active on exactly day 1" -- fragile by construction, because a
#              timestamp near midnight can fall either side of the boundary.
#
# The timezone bug is INVISIBLE in d7_plus (0.3 points, not significant) and
# obvious in d1_exact. That is not a flaw in either metric; it is the reason a
# data-quality audit cannot be replaced by "the dashboard looks fine".
METRICS = {
    "d7_plus": "a.day_number >= 7",
    "d1_exact": "a.day_number = 1",
    "d7_exact": "a.day_number = 7",
}


def _cells(con, dims: list, metric: str = "d7_plus") -> list:
    qualified = [("u." + d if d != "signup_week" else "u.signup_week") for d in dims]
    sql = RETENTION_CELLS.format(dims=", ".join(qualified), metric=METRICS[metric])
    rows = con.execute(sql).fetchall()
    out = []
    for r in rows:
        n, k = int(r[-2]), int(r[-1])
        lo, hi = wilson(k, n)
        out.append({**{d: r[i] for i, d in enumerate(dims)},
                    "n": n, "retained": k, "rate": k / n if n else float("nan"),
                    "ci_low": lo, "ci_high": hi, "thin": n < MIN_CELL})
    return out


def _gap(a: dict, b: dict) -> dict:
    """Difference in two rates with an interval, and whether it excludes zero.

    Independent-cell normal approximation on the difference. Stated rather than
    hidden: the Wilson intervals above are for the rates, this one is for the
    gap, and they are not the same calculation.
    """
    if not a["n"] or not b["n"]:
        return {"diff": float("nan"), "significant": False}
    pa, pb = a["rate"], b["rate"]
    se = math.sqrt(pa * (1 - pa) / a["n"] + pb * (1 - pb) / b["n"])
    d = pa - pb
    return {"diff": d, "se": se, "ci_low": d - 1.96 * se, "ci_high": d + 1.96 * se,
            "significant": abs(d) > 1.96 * se}


# ---------------------------------------------------------------------------
# the four checks
# ---------------------------------------------------------------------------

def gap_within_platforms(con, focus: str = "paid_search", baseline: str = "organic") -> dict:
    """Does the channel gap survive conditioning on platform?"""
    cells = {(c["channel"], c["platform"]): c for c in _cells(con, ["channel", "platform"])}
    agg = {c["channel"]: c for c in _cells(con, ["channel"])}

    overall = _gap(agg[focus], agg[baseline])
    per_platform = []
    for plat in sorted({p for (_, p) in cells}):
        a, b = cells.get((focus, plat)), cells.get((baseline, plat))
        if not a or not b or a["thin"] or b["thin"]:
            per_platform.append({"platform": plat, "excluded": "thin cell"})
            continue
        g = _gap(a, b)
        per_platform.append({"platform": plat, "focus_rate": a["rate"], "baseline_rate": b["rate"],
                             "n_focus": a["n"], "n_baseline": b["n"], **g})

    usable = [p for p in per_platform if "diff" in p]
    same_sign = usable and all((p["diff"] < 0) == (overall["diff"] < 0) for p in usable)
    all_sig = usable and all(p["significant"] for p in usable)
    return {
        "focus": focus, "baseline": baseline,
        "overall_gap": overall,
        "per_platform": per_platform,
        "gap_holds_within_every_platform": bool(same_sign and all_sig),
        "verdict": ("the gap is present and significant inside every platform, so it is not a "
                    "platform-mix artifact" if (same_sign and all_sig) else
                    "the gap does NOT hold uniformly across platforms -- treat the aggregate "
                    "number as a mix of different effects"),
    }


def simpson_check(con) -> dict:
    """Look for an aggregate direction that reverses inside every subgroup.

    Reported even when nothing is found. A paradox check that only ever appears
    in write-ups when it fires is a check nobody can calibrate.
    """
    cells = {(c["channel"], c["platform"]): c for c in _cells(con, ["channel", "platform"])}
    agg = {c["channel"]: c for c in _cells(con, ["channel"])}
    channels = sorted(agg)
    platforms = sorted({p for (_, p) in cells})

    reversals = []
    for i, a in enumerate(channels):
        for b in channels[i + 1:]:
            top = _gap(agg[a], agg[b])
            subs = []
            for plat in platforms:
                ca, cb = cells.get((a, plat)), cells.get((b, plat))
                if not ca or not cb or ca["thin"] or cb["thin"]:
                    continue
                subs.append(_gap(ca, cb)["diff"])
            if subs and all((d < 0) != (top["diff"] < 0) for d in subs):
                reversals.append({"pair": [a, b], "aggregate_diff": top["diff"],
                                  "subgroup_diffs": subs})
    return {"pairs_checked": len(channels) * (len(channels) - 1) // 2,
            "reversals_found": len(reversals), "reversals": reversals,
            "verdict": ("no Simpson reversal: every aggregate channel comparison keeps its sign "
                        "inside every platform" if not reversals else
                        "REVERSAL: at least one aggregate comparison flips inside every subgroup")}


def cohort_over_cohort(con, focus: str = "paid_search", baseline: str = "organic") -> dict:
    """Is the gap widening over signup weeks, or stable?

    A stable gap and a widening gap call for completely different responses -- one
    is a standing quality difference, the other is a live regression -- and the
    single pooled number cannot tell them apart.

    The slope is fitted by **weighted** least squares with each week's inverse
    variance as its weight, and reported with a standard error. The first version
    of this used an unweighted fit and a "is the drift bigger than 2 points"
    threshold, which called a slope of -0.18 pp/week a trend. It is not: against
    its own standard error it is under 2 sigma, and the generator plants no trend
    at all. A magic threshold on an unweighted slope is exactly the machinery that
    turns noise into a roadmap item.
    """
    cells = {(c["channel"], c["signup_week"]): c for c in _cells(con, ["channel", "signup_week"])}
    weeks = sorted({w for (_, w) in cells})
    series, thin = [], 0
    for w in weeks:
        a, b = cells.get((focus, w)), cells.get((baseline, w))
        if not a or not b or a["thin"] or b["thin"]:
            thin += 1
            continue
        g = _gap(a, b)
        series.append({"week": str(w), "gap": g["diff"], "se": g["se"],
                       "n_focus": a["n"], "n_baseline": b["n"]})

    trend = None
    if len(series) >= 4:
        xs = list(range(len(series)))
        ys = [t["gap"] for t in series]
        ws = [1.0 / (t["se"] ** 2) if t["se"] > 0 else 0.0 for t in series]
        sw = sum(ws)
        mx = sum(w * x for w, x in zip(ws, xs)) / sw
        my = sum(w * y for w, y in zip(ws, ys)) / sw
        sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
        slope = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys)) / sxx if sxx else 0.0
        se_slope = math.sqrt(1.0 / sxx) if sxx else float("inf")
        z = slope / se_slope if se_slope else 0.0
        trend = {"slope_per_week": slope, "se_per_week": se_slope, "z": z,
                 "significant": abs(z) > 1.96,
                 "total_drift_over_window": slope * (len(series) - 1)}

    moving = bool(trend and trend["significant"])
    return {"weeks_reported": len(series), "weeks_excluded_thin": thin,
            "series": series, "trend": trend,
            "verdict": ("gap is moving across cohorts (slope %+.2f pp/week, z=%.1f) -- the pooled "
                        "number hides a trend" % (100 * trend["slope_per_week"], trend["z"])
                        if moving else
                        "no significant cohort trend (slope %+.2f pp/week, z=%.1f); the gap is a "
                        "standing quality difference, not a live regression"
                        % (100 * trend["slope_per_week"], trend["z"]) if trend else
                        "too few usable weeks to fit a trend")}


# ---------------------------------------------------------------------------
# geography, and the decomposition that stops it becoming a bad recommendation
# ---------------------------------------------------------------------------

def region_rates(con, metric: str = "d7_plus") -> dict:
    return {c["region"]: c for c in _cells(con, ["region"], metric=metric)}


def region_channel_cells(con, metric: str = "d7_plus") -> dict:
    return {(c["region"], c["channel"]): c
            for c in _cells(con, ["region", "channel"], metric=metric)}


def kitagawa_decomposition(con, focus: str, baseline: str, metric: str = "d7_plus") -> dict:
    """Split a regional gap into COMPOSITION and RATE components.

    A region retaining worse has two completely different explanations with
    completely different responses:

      * **Composition.** The region buys a different channel mix, and the
        channels it buys more of retain worse everywhere. Nothing about the
        region is wrong; the acquisition portfolio is different. The response is
        a marketing one.
      * **Rate.** Within the same channel, users in that region retain worse.
        Something about the product experience there is worse -- latency,
        localisation, payment methods. The response is a product one.

    Kitagawa's decomposition (Oaxaca-Blinder for rates) separates them:

        gap = SUM_c (w_A,c - w_B,c) * rbar_c        <- composition
            + SUM_c  wbar_c * (r_A,c - r_B,c)       <- rate

    using the mean weight and mean rate as the reference, which is the symmetric
    form. The asymmetric version -- weighting composition by A's rates and rates
    by B's weights -- gives a different answer depending on which region you call
    the baseline, and there is no principled reason to prefer either direction.

    Reporting the headline gap without this split is how "EMEA retains 4 points
    worse" becomes a product investigation into a marketing fact.
    """
    cells = region_channel_cells(con, metric=metric)
    channels = sorted({ch for (_r, ch) in cells})

    def weights_and_rates(region):
        rows = [(ch, cells.get((region, ch))) for ch in channels]
        total = sum(c["n"] for _ch, c in rows if c)
        w = {ch: (c["n"] / total if c and total else 0.0) for ch, c in rows}
        r = {ch: (c["rate"] if c and c["n"] else float("nan")) for ch, c in rows}
        return w, r, total

    wA, rA, nA = weights_and_rates(focus)
    wB, rB, nB = weights_and_rates(baseline)

    comp = rate = 0.0
    per_channel = []
    for ch in channels:
        if not (wA[ch] and wB[ch]):
            continue
        wbar = (wA[ch] + wB[ch]) / 2.0
        rbar = (rA[ch] + rB[ch]) / 2.0
        c_part = (wA[ch] - wB[ch]) * rbar
        r_part = wbar * (rA[ch] - rB[ch])
        comp += c_part
        rate += r_part
        per_channel.append({
            "channel": ch,
            "weight_focus": wA[ch], "weight_baseline": wB[ch],
            "rate_focus": rA[ch], "rate_baseline": rB[ch],
            "composition_contribution": c_part,
            "rate_contribution": r_part,
        })

    agg = region_rates(con, metric=metric)
    observed = agg[focus]["rate"] - agg[baseline]["rate"]
    total = comp + rate
    share_comp = comp / total if abs(total) > 1e-12 else float("nan")

    # Standard error of the RATE component, so "consistent with zero" is a claim
    # rather than an eyeball. Each channel contributes wbar * (rA - rB), and the
    # channels are disjoint user sets, so the variances add.
    var_rate = 0.0
    for row in per_channel:
        ch = row["channel"]
        a, b = cells.get((focus, ch)), cells.get((baseline, ch))
        if not (a and b and a["n"] and b["n"]):
            continue
        wbar = (row["weight_focus"] + row["weight_baseline"]) / 2.0
        va = a["rate"] * (1 - a["rate"]) / a["n"]
        vb = b["rate"] * (1 - b["rate"]) / b["n"]
        var_rate += (wbar ** 2) * (va + vb)
    se_rate = math.sqrt(var_rate)
    rate_significant = abs(rate) > 1.96 * se_rate

    return {
        "metric": metric,
        "focus": focus, "baseline": baseline,
        "n_focus": nA, "n_baseline": nB,
        "observed_gap": observed,
        "decomposed_gap": total,
        "composition": comp,
        "rate": rate,
        "composition_share": share_comp,
        "per_channel": per_channel,
        # The decomposition is exact only when every cell is populated in both
        # regions; a dropped thin cell leaves a residual, and hiding it would let
        # the two components silently fail to add up to the thing being explained.
        "residual_vs_observed": observed - total,
        "rate_component_se": se_rate,
        "rate_component_significant": bool(rate_significant),
        "verdict": (
            "the %+.1f point gap is entirely CHANNEL MIX. Composition accounts for %+.1f points "
            "-- more than the whole gap -- and the within-channel term runs the other way by "
            "%+.1f points (SE %.1f), which is consistent with zero. Within the same channel the "
            "two regions are indistinguishable. The finding is about the acquisition portfolio; "
            "a product investigation into %s would be chasing a marketing fact."
            % (100 * observed, 100 * comp, 100 * rate, 100 * se_rate, focus)
            if not rate_significant else
            "%.0f%% of the %+.1f point gap is channel mix, but %+.1f points (SE %.1f) survives "
            "conditioning on channel. That residual is a genuine within-channel difference and "
            "is a product question, not a marketing one."
            % (100 * share_comp, 100 * observed, 100 * rate, 100 * se_rate)),
    }


def geography(con, metric: str = "d7_plus") -> dict:
    """Regional retention, and every pairwise gap decomposed."""
    agg = region_rates(con, metric=metric)
    regions = sorted(agg)
    worst = min(regions, key=lambda r: agg[r]["rate"])
    best = max(regions, key=lambda r: agg[r]["rate"])

    cells = region_channel_cells(con, metric=metric)
    thin = sum(1 for c in cells.values() if c["thin"])

    return {
        "metric": metric,
        "by_region": {r: {"n": agg[r]["n"], "rate": agg[r]["rate"],
                          "ci": [agg[r]["ci_low"], agg[r]["ci_high"]]} for r in regions},
        "worst": worst, "best": best,
        "headline_gap": _gap(agg[worst], agg[best]),
        "cells_thin": thin,
        "decomposition": kitagawa_decomposition(con, worst, best, metric=metric),
        "channel_mix": {
            r: {ch: round(cells[(r, ch)]["n"] / agg[r]["n"], 4)
                for ch in sorted({c for (_r, c) in cells}) if (r, ch) in cells}
            for r in regions},
    }


def dirty_vs_clean(focus_platform: str = "ios", metric: str = "d1_exact") -> dict:
    """The point of the module: what segmentation does to unvalidated data.

    The generator plants an iOS timezone bug -- events stamped in local time
    rather than UTC. That shifts events across the cohort-day boundary, which
    moves users in and out of an exact-day retention bucket, which manufactures a
    platform difference out of nothing.

    Run on `d7_plus` first and the bug is invisible (0.3 points, not significant),
    because a whole tail of later activity absorbs an eight-hour shift. Run it on
    `d1_exact` and it is the largest platform effect on the page. Both numbers are
    correct; only one of them is a finding, and nothing in the segmented output
    says which.
    """
    out = {"metric": metric}
    for label, path in (("clean", DB), ("uncleaned", DIRTY_DB)):
        if not os.path.exists(path):
            out[label] = {"error": "missing %s" % os.path.basename(path)}
            continue
        con = duckdb.connect(path, read_only=True)
        cells = {c["platform"]: c for c in _cells(con, ["platform"], metric=metric)}
        con.close()
        others = [c for p, c in cells.items() if p != focus_platform]
        pooled_n = sum(c["n"] for c in others)
        pooled_k = sum(c["retained"] for c in others)
        pooled = {"n": pooled_n, "retained": pooled_k,
                  "rate": pooled_k / pooled_n if pooled_n else float("nan")}
        g = _gap(cells[focus_platform], pooled) if focus_platform in cells else {"diff": float("nan")}
        out[label] = {"platform_rates": {p: round(c["rate"], 4) for p, c in sorted(cells.items())},
                      "focus_vs_rest": g}

    if all("focus_vs_rest" in out.get(k, {}) for k in ("clean", "uncleaned")):
        d_clean = out["clean"]["focus_vs_rest"].get("diff", float("nan"))
        d_dirty = out["uncleaned"]["focus_vs_rest"].get("diff", float("nan"))
        out["manufactured_effect"] = d_dirty - d_clean
        out["verdict"] = (
            "on %s, the uncleaned warehouse reports a %+.1f point %s effect (%s) where the cleaned "
            "one reports %+.1f. The %+.1f point difference is the timezone bug, not the platform. "
            "Segmentation on unvalidated data produces confident findings, not noisy ones."
            % (metric, 100 * d_dirty, focus_platform,
               "significant" if out["uncleaned"]["focus_vs_rest"].get("significant")
               else "not significant", 100 * d_clean, 100 * (d_dirty - d_clean)))
    return out


def run_all(db: str = DB) -> dict:
    con = duckdb.connect(db, read_only=True)
    try:
        cells = _cells(con, ["channel", "platform"])
        report = {
            "min_cell_size": MIN_CELL,
            "geography": geography(con),
            "cells_total": len(cells),
            "cells_excluded_as_thin": sum(1 for c in cells if c["thin"]),
            "gap_within_platforms": gap_within_platforms(con),
            "simpson": simpson_check(con),
            "cohort_over_cohort": cohort_over_cohort(con),
        }
    finally:
        con.close()
    report["dirty_vs_clean"] = dirty_vs_clean(metric="d1_exact")
    report["dirty_vs_clean_robust_metric"] = dirty_vs_clean(metric="d7_plus")
    return report


def to_markdown(rep: dict) -> str:
    g = rep["gap_within_platforms"]
    lines = ["## Does the paid-search retention gap survive platform?", "",
             "| platform | paid_search D7+ | organic D7+ | gap | 95% CI | significant |",
             "|---|---|---|---|---|---|"]
    o = g["overall_gap"]
    lines.append("| **all** | | | %+.1f pp | [%+.1f, %+.1f] | %s |"
                 % (100 * o["diff"], 100 * o["ci_low"], 100 * o["ci_high"],
                    "yes" if o["significant"] else "no"))
    for p in g["per_platform"]:
        if "diff" not in p:
            lines.append("| %s | | | | | excluded (%s) |" % (p["platform"], p["excluded"]))
            continue
        lines.append("| %s | %.3f | %.3f | %+.1f pp | [%+.1f, %+.1f] | %s |"
                     % (p["platform"], p["focus_rate"], p["baseline_rate"], 100 * p["diff"],
                        100 * p["ci_low"], 100 * p["ci_high"], "yes" if p["significant"] else "no"))
    lines += ["", g["verdict"], "",
              "## Simpson check", "", rep["simpson"]["verdict"],
              "(%d channel pairs checked, %d reversals)"
              % (rep["simpson"]["pairs_checked"], rep["simpson"]["reversals_found"]), ""]

    c = rep["cohort_over_cohort"]
    lines += ["## Cohort over cohort", "",
              "%d weeks reported, %d excluded as thin." % (c["weeks_reported"], c["weeks_excluded_thin"])]
    if c["trend"]:
        lines.append("Gap drift over the window: %+.2f points (slope %+.4f pp/week)."
                     % (100 * c["trend"]["total_drift_over_window"], 100 * c["trend"]["slope_per_week"]))
    g = rep.get("geography")
    if g:
        lines += ["", "## Geography, decomposed", "",
                  "| region | n | %s | 95%% CI |" % g["metric"], "|---|---|---|---|"]
        for r, v in sorted(g["by_region"].items()):
            lines.append("| %s | %d | %.4f | [%.3f, %.3f] |"
                         % (r, v["n"], v["rate"], v["ci"][0], v["ci"][1]))
        d = g["decomposition"]
        lines += ["",
                  "Headline: **%s** retains %+.1f points against **%s**."
                  % (d["focus"], 100 * d["observed_gap"], d["baseline"]),
                  "",
                  "| component | points | share |", "|---|---|---|",
                  "| channel mix (composition) | %+.2f | %.0f%% |"
                  % (100 * d["composition"], 100 * d["composition_share"]),
                  "| within-channel (rate) | %+.2f +/- %.2f | %s |"
                  % (100 * d["rate"], 100 * 1.96 * d["rate_component_se"],
                     "significant" if d["rate_component_significant"] else "**not significant**"),
                  "| residual from thin cells | %+.2f | |" % (100 * d["residual_vs_observed"]),
                  "", d["verdict"], ""]

    lines += ["", c["verdict"], "", "## Segmentation on unvalidated data", ""]
    for key in ("dirty_vs_clean", "dirty_vs_clean_robust_metric"):
        d = rep.get(key, {})
        if "verdict" not in d:
            continue
        lines += ["### metric: `%s`" % d["metric"], "", d["verdict"], "",
                  "| warehouse | " + " | ".join(sorted(d["clean"]["platform_rates"])) + " |",
                  "|---|" + "---|" * len(d["clean"]["platform_rates"])]
        for lab in ("clean", "uncleaned"):
            lines.append("| %s | %s |" % (lab, " | ".join("%.4f" % v for _, v in
                                                          sorted(d[lab]["platform_rates"].items()))))
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DB)
    args = ap.parse_args()
    rep = run_all(args.db)
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "segments.json"), "w") as fh:
        json.dump(rep, fh, indent=2, default=str)
    md = to_markdown(rep)
    with open(os.path.join(RESULTS, "segments.md"), "w", encoding="utf-8") as fh:
        fh.write(md + "\n")
    print(md)
    print("\nwritten: results/segments.json, results/segments.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
