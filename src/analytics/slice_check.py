"""Does a headline comparison survive being cut by everything else?

A channel comparison pooled over the whole population is one number with every
user mixed together. It cannot distinguish two very different worlds:

  * the effect is in the channel, and shows up in every device and region;
  * the effect is really a device or a region, and the channel split is standing
    in for it because the channels have different device and region mixes.

Those have opposite responses. The first is a marketing problem, the second is a
product problem, and shipping the wrong one wastes a quarter.

This module cuts the comparison by each dimension in turn and reports whether it
still holds. It is not a causal test and does not pretend to be: only a
randomised experiment settles direction. What it does is the strongest thing
observational data supports, which is checking that the finding is not an
artifact of composition.

## Why a naive reversal count is wrong

The obvious implementation counts slices where the sign flips and calls that a
warning. That is a bad detector. Cut a population finely enough and some slice
flips sign from sampling noise alone, so the count grows with the number of
slices rather than with the evidence.

Two guards, both reported rather than applied silently:

  * a reversal is only called REAL when the gap's interval excludes zero. A sign
    flip that cannot be told from zero is noise, and is labelled as such.
  * a slice that could not have detected a reversal of the headline size is
    marked UNDERPOWERED rather than counted as agreement. "No reversals" across
    slices too small to resolve one is not evidence of anything, and reporting it
    as confirmation is how a check like this becomes a rubber stamp.

Thin cells are suppressed on the same threshold the rest of the segmentation
uses, because a rate computed on forty users looks precise and is noise.
"""
from __future__ import annotations

import argparse
import json
import math
import os

import duckdb

from .segments import DB, MIN_CELL, _gap, wilson

CONVERSION_CELLS = """
SELECT {dims},
       COUNT(DISTINCT u.user_id) AS n,
       COUNT(DISTINCT CASE WHEN s.converted = 1 THEN u.user_id END) AS k
FROM stg_users u
LEFT JOIN mart_sessions s USING (user_id)
GROUP BY {dims}
ORDER BY {dims}
"""

RETENTION_CELLS = """
WITH activity AS (
    SELECT DISTINCT u.user_id,
           DATE_DIFF('day', u.signup_date, CAST(e.event_ts AS DATE)) AS day_number
    FROM stg_events e
    JOIN stg_users u USING (user_id)
    WHERE CAST(e.event_ts AS DATE) >= u.signup_date
)
SELECT {dims},
       COUNT(DISTINCT u.user_id) AS n,
       COUNT(DISTINCT CASE WHEN {pred} THEN a.user_id END) AS k
FROM stg_users u
LEFT JOIN activity a USING (user_id)
GROUP BY {dims}
ORDER BY {dims}
"""

METRICS = {
    "conversion": None,
    "d7_plus": "a.day_number >= 7",
    "d1_exact": "a.day_number = 1",
}


def _cells(con, dims: list[str], metric: str) -> list[dict]:
    """Rate, count and interval for every cell of the given dimensions."""
    qualified = ", ".join("u." + d for d in dims)
    if metric == "conversion":
        sql = CONVERSION_CELLS.format(dims=qualified)
    else:
        sql = RETENTION_CELLS.format(dims=qualified, pred=METRICS[metric])
    out = []
    for row in con.execute(sql).fetchall():
        n, k = int(row[-2]), int(row[-1])
        lo, hi = wilson(k, n)
        out.append({**{d: row[i] for i, d in enumerate(dims)},
                    "n": n, "k": k,
                    "rate": k / n if n else float("nan"),
                    "ci_low": lo, "ci_high": hi,
                    "thin": n < MIN_CELL})
    return out


def _compare(a: dict, b: dict, headline_gap: float) -> dict:
    """One focus-versus-baseline comparison inside a single slice.

    `headline_gap` is the pooled effect. It is the yardstick for whether this
    slice was powered: if the slice cannot resolve an effect that size, it can
    neither confirm nor refute the headline, and saying "it held here" would be
    claiming information the slice does not contain.
    """
    g = _gap(a, b)
    diff = g["diff"]
    half = 1.96 * g["se"] if g.get("se") is not None else float("nan")
    powered = half < abs(headline_gap) if half == half else False

    if a["thin"] or b["thin"]:
        verdict = "suppressed"
    elif diff * headline_gap < 0 and g["significant"]:
        verdict = "reversed"
    elif diff * headline_gap < 0:
        verdict = "reversed_not_significant"
    elif not powered:
        verdict = "underpowered"
    elif g["significant"]:
        verdict = "holds"
    else:
        verdict = "not_significant"

    return {
        "focus_n": a["n"], "focus_rate": a["rate"],
        "baseline_n": b["n"], "baseline_rate": b["rate"],
        "diff": diff,
        "ratio": (a["rate"] / b["rate"]) if b["rate"] else float("nan"),
        "ci_low": g.get("ci_low"), "ci_high": g.get("ci_high"),
        "significant": g["significant"],
        "powered": powered,
        "min_n": min(a["n"], b["n"]),
        "verdict": verdict,
    }


def slice_check(con, focus: str = "organic", baseline: str = "paid_search",
                metric: str = "conversion",
                dims: tuple[str, ...] = ("platform", "region"),
                by: str = "channel") -> dict:
    """Cut a comparison by each dimension in turn and by their interaction.

    `by` is the column the comparison is drawn from. It defaults to channel, but
    pointing it at region and slicing by channel is how this module gets tested
    against a case where the pooled number is known to be composition.
    """
    if metric not in METRICS:
        raise ValueError("unknown metric %r, expected one of %s" % (metric, sorted(METRICS)))
    if by in dims:
        raise ValueError("cannot slice by the same column being compared (%r)" % by)

    pooled = {c[by]: c for c in _cells(con, [by], metric)}
    if focus not in pooled or baseline not in pooled:
        raise ValueError("%s not present: %r or %r" % (by, focus, baseline))

    headline = _gap(pooled[focus], pooled[baseline])
    headline_gap = headline["diff"]

    groups = []
    for d in dims:
        cells = {}
        for c in _cells(con, [d, by], metric):
            cells.setdefault(c[d], {})[c[by]] = c
        rows = []
        for key in sorted(cells):
            pair = cells[key]
            if focus not in pair or baseline not in pair:
                continue
            rows.append({"slice": str(key),
                         **_compare(pair[focus], pair[baseline], headline_gap)})
        groups.append({"dimension": d, "rows": rows})

    # The interaction. A finding can survive every one-way cut and still be
    # carried by one corner of the joint distribution, which only a crossed cut
    # can show.
    inter_rows = []
    if len(dims) >= 2:
        cells = {}
        for c in _cells(con, [dims[0], dims[1], by], metric):
            cells.setdefault((c[dims[0]], c[dims[1]]), {})[c[by]] = c
        for key in sorted(cells):
            pair = cells[key]
            if focus not in pair or baseline not in pair:
                continue
            inter_rows.append({"slice": "%s/%s" % key,
                               **_compare(pair[focus], pair[baseline], headline_gap)})
        groups.append({"dimension": " x ".join(dims[:2]), "rows": inter_rows})

    every = [r for g in groups for r in g["rows"]]
    counts = {}
    for r in every:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1

    real_reversals = counts.get("reversed", 0)
    evaluated = [r for r in every if r["verdict"] != "suppressed"]
    powered = [r for r in evaluated if r["powered"]]

    # A powered slice that shows no effect is evidence AGAINST the headline, not
    # for it. Counting it as agreement because it did not reverse is the mistake
    # this module exists to prevent, and the first draft made it: run against a
    # pooled gap that is entirely composition, every slice came back "not
    # significant" and the summary read HOLDS.
    holds = [r for r in powered if r["verdict"] == "holds"]
    vanished = [r for r in powered if r["verdict"] == "not_significant"]

    if real_reversals:
        summary = ("REVERSES. %d powered slice(s) run the other way with an interval excluding "
                   "zero. The pooled number is at least partly composition, so decompose before "
                   "acting on it." % real_reversals)
    elif not powered:
        summary = ("INCONCLUSIVE. No slice was powered to detect an effect the size of the "
                   "headline, so nothing here supports or refutes it.")
    elif len(vanished) >= len(holds):
        summary = ("COLLAPSES. The effect survives in %d powered slice(s) and disappears in %d. "
                   "A pooled gap that vanishes once you condition is composition: the "
                   "difference is in what each %s is made of, not in %s itself."
                   % (len(holds), len(vanished), by, by))
    elif vanished:
        summary = ("WEAKENS. Holds in %d powered slice(s) but disappears in %d, so the pooled "
                   "number is part real and part composition. Decompose before quoting it."
                   % (len(holds), len(vanished)))
    elif len(powered) < len(evaluated):
        summary = ("HOLDS in every powered slice (%d of %d evaluated; the other %d were too "
                   "small to resolve the headline effect and are not counted as agreement)."
                   % (len(powered), len(evaluated), len(evaluated) - len(powered)))
    else:
        summary = ("HOLDS in all %d slices, every one powered to detect a reversal. "
                   "Consistent with the effect being in %s rather than in %s."
                   % (len(evaluated), by, " or ".join(dims)))

    return {
        "metric": metric,
        "by": by,
        "focus": focus,
        "baseline": baseline,
        "headline": {
            "focus_n": pooled[focus]["n"], "focus_rate": pooled[focus]["rate"],
            "baseline_n": pooled[baseline]["n"], "baseline_rate": pooled[baseline]["rate"],
            "diff": headline_gap,
            "ratio": (pooled[focus]["rate"] / pooled[baseline]["rate"]
                      if pooled[baseline]["rate"] else float("nan")),
            "significant": headline["significant"],
        },
        "min_cell": MIN_CELL,
        "groups": groups,
        "counts": counts,
        "slices_evaluated": len(evaluated),
        "slices_powered": len(powered),
        "slices_holding": len(holds),
        "slices_vanished": len(vanished),
        "real_reversals": real_reversals,
        "summary": summary,
        "caveat": ("Observational. Consistency across slices is evidence against a composition "
                   "artifact, not evidence of causation. Only a randomised experiment settles "
                   "direction."),
    }


SYMBOL = {
    "holds": "holds",
    "reversed": "REVERSED",
    "reversed_not_significant": "flipped (n.s.)",
    "underpowered": "underpowered",
    "not_significant": "not sig.",
    "suppressed": "suppressed",
}


def to_markdown(rep: dict) -> str:
    h = rep["headline"]
    out = ["# Slice check: %s vs %s (%s)" % (rep["focus"], rep["baseline"], rep["metric"]), ""]
    out += ["Pooled: **%s %.2f%%** (n=%d) vs **%s %.2f%%** (n=%d), gap %.2f pp, ratio %.2fx."
            % (rep["focus"], 100 * h["focus_rate"], h["focus_n"],
               rep["baseline"], 100 * h["baseline_rate"], h["baseline_n"],
               100 * h["diff"], h["ratio"]), ""]
    for g in rep["groups"]:
        out += ["## by %s" % g["dimension"], "",
                "| slice | %s | %s | gap (pp) | ratio | min n | verdict |" % (rep["focus"], rep["baseline"]),
                "|---|---|---|---|---|---|---|"]
        for r in g["rows"]:
            out.append("| %s | %.2f%% | %.2f%% | %+.2f | %.2fx | %d | %s |"
                       % (r["slice"], 100 * r["focus_rate"], 100 * r["baseline_rate"],
                          100 * r["diff"], r["ratio"], r["min_n"], SYMBOL[r["verdict"]]))
        out.append("")
    out += ["**%s**" % rep["summary"], "", rep["caveat"], ""]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", default=DB)
    ap.add_argument("--by", default="channel",
                    help="column the comparison is drawn from (default: channel)")
    ap.add_argument("--focus", default="organic")
    ap.add_argument("--baseline", default="paid_search")
    ap.add_argument("--metric", default="conversion", choices=sorted(METRICS))
    ap.add_argument("--dims", nargs="+", default=["platform", "region"])
    ap.add_argument("--json", action="store_true", help="emit the report as JSON")
    args = ap.parse_args()

    con = duckdb.connect(args.db, read_only=True)
    rep = slice_check(con, focus=args.focus, baseline=args.baseline,
                      metric=args.metric, dims=tuple(args.dims), by=args.by)

    if args.json:
        print(json.dumps(rep, indent=2, default=float))
    else:
        print(to_markdown(rep))

    out_dir = os.path.join(os.path.dirname(DB), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "slice_check_%s_%s.json" % (args.by, args.metric))
    with open(path, "w") as fh:
        json.dump(rep, fh, indent=2, default=float)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
