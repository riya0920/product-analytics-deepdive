"""Answer the stated business question and emit the numbers the memo quotes.

The question, fixed before any query was written:

    "Where do we lose new users in week one, and what is it worth to fix?"

Everything here serves that question. There is no exploratory chart wall: an EDA
without a decision is tourism, and the memo has to end with a recommendation and
a number someone can act on.

    python -m analytics.analysis
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy import stats as sps

from .pipeline import DB, build

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS = os.path.join(ROOT, "results")


def funnel(con) -> list:
    return con.execute("SELECT * FROM mart_funnel ORDER BY step_index").fetchdf().to_dict("records")


def retention_by_channel(con) -> list:
    """Retention by channel, with right-censored cohorts EXCLUDED per horizon.

    The trap this avoids: a user who signed up 3 days before the data ends
    cannot possibly have a D7 event, so including their cohort in the D7
    denominator drives the rate toward zero as a pure artefact of the window.
    Running this without the exclusion reported D30 retention of 0.7%, which is
    not a product finding -- it is a measurement bug.

    Each horizon therefore gets its OWN eligible cohort set: only cohorts whose
    signup date is at least N days before the last observed event count toward
    the DN rate.
    """
    return con.execute(
        """
        WITH bounds AS (SELECT MAX(CAST(event_ts AS DATE)) AS last_date FROM stg_events)
        SELECT
            m.channel,
            SUM(m.cohort_users) AS users_all_cohorts,
            SUM(CASE WHEN DATE_DIFF('day', m.signup_date, b.last_date) >= 1
                     THEN m.cohort_users END) AS eligible_d1,
            ROUND(SUM(CASE WHEN DATE_DIFF('day', m.signup_date, b.last_date) >= 1
                           THEN m.d1_exact END)::DOUBLE
                  / NULLIF(SUM(CASE WHEN DATE_DIFF('day', m.signup_date, b.last_date) >= 1
                                    THEN m.cohort_users END), 0), 4) AS d1_exact_rate,
            ROUND(SUM(CASE WHEN DATE_DIFF('day', m.signup_date, b.last_date) >= 7
                           THEN m.d7_plus END)::DOUBLE
                  / NULLIF(SUM(CASE WHEN DATE_DIFF('day', m.signup_date, b.last_date) >= 7
                                    THEN m.cohort_users END), 0), 4) AS d7_plus_rate,
            SUM(CASE WHEN DATE_DIFF('day', m.signup_date, b.last_date) >= 30
                     THEN m.cohort_users END) AS eligible_d30,
            ROUND(SUM(CASE WHEN DATE_DIFF('day', m.signup_date, b.last_date) >= 30
                           THEN m.d30_plus END)::DOUBLE
                  / NULLIF(SUM(CASE WHEN DATE_DIFF('day', m.signup_date, b.last_date) >= 30
                                    THEN m.cohort_users END), 0), 4) AS d30_plus_rate
        FROM mart_retention m, bounds b
        GROUP BY m.channel
        ORDER BY d7_plus_rate DESC
        """
    ).fetchdf().to_dict("records")


def week_one_dropoff(con) -> dict:
    """The core question: who is gone after seven days, split by channel."""
    return con.execute(
        """
        WITH activity AS (
            SELECT
                u.user_id,
                u.channel,
                MAX(DATE_DIFF('day', u.signup_date, CAST(e.event_ts AS DATE))) AS last_day
            FROM stg_users u
            JOIN stg_events e USING (user_id)
            GROUP BY u.user_id, u.channel
        )
        SELECT
            COUNT(*)                                                    AS users,
            COUNT(*) FILTER (WHERE last_day < 7)                        AS lost_in_week_one,
            ROUND(COUNT(*) FILTER (WHERE last_day < 7)::DOUBLE / COUNT(*), 4) AS week_one_loss_rate
        FROM activity
        """
    ).fetchdf().to_dict("records")[0]


def channel_conversion_test(con) -> dict:
    """One statistical comparison, done properly.

    Compares purchase rate for paid_search against organic with a two-proportion
    z-test and a confidence interval on the difference -- not "the bars look
    different". The CI is the part that makes the memo's dollar figure defensible,
    because it bounds how wrong the point estimate can be.
    """
    rows = con.execute(
        """
        SELECT
            u.channel,
            COUNT(DISTINCT u.user_id)                                          AS users,
            COUNT(DISTINCT CASE WHEN e.event_name = 'purchase' THEN u.user_id END) AS purchasers
        FROM stg_users u
        LEFT JOIN stg_events e USING (user_id)
        WHERE u.channel IN ('paid_search', 'organic')
        GROUP BY u.channel
        """
    ).fetchdf().set_index("channel").to_dict("index")

    n1, x1 = rows["organic"]["users"], rows["organic"]["purchasers"]
    n2, x2 = rows["paid_search"]["users"], rows["paid_search"]["purchasers"]
    p1, p2 = x1 / n1, x2 / n2
    pooled = (x1 + x2) / (n1 + n2)
    se_pooled = np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    se_unpooled = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    z = (p2 - p1) / se_pooled
    p_value = 2 * sps.norm.sf(abs(z))
    crit = sps.norm.ppf(0.975)
    return {
        "organic_rate": p1,
        "paid_search_rate": p2,
        "absolute_difference": p2 - p1,
        "relative_difference": (p2 - p1) / p1,
        "ci95_absolute": [(p2 - p1) - crit * se_unpooled, (p2 - p1) + crit * se_unpooled],
        "p_value": float(p_value),
        "n_organic": int(n1),
        "n_paid_search": int(n2),
        "method": "two-proportion z-test, pooled SE for the p-value, unpooled SE for the CI",
    }


def opportunity_sizing(con, funnel_rows: list, test: dict) -> dict:
    """The dollar number, with every assumption named.

    Deliberately conservative and explicitly bounded, because "fixing step 3 is
    worth $X" is the claim that gets challenged first in an interview and the
    assumptions are the whole defence.
    """
    worst = max(
        (r for r in funnel_rows if r["users_previous_step"] is not None),
        key=lambda r: (r["users_previous_step"] - r["users_reached"]),
    )
    arpu_purchaser = con.execute(
        """
        SELECT SUM(revenue) / NULLIF(COUNT(DISTINCT user_id), 0)
        FROM stg_events
        WHERE user_id IN (SELECT DISTINCT user_id FROM stg_events WHERE event_name = 'purchase')
        """
    ).fetchone()[0]

    users_lost = int(worst["users_previous_step"] - worst["users_reached"])
    downstream_rate = float(funnel_rows[-1]["users_reached"]) / float(worst["users_reached"])

    # Three scenarios rather than one number. A single point estimate invites the
    # reader to treat it as precise; the range is the honest object.
    scenarios = {}
    for label, recovery in (("conservative", 0.05), ("central", 0.10), ("optimistic", 0.20)):
        recovered = users_lost * recovery
        converted = recovered * downstream_rate
        scenarios[label] = {
            "recovery_rate_assumed": recovery,
            "users_recovered": round(recovered),
            "incremental_purchasers": round(converted),
            "incremental_revenue": round(converted * arpu_purchaser, 2),
        }

    return {
        "worst_step": worst["step"],
        "users_lost_at_worst_step": users_lost,
        "step_conversion": worst["step_conversion"],
        "arpu_per_purchaser": float(arpu_purchaser),
        "downstream_purchase_rate_from_worst_step": downstream_rate,
        "scenarios": scenarios,
        "assumptions": [
            "Recovered users convert downstream at the SAME rate as users who already passed the step. "
            "This is optimistic: users who dropped are systematically less engaged, so treat every "
            "figure as an upper bound.",
            "ARPU per purchaser is computed over the full observation window, not a fixed horizon, "
            "so late cohorts contribute less revenue and the figure is mildly understated.",
            "No cost of the fix is included. The number is gross opportunity, not ROI.",
            "Revenue is simulated. The magnitude is only meaningful relative to other numbers in "
            "this dataset, never as a real-world dollar claim.",
        ],
    }


def run() -> dict:
    con = build()
    rows = funnel(con)
    test = channel_conversion_test(con)
    out = {
        "question": "Where do we lose new users in week one, and what is it worth to fix?",
        "funnel": rows,
        "retention_by_channel": retention_by_channel(con),
        "week_one": week_one_dropoff(con),
        "channel_comparison": test,
        "opportunity": opportunity_sizing(con, rows, test),
        "data_quality": con.execute("SELECT * FROM audit_data_quality").fetchdf().to_dict("records")[0],
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(RESULTS, "analysis.json"))
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    result = run()
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2, default=float)
    print(json.dumps(result, indent=2, default=float))
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
