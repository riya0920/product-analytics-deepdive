"""Pure transform: raw product-analytics events -> a daily metrics table.

This is deliberately dependency-light (pandas only) and has no GCP imports, so it
is unit-testable on its own and is the single place the aggregation logic lives.
The Cloud Function (``main.py``) is a thin wrapper: read object -> ``aggregate``
-> load to BigQuery.

Input rows are the events produced by ``analytics.generate`` and stored in
``data/events.parquet`` (columns: event_id, user_id, event_name, event_ts,
channel, platform, region, revenue). Output is one row per
(event_date, channel, platform, region) with the funnel counts, unique users,
revenue and a conversion rate.
"""
from __future__ import annotations

import pandas as pd

# The funnel stages we count, and the output column each maps to.
STAGES = {
    "session_start": "sessions",
    "signup": "signups",
    "first_search": "searches",
    "add_to_cart": "add_to_cart",
    "activate": "activations",
    "purchase": "purchases",
}
DIMENSIONS = ["event_date", "channel", "platform", "region"]
OUTPUT_COLUMNS = (
    DIMENSIONS
    + list(STAGES.values())
    + ["unique_users", "revenue", "conversion_rate"]
)


def aggregate(events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw events to the daily metrics grain."""
    df = events.copy()
    # event_date from the UTC timestamp; robust to str/naive/aware inputs.
    df["event_date"] = pd.to_datetime(df["event_ts"], utc=True).dt.date

    # Funnel counts: one column per stage via a pivot on event_name.
    counts = (
        df.assign(_n=1)
        .pivot_table(index=DIMENSIONS, columns="event_name", values="_n",
                     aggfunc="sum", fill_value=0)
    )
    for src, dst in STAGES.items():
        counts[dst] = counts[src] if src in counts.columns else 0
    counts = counts[list(STAGES.values())]

    # Unique users and revenue per cell.
    extras = df.groupby(DIMENSIONS).agg(
        unique_users=("user_id", "nunique"),
        revenue=("revenue", "sum"),
    )

    out = counts.join(extras).reset_index()
    # Conversion = purchases / sessions, guarded against divide-by-zero.
    out["conversion_rate"] = (
        out["purchases"] / out["sessions"].where(out["sessions"] > 0)
    ).fillna(0.0).round(6)

    # Stable dtypes for a clean BigQuery load.
    for c in list(STAGES.values()) + ["unique_users"]:
        out[c] = out[c].astype("int64")
    out["revenue"] = out["revenue"].astype("float64").round(2)
    out["event_date"] = out["event_date"].astype("string")  # ISO date for JSON/BQ
    return out[OUTPUT_COLUMNS].sort_values(DIMENSIONS).reset_index(drop=True)


def to_json_rows(df: pd.DataFrame) -> list[dict]:
    """Rows as JSON-serialisable dicts, the shape a BigQuery load_table expects."""
    return df.to_dict(orient="records")
