"""Tests for the GCP pipeline: the pure transform, and the function wiring.

The transform is checked directly. The Cloud Function path (read GCS -> aggregate
-> load BigQuery) is checked with the storage and BigQuery clients mocked, so the
wiring is proven end to end without a real GCP project: the rows the function
'loads' must equal the transform's output for the same input.
"""
import os
import sys
import types
from unittest import mock

import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
CLOUD = os.path.dirname(HERE)
sys.path.insert(0, CLOUD)  # cloud/ is the function's source root (flat imports)

from transform import aggregate, OUTPUT_COLUMNS, STAGES  # noqa: E402


def _events():
    ts = pd.to_datetime("2026-01-02T10:00:00Z")
    rows = [
        # channel/platform/region cell A: 2 sessions, 1 purchase, revenue 30
        dict(event_id="1", user_id="u1", event_name="session_start", event_ts=ts, channel="email", platform="ios", region="amer", revenue=0.0),
        dict(event_id="2", user_id="u2", event_name="session_start", event_ts=ts, channel="email", platform="ios", region="amer", revenue=0.0),
        dict(event_id="3", user_id="u1", event_name="add_to_cart", event_ts=ts, channel="email", platform="ios", region="amer", revenue=0.0),
        dict(event_id="4", user_id="u1", event_name="purchase", event_ts=ts, channel="email", platform="ios", region="amer", revenue=30.0),
        # cell B, different day: 1 session, 0 purchases
        dict(event_id="5", user_id="u3", event_name="session_start", event_ts=pd.to_datetime("2026-01-03T09:00:00Z"), channel="social", platform="web", region="emea", revenue=0.0),
    ]
    return pd.DataFrame(rows)


def test_transform_grain_and_counts():
    out = aggregate(_events())
    assert list(out.columns) == OUTPUT_COLUMNS
    a = out[(out.channel == "email") & (out.event_date == "2026-01-02")].iloc[0]
    assert a.sessions == 2 and a.purchases == 1 and a.add_to_cart == 1
    assert a.unique_users == 2 and a.revenue == 30.0
    assert abs(a.conversion_rate - 0.5) < 1e-9  # 1 purchase / 2 sessions


def test_conversion_rate_is_zero_when_no_sessions():
    ts = pd.to_datetime("2026-01-02T10:00:00Z")
    df = pd.DataFrame([dict(event_id="1", user_id="u", event_name="signup", event_ts=ts,
                            channel="email", platform="ios", region="amer", revenue=0.0)])
    out = aggregate(df)
    assert out.iloc[0].sessions == 0 and out.iloc[0].conversion_rate == 0.0


def test_function_loads_transform_output(monkeypatch):
    """on_gcs_finalize: mock GCS (returns NDJSON) and BigQuery (captures rows);
    the loaded rows must equal aggregate() on the same events."""
    events = _events()
    ndjson = events.assign(
        event_ts=events.event_ts.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    ).to_json(orient="records", lines=True)

    # Fake google.cloud.storage: blob.download_as_text() -> our NDJSON.
    storage_mod = types.ModuleType("google.cloud.storage")
    blob = mock.Mock()
    blob.download_as_text.return_value = ndjson
    bucket = mock.Mock()
    bucket.blob.return_value = blob
    client = mock.Mock()
    client.bucket.return_value = bucket
    storage_mod.Client = mock.Mock(return_value=client)

    captured = {}

    def fake_load_rows(rows, project, dataset, table, **kw):
        captured["rows"] = rows
        captured["target"] = f"{project}.{dataset}.{table}"
        return len(rows)

    monkeypatch.setitem(sys.modules, "google.cloud.storage", storage_mod)
    monkeypatch.setenv("GCP_PROJECT", "demo-proj")
    monkeypatch.setenv("BQ_DATASET", "product_analytics")
    monkeypatch.setenv("BQ_TABLE", "daily_metrics")

    import main
    monkeypatch.setattr(main, "load_rows", fake_load_rows)

    ce = types.SimpleNamespace(data={"bucket": "landing", "name": "events/2026-01-02.jsonl"})
    result = main.on_gcs_finalize(ce)

    expected = aggregate(events).to_dict(orient="records")
    assert captured["rows"] == expected
    assert captured["target"] == "demo-proj.product_analytics.daily_metrics"
    assert result["rows_loaded"] == len(expected)


def test_schema_matches_transform_columns():
    from bq import load_schema
    names = [f["name"] for f in load_schema()]
    assert names == OUTPUT_COLUMNS
