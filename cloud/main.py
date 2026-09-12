"""Cloud Function entrypoints for the events -> BigQuery pipeline.

Two triggers, one processing path:

* ``on_gcs_finalize`` — a CloudEvent fired when a raw-events file lands in the
  landing bucket (event-driven ingestion).
* ``scheduled_load`` — a Pub/Sub message from Cloud Scheduler, carrying a date;
  it processes ``events/<date>.jsonl`` (the managed daily orchestration).

Both call ``process_object``: read the NDJSON object, aggregate with the pure
``transform.aggregate``, and load the day into BigQuery (per-day idempotent).

Config comes from env vars set by Terraform: ``GCP_PROJECT``, ``BQ_DATASET``,
``BQ_TABLE``, ``LANDING_BUCKET``.
"""
from __future__ import annotations

import base64
import io
import json
import os

import functions_framework
import pandas as pd

from transform import aggregate, to_json_rows
from bq import load_rows


def _cfg() -> dict:
    return {
        "project": os.environ["GCP_PROJECT"],
        "dataset": os.environ.get("BQ_DATASET", "product_analytics"),
        "table": os.environ.get("BQ_TABLE", "daily_metrics"),
        "bucket": os.environ.get("LANDING_BUCKET", ""),
    }


def read_ndjson(bucket: str, name: str) -> pd.DataFrame:
    """Download a newline-delimited JSON object from GCS into a DataFrame."""
    from google.cloud import storage

    blob = storage.Client().bucket(bucket).blob(name)
    text = blob.download_as_text()
    return pd.read_json(io.StringIO(text), lines=True)


def process_object(bucket: str, name: str) -> dict:
    """The whole path for one object: read -> aggregate -> load to BigQuery."""
    cfg = _cfg()
    events = read_ndjson(bucket, name)
    metrics = aggregate(events)
    n = load_rows(to_json_rows(metrics), cfg["project"], cfg["dataset"], cfg["table"])
    return {"object": f"gs://{bucket}/{name}", "events_in": int(len(events)),
            "rows_loaded": int(n)}


@functions_framework.cloud_event
def on_gcs_finalize(cloud_event):
    """GCS object-finalize trigger: process the object that just landed."""
    data = cloud_event.data
    bucket, name = data["bucket"], data["name"]
    if not name.endswith(".jsonl"):
        print(f"skipping non-events object gs://{bucket}/{name}")
        return
    result = process_object(bucket, name)
    print(json.dumps(result))
    return result


@functions_framework.cloud_event
def scheduled_load(cloud_event):
    """Cloud Scheduler -> Pub/Sub trigger. Message payload: {"date": "YYYY-MM-DD"}.

    Processes events/<date>.jsonl from the landing bucket. With no date it
    processes yesterday (UTC), the usual 'load yesterday's data' cadence.
    """
    cfg = _cfg()
    payload = {}
    msg = (cloud_event.data or {}).get("message", {})
    if msg.get("data"):
        try:
            payload = json.loads(base64.b64decode(msg["data"]).decode())
        except Exception:
            payload = {}
    date = payload.get("date")
    if not date:
        date = (pd.Timestamp.utcnow() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    result = process_object(cfg["bucket"], f"events/{date}.jsonl")
    print(json.dumps(result))
    return result
