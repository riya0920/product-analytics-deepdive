"""Split events.parquet into daily NDJSON files — the pipeline's raw input.

    python -m cloud.seed_gcs                       # write files under cloud/_landing/events/
    python -m cloud.seed_gcs --upload BUCKET       # also upload to gs://BUCKET/events/

Each ``events/<date>.jsonl`` is what an app's event export would drop into the
landing bucket; the Cloud Function processes one when it lands (or on schedule).
Uses data ``analytics.generate`` already produced — no new dataset invented.
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def write_daily(events: pd.DataFrame, out_dir: str) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    df = events.copy()
    df["event_ts"] = pd.to_datetime(df["event_ts"], utc=True)
    df["_date"] = df["event_ts"].dt.strftime("%Y-%m-%d")
    # ISO-8601 timestamps so the JSON round-trips cleanly.
    df["event_ts"] = df["event_ts"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    paths = []
    for date, g in df.groupby("_date"):
        p = os.path.join(out_dir, f"{date}.jsonl")
        g.drop(columns="_date").to_json(p, orient="records", lines=True)
        paths.append(p)
    return paths


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=os.path.join(REPO, "data", "events.parquet"))
    ap.add_argument("--out", default=os.path.join(HERE, "_landing", "events"))
    ap.add_argument("--upload", help="GCS bucket to upload events/<date>.jsonl to")
    args = ap.parse_args()

    events = pd.read_parquet(args.parquet)
    paths = write_daily(events, args.out)
    print(f"wrote {len(paths)} daily files to {args.out}")

    if args.upload:
        from google.cloud import storage
        bucket = storage.Client().bucket(args.upload)
        for p in paths:
            name = f"events/{os.path.basename(p)}"
            bucket.blob(name).upload_from_filename(p)
        print(f"uploaded {len(paths)} files to gs://{args.upload}/events/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
