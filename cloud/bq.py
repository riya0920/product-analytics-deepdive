"""BigQuery schema and load for the daily-metrics table.

The schema is defined once, in JSON that both this loader and the Terraform table
resource read, so the deployed table and the code that writes it cannot drift.
The table is partitioned by ``event_date`` and clustered by ``channel``: the two
predicates a product dashboard filters on, so a day/channel query scans one
partition instead of the whole table (and stays inside the BigQuery free tier).
"""
from __future__ import annotations

import json
import os

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.json")


def load_schema() -> list[dict]:
    with open(SCHEMA_PATH) as f:
        return json.load(f)


def bq_schema():
    """The schema as google.cloud.bigquery.SchemaField objects (lazy import)."""
    from google.cloud import bigquery

    type_map = {
        "DATE": "DATE", "STRING": "STRING", "INTEGER": "INTEGER",
        "FLOAT": "FLOAT", "NUMERIC": "NUMERIC",
    }
    return [
        bigquery.SchemaField(f["name"], type_map[f["type"]], mode=f.get("mode", "NULLABLE"))
        for f in load_schema()
    ]


def load_rows(rows: list[dict], project: str, dataset: str, table: str,
              write_disposition: str = "WRITE_TRUNCATE_ON_DATE") -> int:
    """Load rows into ``project.dataset.table``.

    Idempotent by design: the default deletes the partitions present in ``rows``
    before inserting, so re-running a day's file replaces that day rather than
    double-counting it. Returns the number of rows loaded.
    """
    from google.cloud import bigquery

    client = bigquery.Client(project=project)
    table_id = f"{project}.{dataset}.{table}"

    if write_disposition == "WRITE_TRUNCATE_ON_DATE" and rows:
        dates = sorted({r["event_date"] for r in rows})
        # Delete just the affected partitions, then append (per-day idempotency).
        in_list = ", ".join(f"DATE('{d}')" for d in dates)
        client.query(
            f"DELETE FROM `{table_id}` WHERE event_date IN ({in_list})"
        ).result()
        disposition = bigquery.WriteDisposition.WRITE_APPEND
    else:
        disposition = getattr(bigquery.WriteDisposition, write_disposition,
                              bigquery.WriteDisposition.WRITE_APPEND)

    job_config = bigquery.LoadJobConfig(
        schema=bq_schema(),
        write_disposition=disposition,
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
    )
    job = client.load_table_from_json(rows, table_id, job_config=job_config)
    job.result()
    return len(rows)
