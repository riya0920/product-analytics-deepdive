"""Build the warehouse: raw -> staging -> marts, then run the assertion tests.

A minimal dbt-shaped runner. Models are plain .sql files with a declared
dependency order, materialised as views/tables in DuckDB; tests are .sql files
that must return ZERO rows. That is exactly dbt's contract, implemented in ~100
lines so the SQL stays the artifact rather than the tool configuration.

    python -m analytics.pipeline build
    python -m analytics.pipeline test
    python -m analytics.pipeline test --skip-cleaning   # prove the tests catch the planted defects
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import duckdb

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SQL = os.path.join(ROOT, "sql")
DATA = os.path.join(ROOT, "data")
DB = os.path.join(DATA, "warehouse.duckdb")
EVENTS = os.path.join(DATA, "events.parquet")

# Declared order = the DAG. Small enough to be explicit and obvious.
MODELS = [
    ("staging", "stg_events"),
    ("staging", "stg_users"),
    ("marts", "mart_sessions"),
    ("marts", "mart_funnel"),
    ("marts", "mart_retention"),
    ("marts", "audit_data_quality"),
]

# The "no cleaning" variant used to prove the tests have teeth: raw events pass
# straight through, defects intact.
PASSTHROUGH_STG_EVENTS = """
SELECT event_id, user_id, event_name, event_ts, CAST(event_ts AS DATE) AS event_date,
       channel, platform, revenue
FROM raw_events
"""


def read_model(folder: str, name: str) -> str:
    with open(os.path.join(SQL, folder, name + ".sql"), encoding="utf-8") as fh:
        return fh.read()


def build(db_path: str = DB, events: str = EVENTS, skip_cleaning: bool = False) -> duckdb.DuckDBPyConnection:
    if not os.path.exists(events):
        raise SystemExit("missing %s -- run `python -m analytics.generate` first" % events)
    os.makedirs(DATA, exist_ok=True)
    if os.path.exists(db_path):
        os.remove(db_path)

    con = duckdb.connect(db_path)
    con.execute("CREATE OR REPLACE TABLE raw_events AS SELECT * FROM read_parquet(?)", [events])

    for folder, name in MODELS:
        sql = PASSTHROUGH_STG_EVENTS if (skip_cleaning and name == "stg_events") else read_model(folder, name)
        con.execute("CREATE OR REPLACE TABLE %s AS %s" % (name, sql))
        n = con.execute("SELECT COUNT(*) FROM %s" % name).fetchone()[0]
        print("built %-22s %8d rows%s" % (name, n, "  [CLEANING SKIPPED]" if skip_cleaning and name == "stg_events" else ""))
    return con


def run_tests(con: duckdb.DuckDBPyConnection) -> int:
    """Every test must return zero rows. Returns the number of failures."""
    failures = 0
    for path in sorted(glob.glob(os.path.join(SQL, "tests", "*.sql"))):
        name = os.path.basename(path)[:-4]
        with open(path, encoding="utf-8") as fh:
            sql = fh.read()
        rows = con.execute(sql).fetchall()
        if rows:
            failures += 1
            print("FAIL %-45s %d offending row(s); first: %s" % (name, len(rows), rows[0]))
        else:
            print("PASS %s" % name)
    return failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["build", "test"])
    ap.add_argument("--skip-cleaning", action="store_true",
                    help="build staging WITHOUT the fixes, to prove the tests catch the planted defects")
    args = ap.parse_args()

    con = build(skip_cleaning=args.skip_cleaning)
    if args.command == "test":
        print()
        failures = run_tests(con)
        print()
        if args.skip_cleaning:
            if failures == 0:
                print("ERROR: the tests passed on deliberately dirty data. The tests are broken.")
                return 1
            print("%d test(s) failed on uncleaned data, as they must. The tests have teeth." % failures)
            return 0
        if failures:
            print("%d data-quality test(s) FAILED" % failures)
            return 1
        print("all data-quality tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
