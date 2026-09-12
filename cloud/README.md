# Cloud pipeline — events → GCS → Cloud Function → BigQuery (GCP)

A cloud-native ELT path over the events this repo already generates
(`data/events.parquet`, produced by `analytics.generate` — no new dataset
invented). Raw event files land in Cloud Storage; a Cloud Function aggregates
them to a daily metrics table in BigQuery; a Cloud Scheduler job drives the daily
load. Everything is inside the GCP Always-Free tier at this volume.

```
 events.parquet ──seed_gcs──▶ gs://<proj>-events-landing/events/<date>.jsonl
                                          │ (object.finalized)
                                          ▼
                              Cloud Function  events-on-finalize
                              (transform.aggregate → BigQuery load)
                                          │
                                          ▼
                     BigQuery  product_analytics.daily_metrics
                     (partitioned by event_date, clustered by channel)
                                          ▲
                                          │ (Pub/Sub messagePublished)
                              Cloud Function  events-scheduled-load
                                          ▲
                                          │
                    Cloud Scheduler  "0 6 * * *"  →  Pub/Sub  events-daily-load
```

Two triggers, one code path (`cloud/main.py` → `process_object`): the
**event-driven** function fires when a file lands; the **scheduled** function is
Cloud Scheduler → Pub/Sub → function, loading the previous day at 06:00 UTC. The
aggregation itself is a pure, unit-tested function (`cloud/transform.py`).

**What it produces on the real data:** 380,798 raw events → **8,846** daily-metric
rows (2025-12-31 … 2026-07-29), 205,874 sessions, 12,551 purchases, $521,036
revenue, 6.10% overall session→purchase conversion. The table is partitioned by
`event_date` and clustered by `channel`, so a "last-7-days, paid_search" query
scans a few partitions, not the whole table.

## IAM

One dedicated service account (`events-pipeline-fn`) runs both functions, with the
narrowest roles that let the path work — no project-wide data access:

| Principal | Role | Scope | Why |
| --- | --- | --- | --- |
| `events-pipeline-fn` SA | `roles/storage.objectViewer` | landing bucket | read the raw file it was handed |
| `events-pipeline-fn` SA | `roles/bigquery.dataEditor` | `product_analytics` dataset | write rows into the one table (delete-partition + append) |
| `events-pipeline-fn` SA | `roles/bigquery.jobUser` | project | run the load/query jobs (narrowest role that permits jobs) |
| Eventarc/GCS agents | `roles/eventarc.eventReceiver`, `pubsub.publisher` | — | created by the Gen2 trigger; granted at deploy |

The function is **not** a project editor and cannot read other datasets or
buckets. Cloud Scheduler publishes to the Pub/Sub topic; the function is invoked
by Eventarc, not exposed as a public HTTP endpoint.

## Cost

Always-Free monthly allowances comfortably cover this workload:

| Service | Free allowance (monthly) | This pipeline |
| --- | --- | --- |
| Cloud Storage | 5 GB-months, 5k class-A ops | a few MB of JSONL, ~200 writes |
| Cloud Functions (Gen2/Run) | 2M invocations, 360k GB-s | ~30 invocations/mo (1/day + landings) |
| BigQuery storage | 10 GB | daily_metrics is < 5 MB |
| BigQuery query | 1 TB scanned | partition+cluster pruning keeps scans in MB |
| Cloud Scheduler | 3 jobs | 1 job |
| Pub/Sub | 10 GB | kilobytes |

**Expected steady-state cost: $0.** The only way to leave the free tier here is a
full-table scan on BigQuery without the partition filter — the table is
partitioned precisely to avoid that. (Costs are your responsibility to monitor;
set a budget alert.)

## Deploy (requires your GCP auth — I cannot run this for you)

```bash
gcloud auth application-default login
gcloud config set project <PROJECT_ID>

cd infra
terraform init
terraform apply -var project_id=<PROJECT_ID>      # bucket, BQ, 2 functions, scheduler, IAM

# seed the landing bucket from the repo's events (triggers the on-finalize function)
python -m cloud.seed_gcs --upload $(terraform -chdir=infra output -raw landing_bucket)

# or run the scheduled path once, now:
gcloud scheduler jobs run events-daily-load --location us-central1

# check the result
bq query --use_legacy_sql=false \
  'SELECT event_date, SUM(sessions) s, SUM(purchases) p FROM `<PROJECT_ID>.product_analytics.daily_metrics` GROUP BY 1 ORDER BY 1 DESC LIMIT 7'
```

## Run the logic locally (no GCP needed)

```bash
python -m pytest cloud/tests -q         # transform + mocked GCS/BigQuery wiring
cd infra && terraform validate          # the IaC is valid HCL
```

## Status

Built and tested locally: the transform is unit-tested on the real events, the
function's read→aggregate→load path is tested with the storage and BigQuery
clients mocked (loaded rows == transform output), and the Terraform validates.
It has **not** been `terraform apply`-ed to a live project from here — there is no
`gcloud`/credentials in this environment, and deploying to your project (with its
billing and IAM) is your step. The commands above are the whole of it.
