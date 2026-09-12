# Events -> Cloud Storage -> Cloud Function -> BigQuery, with a scheduled trigger.
# Everything here stays inside the GCP Always-Free tier for this data volume
# (see cloud/README.md for the cost breakdown). `terraform apply` after auth.

terraform {
  required_version = ">= 1.5"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
    archive = {
      source  = "hashicorp/archive"
      version = "~> 2.4"
    }
  }
}

variable "project_id" { type = string }
variable "region" {
  type    = string
  default = "us-central1"
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# --- APIs this pipeline needs -------------------------------------------------
resource "google_project_service" "apis" {
  for_each = toset([
    "storage.googleapis.com",
    "bigquery.googleapis.com",
    "cloudfunctions.googleapis.com",
    "run.googleapis.com",
    "cloudbuild.googleapis.com",
    "cloudscheduler.googleapis.com",
    "pubsub.googleapis.com",
    "eventarc.googleapis.com",
  ])
  service            = each.value
  disable_on_destroy = false
}

# --- Landing bucket for raw events -------------------------------------------
resource "google_storage_bucket" "landing" {
  name                        = "${var.project_id}-events-landing"
  location                    = var.region
  uniform_bucket_level_access = true
  force_destroy               = true
  lifecycle_rule {
    condition { age = 30 } # raw files are replayable; expire after 30 days
    action { type = "Delete" }
  }
}

# --- BigQuery dataset + partitioned/clustered table --------------------------
resource "google_bigquery_dataset" "ds" {
  dataset_id = "product_analytics"
  location   = var.region
}

resource "google_bigquery_table" "daily_metrics" {
  dataset_id          = google_bigquery_dataset.ds.dataset_id
  table_id            = "daily_metrics"
  deletion_protection = false
  schema              = file("${path.module}/../cloud/schema.json")

  time_partitioning {
    type  = "DAY"
    field = "event_date"
  }
  clustering = ["channel"]
}

# --- Least-privilege service account for the functions -----------------------
resource "google_service_account" "fn" {
  account_id   = "events-pipeline-fn"
  display_name = "events -> BigQuery pipeline function"
}

# Read the landing bucket only.
resource "google_storage_bucket_iam_member" "fn_read_bucket" {
  bucket = google_storage_bucket.landing.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.fn.email}"
}

# Write rows into the one dataset (not project-wide data access).
resource "google_bigquery_dataset_iam_member" "fn_write_dataset" {
  dataset_id = google_bigquery_dataset.ds.dataset_id
  role       = "roles/bigquery.dataEditor"
  member     = "serviceAccount:${google_service_account.fn.email}"
}

# Run load/query jobs (project scope; the narrowest role that allows jobs).
resource "google_project_iam_member" "fn_jobuser" {
  project = var.project_id
  role    = "roles/bigquery.jobUser"
  member  = "serviceAccount:${google_service_account.fn.email}"
}

# --- Function source, zipped and uploaded ------------------------------------
data "archive_file" "src" {
  type        = "zip"
  source_dir  = "${path.module}/../cloud"
  output_path = "${path.module}/build/source.zip"
  excludes    = ["tests", "_landing", "__pycache__"]
}

resource "google_storage_bucket" "source" {
  name                        = "${var.project_id}-fn-source"
  location                    = var.region
  uniform_bucket_level_access = true
  force_destroy               = true
}

resource "google_storage_bucket_object" "src" {
  name   = "source-${data.archive_file.src.output_md5}.zip"
  bucket = google_storage_bucket.source.name
  source = data.archive_file.src.output_path
}

locals {
  env = {
    GCP_PROJECT    = var.project_id
    BQ_DATASET     = google_bigquery_dataset.ds.dataset_id
    BQ_TABLE       = google_bigquery_table.daily_metrics.table_id
    LANDING_BUCKET = google_storage_bucket.landing.name
  }
}

# --- Event-driven function: fires when a raw file lands ----------------------
resource "google_cloudfunctions2_function" "on_finalize" {
  name     = "events-on-finalize"
  location = var.region
  build_config {
    runtime     = "python312"
    entry_point = "on_gcs_finalize"
    source {
      storage_source {
        bucket = google_storage_bucket.source.name
        object = google_storage_bucket_object.src.name
      }
    }
  }
  service_config {
    service_account_email = google_service_account.fn.email
    environment_variables = local.env
    max_instance_count    = 3
    available_memory      = "512Mi"
    timeout_seconds       = 300
  }
  event_trigger {
    event_type            = "google.cloud.storage.object.v1.finalized"
    service_account_email = google_service_account.fn.email
    event_filters {
      attribute = "bucket"
      value     = google_storage_bucket.landing.name
    }
  }
}

# --- Scheduled function: Cloud Scheduler -> Pub/Sub -> function ---------------
resource "google_pubsub_topic" "daily" {
  name = "events-daily-load"
}

resource "google_cloudfunctions2_function" "scheduled" {
  name     = "events-scheduled-load"
  location = var.region
  build_config {
    runtime     = "python312"
    entry_point = "scheduled_load"
    source {
      storage_source {
        bucket = google_storage_bucket.source.name
        object = google_storage_bucket_object.src.name
      }
    }
  }
  service_config {
    service_account_email = google_service_account.fn.email
    environment_variables = local.env
    max_instance_count    = 1
    available_memory      = "512Mi"
    timeout_seconds       = 300
  }
  event_trigger {
    event_type            = "google.cloud.pubsub.topic.v1.messagePublished"
    pubsub_topic          = google_pubsub_topic.daily.id
    service_account_email = google_service_account.fn.email
  }
}

resource "google_cloud_scheduler_job" "daily" {
  name      = "events-daily-load"
  schedule  = "0 6 * * *" # 06:00 UTC daily: load the previous day
  time_zone = "Etc/UTC"
  pubsub_target {
    topic_name = google_pubsub_topic.daily.id
    data       = base64encode("{}") # empty -> function defaults to yesterday
  }
}

output "landing_bucket" { value = google_storage_bucket.landing.name }
output "bigquery_table" { value = "${var.project_id}.${google_bigquery_dataset.ds.dataset_id}.${google_bigquery_table.daily_metrics.table_id}" }
