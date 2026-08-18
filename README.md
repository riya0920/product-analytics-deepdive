# Product Analytics Deep-Dive (SQL-First)

One stated business question, answered end-to-end in SQL over a DuckDB
warehouse, with data-quality tests that are proven to catch planted defects and a
decision memo that leads with the recommendation.

> **Status: ~40% built.** The warehouse, the staging/marts models, the assertion
> tests, the analysis and the memo are done and **runnable**. dbt proper, the
> visual layer and query-performance work are not — see [Roadmap](#roadmap).

## The question

> **Where do we lose new users in week one, and what is it worth to fix?**

Fixed before any query was written. There is no exploratory chart wall here: EDA
without a decision is tourism. The answer is in **[docs/MEMO.md](docs/MEMO.md)**,
which is the artifact meant to be read first.

**Headline:** add-to-cart loses 18,664 users at a 49.9% pass rate — the outlier
in a funnel whose other steps convert at 73–85% — worth ~$50K at a 10% recovery
assumption. And paid search purchases at 5.9% vs organic's 29.8% (95% CI on the
difference: −24.7pp to −23.2pp).

## Run it

```bash
pip install -r requirements.txt
make generate    # build the event dataset
make build       # raw -> staging -> marts in DuckDB
make test        # SQL assertions + pytest
make analysis    # the numbers the memo quotes
make prove       # build WITHOUT cleaning; the tests must fail
```

## The pipeline

```
data/events.parquet          380,798 events, 59,998 users, 120 days
  -> sql/staging/stg_events  dedupe, timezone fix, null-id exclusion
  -> sql/staging/stg_users   one row per user, first-touch attribution
  -> sql/marts/mart_sessions 30-minute-gap sessionisation (LAG + running SUM)
  -> sql/marts/mart_funnel   per-step conversion, users lost, revenue at risk
  -> sql/marts/mart_retention D1/D7/D30 by cohort and channel
  -> sql/marts/audit_data_quality  what cleaning removed, quantified
```

Models are plain `.sql` files with a declared dependency order, run by a ~100-line
runner. Tests are `.sql` files that must return **zero rows** — dbt's contract,
implemented small enough that the SQL stays the artifact rather than the tool
configuration.

## The SQL worth reading

**Sessionisation** (`mart_sessions.sql`) — the canonical live-round exercise:
`LAG` for the previous event, a boolean for gap-exceeded, then a running `SUM`
over that boolean to allocate session ids.

**Funnel** (`mart_funnel.sql`) — `LAG` for the previous step's population,
`FIRST_VALUE` for cumulative conversion, drop-off quantified in users *and*
dollars.

**Retention** (`mart_retention.sql`) — emits both `dN_exact` and `dN_plus`,
because those two definitions disagree and papering over that is how retention
numbers stop being comparable between teams. `COUNT(DISTINCT user_id)` inside the
day filter is what stops a user active three times on day 7 counting three times,
and an assertion test enforces that retention never exceeds cohort size.

## The tests have teeth, and it's proven

Three defects are planted in the generator on purpose. `make prove` builds
staging **without** the fixes and the suite must fail:

```
$ make prove
FAIL assert_no_duplicate_events        4851 offending row(s)
FAIL assert_no_null_user_ids           1 offending row(s)
FAIL assert_no_platform_hour_skew      ('ios','web', 4.92, 12.94, 8.02)
3 test(s) failed on uncleaned data, as they must. The tests have teeth.
```

**Two of these were harder than they look:**

*Duplicates* are deduplicated on `(user_id, event_name, event_ts)` and **not** on
`event_id` — retried client beacons carry a fresh request id, so a surrogate-key
dedupe finds nothing and leaves every count inflated. The dedupe recovers the
planted rate: 1.48% measured against 1.5% planted.

*The timezone bug* passes every ordering assertion, because a uniform per-platform
shift preserves event order — signup is still first, the funnel is still
monotonic — while every cohort-day boundary is silently wrong. It is only visible
in the **distribution**, and only under a **circular** mean: hour-of-day wraps at
midnight, and with an evening-peaked traffic profile the arithmetic means differ
by just 2.4 hours, which hides under any sensible threshold. The circular mean
recovers the true separation exactly: **8.02 hours**.

## Ground truth is recoverable — that's what validates the pipeline

The generator's true step rates are known, so the pipeline can be checked rather
than trusted. `test_funnel_recovers_the_true_step_rates_within_each_channel`
asserts every channel/step transition lands within 0.025 of
`true_rate x channel_multiplier`.

That test is checked **per channel, and the reason is a finding in itself.** The
pooled funnel does *not* equal `true_rate x average_multiplier`: each step
selects on survival, so users still in the funnel at add-to-cart are
disproportionately from high-quality channels and the effective mix drifts upward
step by step. Pooled add-to-cart measures 0.499 against a naive 0.468 prediction
— that 3-point gap is survivorship, not error. Within a single channel there is
no mix to drift, and the generator's rate comes back exactly.

## Two analysis bugs this project found and fixed

1. **Right-censoring.** The first version reported 0.7% D30 retention. A user who
   signed up three days before the data ends cannot have a D7 event, and leaving
   their cohort in the denominator manufactures a decline that is purely an
   artefact of the window. Each horizon now gets its own eligible cohort set.
2. **A degenerate retention curve.** The generator's original pure-exponential
   return hazard drove 30-day retention to nearly zero, so the D30 number was a
   property of the generator rather than of anything worth analysing. Real
   retention curves flatten because a minority of users form a habit; the
   generator now has a loyal segment and D30+ lands at a realistic 22–25%.

Both are in the git history rather than quietly corrected, because the second one
in particular would have made the memo size a fake problem.

## Roadmap (the remaining ~60%)

| Milestone | Status |
|---|---|
| Event generator with planted defects + known ground truth | done |
| Staging models: dedupe, timezone fix, attribution | done |
| Marts: sessions, funnel, retention, quality audit | done |
| SQL assertion tests + inverted proof they catch defects | done |
| One statistical comparison done properly (CI + z-test) | done |
| Opportunity sizing with named assumptions | done |
| Decision memo | done |
| **dbt proper: `dbt test`, `dbt docs`, lineage graph** | not started (hand-rolled runner instead) |
| **Charts: funnel and retention curves as committed figures** | not started |
| **BigQuery variant + partitioning/clustering with before/after timings** | not started |
| **Segmentation beyond channel (platform, geo, cohort-over-cohort)** | not started |
| **Incrementality: is paid search causing the gap or selecting it?** | not started |

That last row is the most important limitation. The memo reports that paid search
retains worse; it does **not** claim paid search *causes* worse retention. That
is a selection-vs-treatment question this data cannot settle.

## Honesty notes

* Data is **simulated**. Dollar magnitudes are meaningful only relative to other
  figures in this dataset and are not real-world claims.
* The memo's dollar figures are **upper bounds** and say so, with the largest
  source of optimism (recovered users assumed to convert like users who already
  passed) named explicitly.
* No causal claim is made anywhere. The recommendation ends by proposing the A/B
  test that would actually establish one, sized against the observed baseline.
