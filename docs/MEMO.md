# Where we lose new users in week one — and what it is worth to fix

**To:** Growth / Product
**From:** Analytics
**Date:** 2026-08-18
**Data:** 380,798 events, 59,998 users, 120 days. Reproducible: `make all`.

---

## Recommendation

**Fix the add-to-cart step, and stop treating paid search as if it produced the
same users as organic.**

Add-to-cart is where we lose the most people by a wide margin: **18,664 users
(50.1% of everyone who reaches it)** drop there. Recovering even a tenth of them
is worth roughly **$50K** on this dataset — and it is the only step where a
double-digit relative improvement is plausible, because the other three steps
already convert at 73–85%.

The second finding is cheaper to act on: **paid search users purchase at 5.9% vs
organic's 29.8%** — an 80% relative gap that is not noise (95% CI on the
difference: **−24.7pp to −23.2pp**). We are paying to acquire our worst-retaining
cohort, and week-one loss is concentrated there.

---

## The funnel

| step | users | step conversion | users lost | cumulative |
|---|---|---|---|---|
| signup | 59,771 | — | — | 100.0% |
| activate | 43,731 | 73.2% | 16,040 | 73.2% |
| first_search | 37,232 | 85.1% | 6,499 | 62.3% |
| **add_to_cart** | **18,568** | **49.9%** | **18,664** | **31.1%** |
| purchase | 11,877 | 64.0% | 6,691 | 19.9% |

Two steps lose ~16–19K users each, but they are not equally fixable. Activation
loses 16,040 at a 73.2% pass rate — already decent, so a fix there is fighting for
the last quartile. Add-to-cart passes only **49.9%**, which is the outlier in this
funnel and the step where the headroom is.

## Week-one retention, by channel

Right-censored cohorts are excluded per horizon: a user who signed up three days
before the data ends cannot have a D7 event, and leaving them in the denominator
manufactures a decline that is purely an artefact of the observation window. (An
earlier version of this analysis did exactly that and reported 0.7% D30
retention. It was a measurement bug, not a finding.)

| channel | users | D1 | D7+ | D30+ |
|---|---|---|---|---|
| referral | 7,036 | 52.8% | 85.1% | 25.1% |
| organic | 20,245 | 49.5% | 82.5% | 25.3% |
| email | 6,076 | 45.0% | 80.4% | 24.4% |
| social | 10,941 | 33.7% | 73.3% | 22.2% |
| **paid_search** | **15,700** | **30.9%** | **70.4%** | **21.6%** |

**22.3% of all users are gone within seven days.** The spread across channels is
the actionable part: paid search starts 19 percentage points below referral at D1
and never closes the gap. Paid search is our second-largest channel by volume
(15,700 users, 26%) and our worst on every retention horizon.

## The statistical comparison

Purchase rate, paid search vs organic, two-proportion z-test (pooled SE for the
p-value, unpooled SE for the interval — using one for both is a real
inconsistency near the boundary):

| | organic | paid_search |
|---|---|---|
| users | 20,244 | 15,695 |
| purchase rate | 29.8% | 5.9% |

* Absolute difference: **−23.9pp**, 95% CI **[−24.7pp, −23.2pp]**
* p < 0.001

The interval is narrow because the samples are large. This is a real difference,
not a bar chart that looks suggestive.

## What the fix is worth

Sized three ways rather than one, because a single point estimate reads as more
precise than it is:

| scenario | recovery assumed | users recovered | extra purchasers | incremental revenue |
|---|---|---|---|---|
| conservative | 5% | 933 | 597 | $24,822 |
| **central** | **10%** | **1,866** | **1,194** | **$49,645** |
| optimistic | 20% | 3,733 | 2,388 | $99,289 |

### Defending the number

Every one of these is an **upper bound**, and here is exactly why:

1. **Recovered users are assumed to convert downstream at 64.0%** — the rate of
   users who *already passed* add-to-cart. Users who dropped are systematically
   less engaged, so their true downstream rate is lower. This is the single
   largest source of optimism in the estimate.
2. **ARPU per purchaser ($41.55) is measured over the full window**, so late
   cohorts contribute less revenue than they eventually will. This pushes the
   estimate *down*, partly offsetting (1).
3. **No cost of the fix is included.** This is gross opportunity, not ROI. A
   change requiring a quarter of eng time may not clear the bar even at the
   optimistic figure.
4. **Recovery rates of 5/10/20% are assumptions, not measurements.** Nothing in
   this data tells us how many droppers a better cart flow would save. That
   number comes from an experiment, which is the actual next step.

## Data quality caveats

Bounding what the conclusions can carry:

| issue | scale | handling |
|---|---|---|
| duplicate events | 4,851 rows (1.48%) | deduplicated on `(user_id, event_name, event_ts)`, **not** on `event_id` — retried beacons carry fresh ids, so a surrogate-key dedupe finds nothing |
| ios timezone bug | all ios events, −8h | corrected in staging; caught by a circular-mean hour-of-day test, since a uniform shift preserves event *order* and passes every ordering assertion |
| null `user_id` | 1,293 rows (0.40%) | excluded from staging, counted in the audit model |
| right-censoring | recent cohorts | excluded per retention horizon |

**What could still make this analysis wrong:**

* **The duplicates were biased toward late funnel steps** (retry storms happen on
  slow requests). Had they not been deduplicated, add-to-cart and purchase would
  be inflated *more* than the earlier steps, making the cart problem look smaller
  than it is. The direction of that bias matters and it runs against the finding.
* **Consent-blocked clients (0.40%) are not missing at random.** They skew toward
  privacy-conscious users, likely on ios. Any platform-level conclusion inherits
  that bias; this memo deliberately makes none.
* **Channel is attributed at first touch.** Users re-attributed later are
  unaffected here, but a last-touch model would move users between these buckets
  and could change the paid-search gap.
* **Seasonality is not controlled.** 120 days covers roughly one quarter with no
  holiday period, so no seasonal correction was applied — and none is claimed.

## Next step

The recommendation above is a hypothesis with a price tag, not a proven win. The
correct next move is an **A/B test on the add-to-cart step**, powered against the
49.9% baseline. At ~18,500 users per arm, that test can detect roughly a **+3%
relative** change at 80% power — comfortably inside the range worth shipping for.

---

*Data is simulated. Dollar magnitudes are meaningful only relative to other
figures in this dataset and are not real-world claims. The generator's true step
rates are known, and the pipeline recovers them: true activation 0.72 against a
measured 0.732, which is exactly the 1.017 channel-mix multiplier — the recovery
of ground truth is what validates the pipeline.*
