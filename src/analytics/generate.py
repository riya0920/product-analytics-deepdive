"""Event-level generator for the analytics warehouse.

Same design principle as the experimentation platform in this portfolio: the
ground truth is an INPUT, so every conclusion the analysis reaches can be checked
against what was actually generated. Here that means the true week-one drop-off
is known, and the memo's headline number can be verified rather than believed.

Deliberately planted data-quality problems, because an analysis that has only
ever seen clean data is not an analysis:

  * ~1.5% duplicated events (retried client beacons) -- inflates every count if
    not deduplicated, and inflates conversion MORE than traffic because retries
    cluster on the slow steps
  * a timezone bug on one platform (ios events stamped in local time, not UTC)
    which shifts cohort-day boundaries and silently distorts D1 retention
  * a small number of events with a NULL user_id (consent-blocked clients)
  * out-of-order arrival: event timestamps are not monotonic in insertion order

These are found by the dbt-style tests in sql/tests/, not by inspection.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(ROOT, "data")

FUNNEL = ["signup", "activate", "first_search", "add_to_cart", "purchase"]

# TRUE step-through rates. The analysis has to recover these from the events.
TRUE_STEP_RATES = {"activate": 0.72, "first_search": 0.81, "add_to_cart": 0.46, "purchase": 0.58}

CHANNELS = ["organic", "paid_search", "social", "referral", "email"]
CHANNEL_WEIGHTS = [0.34, 0.26, 0.18, 0.12, 0.10]
# Paid traffic converts worse and retains worse -- the segmentation cut that
# makes the memo's recommendation actionable rather than generic.
CHANNEL_QUALITY = {"organic": 1.18, "paid_search": 0.78, "social": 0.85, "referral": 1.25, "email": 1.10}
PLATFORMS = ["ios", "android", "web"]
PLATFORM_WEIGHTS = [0.38, 0.34, 0.28]


def time_of_day_seconds(rng, n: int) -> np.ndarray:
    """Seconds-into-day sampled from a realistic diurnal profile.

    Not uniform, and that matters for more than realism: with events spread
    evenly across 24 hours, a whole-hours timezone shift leaves every hour-of-day
    statistic unchanged, so a timezone bug becomes literally undetectable in the
    distribution. Real traffic has a daily rhythm, and that rhythm is what makes
    the bug visible to `assert_no_platform_hour_skew`.

    Profile: a broad evening peak (~19:00) with a smaller midday shoulder.
    """
    hours = np.arange(24)
    weights = (
        np.exp(-0.5 * ((hours - 19) / 3.0) ** 2)          # evening peak
        + 0.55 * np.exp(-0.5 * ((hours - 12.5) / 2.5) ** 2)  # lunchtime shoulder
        + 0.03                                             # overnight floor
    )
    weights /= weights.sum()
    hour = rng.choice(hours, size=n, p=weights)
    return hour * 3600 + rng.integers(0, 3600, size=n)


@dataclass
class GenConfig:
    n_users: int = 60_000
    days: int = 120
    dup_rate: float = 0.015
    null_user_rate: float = 0.004
    ios_tz_offset_hours: int = -8      # the planted timezone bug
    seed: int = 42


def generate(cfg: GenConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)

    signup_day = rng.integers(0, cfg.days, size=cfg.n_users)
    channel = rng.choice(CHANNELS, size=cfg.n_users, p=CHANNEL_WEIGHTS)
    platform = rng.choice(PLATFORMS, size=cfg.n_users, p=PLATFORM_WEIGHTS)
    quality = np.array([CHANNEL_QUALITY[c] for c in channel])
    user_ids = np.array(["u_%06d" % i for i in range(cfg.n_users)])

    rows = []
    base_ts = pd.Timestamp("2026-01-01T00:00:00Z")

    # Every user emits a signup event.
    signup_ts = base_ts + pd.to_timedelta(signup_day, unit="D") + pd.to_timedelta(time_of_day_seconds(rng, cfg.n_users), unit="s")
    rows.append(pd.DataFrame({"user_id": user_ids, "event_name": "signup", "event_ts": signup_ts,
                              "channel": channel, "platform": platform, "revenue": 0.0}))

    alive = np.ones(cfg.n_users, dtype=bool)
    last_ts = signup_ts.copy()
    for step in FUNNEL[1:]:
        p = np.clip(TRUE_STEP_RATES[step] * quality, 0.01, 0.99)
        advanced = alive & (rng.random(cfg.n_users) < p)
        # Time between steps is lognormal: most users move fast, a long tail
        # takes days. A normal delay would make week-one retention artificially
        # crisp and hide exactly the effect the memo is about.
        delay_s = rng.lognormal(mean=8.5, sigma=1.6, size=cfg.n_users).clip(1, 14 * 86400)
        ts = last_ts + pd.to_timedelta(delay_s, unit="s")
        revenue = np.where(advanced & (step == "purchase"), rng.lognormal(3.4, 0.8, cfg.n_users), 0.0)
        rows.append(pd.DataFrame({"user_id": user_ids[advanced], "event_name": step,
                                  "event_ts": ts[advanced], "channel": channel[advanced],
                                  "platform": platform[advanced], "revenue": revenue[advanced]}))
        last_ts = np.where(advanced, ts, last_ts)
        last_ts = pd.to_datetime(pd.Series(last_ts), utc=True)
        alive = advanced

    # Return sessions after signup: a decaying hazard PLUS a loyal segment.
    #
    # A pure exponential decay drives 30-day retention to essentially zero, which
    # is not what real products look like -- retention curves flatten because a
    # minority of users form a habit. Without that flattening the D30 number is
    # an artefact of the generator rather than a property worth analysing, and
    # the memo would be sizing a fake problem.
    loyal = rng.random(cfg.n_users) < 0.16
    for day_offset in range(1, 91):
        p_return = (0.30 * np.exp(-day_offset / 9.0) + np.where(loyal, 0.055, 0.0)) * quality
        returned = rng.random(cfg.n_users) < p_return
        ts = (signup_ts.normalize() + pd.to_timedelta(day_offset, unit="D")
              + pd.to_timedelta(time_of_day_seconds(rng, cfg.n_users), unit="s"))
        rows.append(pd.DataFrame({"user_id": user_ids[returned], "event_name": "session_start",
                                  "event_ts": ts[returned], "channel": channel[returned],
                                  "platform": platform[returned], "revenue": 0.0}))

    df = pd.concat(rows, ignore_index=True)

    # --- planted defects -------------------------------------------------
    # 1. duplicates, weighted toward the later funnel steps (retry storms happen
    #    on the slow requests, which is what makes naive counts overstate the end
    #    of the funnel more than the start)
    weight = df["event_name"].map({"add_to_cart": 3.0, "purchase": 4.0}).fillna(1.0).to_numpy()
    p_dup = cfg.dup_rate * weight / weight.mean()
    dups = df[rng.random(len(df)) < p_dup].copy()
    df = pd.concat([df, dups], ignore_index=True)

    # 2. timezone bug: ios events stamped in local time instead of UTC
    ios = df["platform"] == "ios"
    df.loc[ios, "event_ts"] = df.loc[ios, "event_ts"] + pd.to_timedelta(cfg.ios_tz_offset_hours, unit="h")

    # 3. consent-blocked clients emit events with no user id
    null_mask = rng.random(len(df)) < cfg.null_user_rate
    df.loc[null_mask, "user_id"] = None

    # 4. out-of-order arrival
    df = df.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)
    df["event_id"] = ["e_%08d" % i for i in range(len(df))]
    return df[["event_id", "user_id", "event_name", "event_ts", "channel", "platform", "revenue"]]


def ground_truth(cfg: GenConfig) -> dict:
    return {
        "true_step_rates": TRUE_STEP_RATES,
        "channel_quality_multipliers": CHANNEL_QUALITY,
        "planted_dup_rate": cfg.dup_rate,
        "planted_null_user_rate": cfg.null_user_rate,
        "planted_ios_tz_offset_hours": cfg.ios_tz_offset_hours,
        "n_users": cfg.n_users,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=60_000)
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--out", default=os.path.join(DATA, "events.parquet"))
    args = ap.parse_args()

    cfg = GenConfig(n_users=args.users, days=args.days)
    df = generate(cfg)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_parquet(args.out, index=False)
    print("wrote %s  rows=%d  users=%d" % (args.out, len(df), df["user_id"].nunique()))
    print("planted defects:", ground_truth(cfg))


if __name__ == "__main__":
    main()
