"""Generate the figures the memo refers to.

    python -m analytics.charts

Every figure is produced from the warehouse, never hand-drawn, so re-running the
pipeline on new data regenerates them and they cannot silently disagree with the
numbers in the memo.

Chart choices worth defending:
* The funnel is a **bar chart with absolute counts**, not the tapering-trapezoid
  "funnel" graphic. The trapezoid encodes the same number twice (width and area)
  and makes small differences look large.
* Retention curves are plotted **per channel on one axis** rather than as small
  multiples, because the whole finding is the gap between channels, and
  separating them into panels is exactly what makes a gap hard to see.
* No dual axes anywhere. Two y-scales on one plot let the author choose the
  story by choosing the scaling.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .pipeline import build

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIGS = os.path.join(ROOT, "results", "figures")

INK = "#22303f"
MUTED = "#8a97a5"
ACCENT = "#c2402f"
BAR = "#4c78a8"


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9)
    ax.yaxis.label.set_color(INK)
    ax.xaxis.label.set_color(INK)
    ax.title.set_color(INK)


def funnel_chart(con, path: str):
    rows = con.execute(
        "SELECT step, users_reached, step_conversion FROM mart_funnel ORDER BY step_index"
    ).fetchall()
    steps = [r[0] for r in rows]
    users = [r[1] for r in rows]
    convs = [r[2] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    # Highlight the worst step -- the one the memo recommends fixing.
    worst = max(range(1, len(rows)), key=lambda i: users[i - 1] - users[i])
    colours = [ACCENT if i == worst else BAR for i in range(len(rows))]
    bars = ax.bar(steps, users, color=colours)

    for i, (b, c) in enumerate(zip(bars, convs)):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.01,
                f"{users[i]:,}", ha="center", va="bottom", fontsize=9, color=INK)
        if c is not None and i > 0:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 0.5,
                    f"{100 * c:.0f}%", ha="center", va="center", fontsize=10,
                    color="white", fontweight="bold")

    ax.set_ylabel("users reaching step")
    ax.set_title("Onboarding funnel — %s loses the most users (%.0f%% pass rate)"
                 % (steps[worst], 100 * convs[worst]), fontsize=11)
    _style(ax)
    ax.set_ylim(0, max(users) * 1.15)
    plt.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def retention_chart(con, path: str):
    rows = con.execute(
        """
        WITH bounds AS (SELECT MAX(CAST(event_ts AS DATE)) AS last_date FROM stg_events),
        act AS (
            SELECT u.channel,
                   DATE_DIFF('day', u.signup_date, CAST(e.event_ts AS DATE)) AS day_number,
                   u.user_id
            FROM stg_users u JOIN stg_events e USING (user_id)
        ),
        sizes AS (SELECT channel, COUNT(*) AS n FROM stg_users GROUP BY channel)
        SELECT a.channel, a.day_number,
               COUNT(DISTINCT a.user_id)::DOUBLE / MAX(s.n) AS rate
        FROM act a JOIN sizes s USING (channel)
        WHERE a.day_number BETWEEN 0 AND 30
        GROUP BY a.channel, a.day_number
        ORDER BY a.channel, a.day_number
        """
    ).fetchall()

    by_channel = {}
    for channel, day, rate in rows:
        by_channel.setdefault(channel, ([], []))
        by_channel[channel][0].append(day)
        by_channel[channel][1].append(rate)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    for channel, (days, rates) in sorted(by_channel.items()):
        highlight = channel == "paid_search"
        ax.plot(days, rates,
                color=ACCENT if highlight else MUTED,
                linewidth=2.4 if highlight else 1.4,
                zorder=3 if highlight else 2,
                label=channel)
    ax.set_xlabel("days since signup")
    ax.set_ylabel("fraction of cohort active")
    ax.set_title("Retention by acquisition channel — paid search trails throughout", fontsize=11)
    ax.legend(frameon=False, fontsize=9, ncol=5, loc="upper right")
    _style(ax)
    plt.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    os.makedirs(FIGS, exist_ok=True)
    con = build()
    funnel_chart(con, os.path.join(FIGS, "funnel.png"))
    retention_chart(con, os.path.join(FIGS, "retention_by_channel.png"))
    print("wrote", FIGS)


if __name__ == "__main__":
    main()
