"""User1341's working pattern from all their events (every activity, every PO type): which days, which hours, how long.

- Period: first to last User1341 event. Working days = Mon-Fri in that period.
- Active day = a day with at least one User1341 event.
- Daily span = first to last event of the day.
- Estimated active hours (method of notes/user1341_daily.py): the log has completion timestamps only, so the gap between
  two consecutive User1341 events on the same day is taken as working time; gaps > BREAK_MIN minutes are breaks / other
  work and start a new session; each session's first event gets the median in-session gap.
- Absence = run of >= 2 working days in the period without any event.
Outputs: notes/figures/user1341_work_pattern.png, notes/user1341_work_pattern_daily.csv
Run from the repo root:  PYTHONPATH=. conda run -n datasci python -W ignore notes/user1341_work_pattern.py
"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from common import load_events
from common import use_invoice_proxy_labels

use_invoice_proxy_labels()  # chart text: invoice -> invoice proxy

NOTES = Path(__file__).resolve().parent
USER = "User1341"
BREAK_MIN = 30

ev = load_events()
u = ev[ev.User_ID == USER].sort_values(["ts", "Event_ID"]).copy()
u["day"] = u.ts.dt.normalize()
first, last = u.day.min(), u.day.max()
workdays = pd.bdate_range(first, last)

# ---------------------------------------------------------------- per day
rows = []
for d, g in u.groupby("day"):
    t = g.ts.sort_values()
    gaps = t.diff().dt.total_seconds().div(60).iloc[1:]
    inside = gaps[gaps <= BREAK_MIN]
    sessions = int((gaps > BREAK_MIN).sum()) + 1
    med = inside.median() if len(inside) else 2.0
    rows.append(dict(day=d, weekday=d.day_name()[:3], events=len(g),
                     postings=int((g.Activity_ID == "Posted manually").sum()),
                     first=t.iloc[0], last=t.iloc[-1], span_h=(t.iloc[-1] - t.iloc[0]).total_seconds() / 3600,
                     sessions=sessions, active_h=(inside.sum() + sessions * med) / 60,
                     hours_with_events=g.ts.dt.hour.nunique()))
D = pd.DataFrame(rows).set_index("day")
D["first_h"] = D["first"].dt.hour + D["first"].dt.minute / 60
D["last_h"] = D["last"].dt.hour + D["last"].dt.minute / 60
D.drop(columns=["first", "last"]).round(2).to_csv(NOTES / "user1341_work_pattern_daily.csv")

act_wd = D.index[D.index.dayofweek < 5]
weekend_days = int((D.index.dayofweek >= 5).sum())
share_active = len(act_wd) / len(workdays)

# Absences: runs of >= 2 inactive working days
inactive = pd.Series(~workdays.isin(D.index), index=workdays)
runs, start = [], None
for d, off in inactive.items():
    if off and start is None:
        start = d
    if not off and start is not None:
        runs.append((start, prev))
        start = None
    prev = d
if start is not None:
    runs.append((start, prev))
absences = [(a, b, len(pd.bdate_range(a, b))) for a, b in runs if len(pd.bdate_range(a, b)) >= 2]
single_off = sum(1 for a, b in runs if len(pd.bdate_range(a, b)) == 1)
days_in_absence = sum(n for *_, n in absences)

wd_rate = pd.Series({k: (D.index[D.index.dayofweek == i].size / (workdays.dayofweek == i).sum())
                     for i, k in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri"])})
hour_share = u.ts.dt.hour.value_counts(normalize=True).reindex(range(24), fill_value=0)
core = hour_share[hour_share > 0.02].index
med = D.loc[act_wd].median(numeric_only=True)
q = lambda c, p: D.loc[act_wd, c].quantile(p)
short_days = (D.loc[act_wd, "active_h"] < 2).mean()

# ---------------------------------------------------------------- figure (16:9)
INK, MUTED, GRID, ACC = "#16201c", "#5a6a63", "#d9e1dd", "#2a78d6"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
                     "xtick.color": MUTED, "ytick.color": MUTED})
fig = plt.figure(figsize=(13.333, 7.5), facecolor="white")
fig.text(0.03, 0.94, f"{USER}: when and how long they work", fontsize=20, fontweight="bold", color=INK)
fig.text(0.03, 0.915, va="top", s=(
    f"Active on {len(act_wd)} of {len(workdays)} working days ({share_active:.0%}) between {first:%d %b %Y} and {last:%d %b %Y}, "
    f"with {len(absences)} absences of 2+ working days ({days_in_absence} days) and {single_off} single days off.\n"
    f"On an active day: first event {int(med.first_h):02d}:{int(med.first_h % 1 * 60):02d}, last {int(med.last_h):02d}:{int(med.last_h % 1 * 60):02d} "
    f"(median), about {med.active_h:.1f} h of logged work spread over a {med.span_h:.1f} h span, {med.events:.0f} events."),
    fontsize=11.5, color=MUTED, linespacing=1.45)

# A: calendar (weeks x weekdays), colour = events per day
axA = fig.add_axes([0.03, 0.12, 0.42, 0.62])
weeks = pd.date_range(first - pd.Timedelta(days=first.dayofweek), last, freq="W-MON")
grid = np.full((5, len(weeks)), np.nan)
for j, w in enumerate(weeks):
    for i in range(5):
        d = w + pd.Timedelta(days=i)
        if first <= d <= last:
            grid[i, j] = D.events.get(d, 0)
cmap = LinearSegmentedColormap.from_list("blue", ["#eef4fb", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])
cmap.set_bad("white")
vmax = np.nanpercentile(grid[grid > 0], 95)
for i in range(5):
    for j in range(len(weeks)):
        v = grid[i, j]
        if np.isnan(v):
            continue
        col = "#f1f3f2" if v == 0 else cmap(min(v / vmax, 1))
        axA.add_patch(plt.Rectangle((j + 0.06, 4 - i + 0.06), 0.88, 0.88, color=col, lw=0))
        if v == 0:
            axA.plot([j + 0.3, j + 0.7], [4 - i + 0.5] * 2, color="#b9c3be", lw=1)
axA.set_xlim(0, len(weeks))
axA.set_ylim(0, 5)
axA.set_yticks([4.5, 3.5, 2.5, 1.5, 0.5])
axA.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri"], color=INK)
month_starts = [(j, w) for j, w in enumerate(weeks) if j == 0 or w.month != weeks[j - 1].month]
axA.set_xticks([j + 0.5 for j, _ in month_starts])
axA.set_xticklabels([w.strftime("%b") for _, w in month_starts], color=INK)
axA.tick_params(length=0)
for s in axA.spines.values():
    s.set_visible(False)
axA.set_title("Every working day: events logged (grey dash = no activity)", loc="left", fontsize=12, fontweight="bold",
              color=INK, pad=8)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, vmax))
cb = fig.colorbar(sm, ax=axA, orientation="horizontal", fraction=0.04, pad=0.09, aspect=40)
cb.set_label("events per day", color=MUTED, fontsize=9)
cb.outline.set_visible(False)

# B: share of active working days per weekday
axB = fig.add_axes([0.53, 0.52, 0.18, 0.22])
axB.bar(wd_rate.index, wd_rate.values * 100, color=ACC, width=0.6)
for i, v in enumerate(wd_rate.values):
    axB.text(i, v * 100 + 2, f"{v:.0%}", ha="center", fontsize=9.5, color=INK)
axB.set_ylim(0, 110)
axB.set_yticks([])
axB.set_title("Active, by weekday", loc="left", fontsize=12, fontweight="bold", color=INK, pad=8)

# C: hour of day
axC = fig.add_axes([0.76, 0.52, 0.21, 0.22])
hrs = range(6, 21)
axC.bar(list(hrs), [hour_share[h] * 100 for h in hrs], color=ACC, width=0.75)
axC.set_xticks([6, 9, 12, 15, 18])
axC.set_xticklabels(["06", "09", "12", "15", "18"])
axC.set_yticks([])
peak = hour_share.idxmax()
axC.text(peak, hour_share[peak] * 100 + 0.6, f"{hour_share[peak]:.0%}", ha="center", fontsize=9.5, color=INK)
axC.set_ylim(0, hour_share.max() * 100 * 1.25)
axC.set_title("Events by hour of day", loc="left", fontsize=12, fontweight="bold", color=INK, pad=8)

# D: hours per active day (distribution of estimated active hours and span)
axD = fig.add_axes([0.53, 0.12, 0.44, 0.27])
bins = np.arange(0, 11, 0.5)
axD.hist(D.loc[act_wd, "active_h"], bins=bins, color=ACC, alpha=0.9, label="estimated active hours", rwidth=0.85)
axD.hist(D.loc[act_wd, "span_h"], bins=bins, histtype="step", color=INK, lw=1.6, label="span first → last event")
axD.axvline(med.active_h, color=ACC, ls=(0, (3, 2)), lw=1)
axD.set_ylim(0, axD.get_ylim()[1] * 1.15)
axD.annotate(f"median {med.active_h:.1f} h active", xy=(med.active_h, axD.get_ylim()[1] * 0.97), xytext=(3.0, axD.get_ylim()[1] * 0.9),
             color=INK, fontsize=9.5, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
axD.set_xlabel("hours per active working day", color=MUTED)
axD.set_yticks([])
axD.legend(frameon=False, fontsize=9, loc="upper right")
axD.set_title("Hours per active day", loc="left", fontsize=12, fontweight="bold", color=INK, pad=8)
for ax in (axB, axC, axD):
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="x", length=0)

fig.text(0.03, 0.012,
         f"All {len(u):,} {USER} events in the log (every activity, every PO type). Active hours are estimated from the gaps between consecutive "
         f"events on the same day; gaps over {BREAK_MIN} min count as breaks or other work.\nThe log records completed steps only, so time "
         f"spent without logging (e.g. reading, phone, other systems) is not visible: treat hours as a lower bound.\n"
         f"Weekend events: {weekend_days} days. Source: notes/user1341_work_pattern_daily.csv",
         fontsize=8.5, color=MUTED, linespacing=1.4)
png = NOTES / "figures" / "user1341_work_pattern.png"
fig.savefig(png, dpi=200, facecolor="white")
plt.close(fig)

print("wrote", png)
print(f"events {len(u)}, activities {u.Activity_ID.value_counts().head(6).to_dict()}")
print(f"period {first:%Y-%m-%d} - {last:%Y-%m-%d}, workdays {len(workdays)}, active workdays {len(act_wd)} ({share_active:.1%}), weekend days {weekend_days}")
print("absences:", [(a.strftime('%d %b'), b.strftime('%d %b'), n) for a, b, n in absences], "single days off:", single_off)
print("weekday active rate:", wd_rate.round(2).to_dict())
print("hour share:", hour_share[hour_share > 0.01].round(3).to_dict())
for c in ["events", "postings", "first_h", "last_h", "span_h", "active_h", "sessions", "hours_with_events"]:
    print(f"{c:18} median {D.loc[act_wd, c].median():6.2f}  p25 {q(c, .25):6.2f}  p75 {q(c, .75):6.2f}  max {D.loc[act_wd, c].max():6.2f}")
print(f"share of active days with < 2 h active: {short_days:.0%}; >= 6 h: {(D.loc[act_wd, 'active_h'] >= 6).mean():.0%}")
print("by month: active days / workdays:", {m.strftime('%b %y'): f"{(act_wd.to_period('M') == m).sum()}/{(workdays.to_period('M') == m).sum()}" for m in workdays.to_period('M').unique()})
