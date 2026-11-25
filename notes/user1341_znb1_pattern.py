"""User1341 and ZNB1: the one-week conveyor belt, plus their workload for the suppliers that go 100% to CTRL1.

Age at posting = calendar days from the invoice's first CTRL1 routing to User1341's "Posted manually".
100%-manual suppliers = ZNB1 suppliers whose invoices are all routed to CTRL1 at first validation (as in
znb1_ctrl1_100pct_suppliers.py); their invoices posted straight from the queue.
Active working day = Mon-Fri with at least one User1341 event (they have none at weekends); estimated active hours from
notes/user1341_work_pattern_daily.csv (run notes/user1341_work_pattern.py first).
Outputs: notes/figures/user1341_znb1_pattern.png, notes/user1341_znb1_pattern.csv (per active day)
Run from the repo root:  PYTHONPATH=. conda run -n datasci python -W ignore notes/user1341_znb1_pattern.py
"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import POSTING_ACTIVITIES, SLA_SOFT_DAYS, load_events

NOTES = Path(__file__).resolve().parent
USER = "User1341"
K = ["PO_ID", "SAP_PO_No"]

ev = load_events().sort_values(["PO_ID", "ts", "Event_ID"])
z = ev[ev.SAP_PO_Type_Code == "ZNB1"].copy()
qin = z[z.SAP_Next_Processing_Type == "CTRL1"].groupby(K).ts.min()
pm = z[z.Activity_ID == "Posted manually"].groupby(K).first()
u = pm[pm.User_ID == USER].copy()
u["qin"] = qin.reindex(u.index)
u = u.dropna(subset=["qin"])
u["age"] = (u.ts.dt.normalize() - u.qin.dt.normalize()).dt.days
u["day"] = u.ts.dt.normalize()

# 100%-manual suppliers, invoices posted straight from the queue
z["nxt"] = z.groupby(K).Activity_ID.shift(-1)
fv = z[z.Activity_ID == "Initial validation"].groupby(K).agg(route=("SAP_Next_Processing_Type", "first"),
                                                             nxt=("nxt", "first"), sup=("Supplier_ID", "first"))
sh = fv.groupby("sup").route.agg(lambda x: (x == "CTRL1").mean())
ids = fv.index[fv.sup.isin(sh.index[sh == 1]) & fv.nxt.isin(POSTING_ACTIVITIES)]
first_post = z[z.Activity_ID.isin(POSTING_ACTIVITIES)].groupby(K).first().loc[ids]
mine = first_post[first_post.User_ID == USER]
n_sup = fv.loc[ids, "sup"].nunique()
n_all100 = len(ids)

# Active days and hours (all User1341 events)
wp = pd.read_csv(NOTES / "user1341_work_pattern_daily.csv", parse_dates=["day"]).set_index("day")
days = wp.index[wp.index.dayofweek < 5]
workdays = pd.bdate_range(days.min(), days.max())
per_day = pd.DataFrame({"postings_100pct": mine.groupby(mine.ts.dt.normalize()).size().reindex(days, fill_value=0),
                        "all_postings": wp.loc[days, "postings"], "active_h": wp.loc[days, "active_h"]})
per_day["per_active_hour"] = per_day.postings_100pct / per_day.active_h
per_day.round(2).to_csv(NOTES / "user1341_znb1_pattern.csv")
tot = int(per_day.postings_100pct.sum())
pdm, pdmed = per_day.postings_100pct.mean(), per_day.postings_100pct.median()
q1, q3, mx = per_day.postings_100pct.quantile(0.25), per_day.postings_100pct.quantile(0.75), per_day.postings_100pct.max()
per_h = tot / per_day.active_h.sum()
per_cal = tot / len(workdays)

# Weekly median age and absences (>= 2 inactive working days)
u["week"] = u.day.dt.to_period("W").dt.start_time
wk = u.groupby("week").agg(n=("age", "size"), med=("age", "median"))
wk = wk[wk.n >= 10]
inactive = pd.Series(~workdays.isin(days), index=workdays)
absences, start = [], None
for d, off in inactive.items():
    if off and start is None:
        start = d
    if not off and start is not None:
        if len(pd.bdate_range(start, prev)) >= 2:
            absences.append((start, prev))
        start = None
    prev = d

# ---------------------------------------------------------------- figure
INK, MUTED, GRID, ACC, LATE, SHADE = "#16201c", "#5a6a63", "#d9e1dd", "#2a78d6", "#eb6834", "#f1f3f2"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
                     "xtick.color": MUTED, "ytick.color": MUTED})
fig = plt.figure(figsize=(13.333, 7.5), facecolor="white")
late = ((u.ts - u.qin).dt.total_seconds() / 86400 > SLA_SOFT_DAYS).mean()  # exact time, as in the other analyses
fig.text(0.03, 0.94, f"{USER} and ZNB1: a one-week conveyor belt", fontsize=20, fontweight="bold", color=INK)
fig.text(0.03, 0.915, va="top", s=(
    f"Each posting day {USER} takes the invoices that entered the manual queue about a week earlier, oldest first, regardless of value.\n"
    f"Median age at posting {u.age.median():.0f} days; {(u.age == 7).mean():.0%} exactly 7 days; {late:.0%} past the 10-day discount limit, "
    f"mostly after days off, when nobody else picks the invoices up."), fontsize=11.5, color=MUTED, linespacing=1.45)

# A: age at posting
axA = fig.add_axes([0.05, 0.14, 0.36, 0.58])
ages = u.age.value_counts().reindex(range(0, 17), fill_value=0)
cols = [LATE if a > SLA_SOFT_DAYS else ACC for a in ages.index]
axA.bar(ages.index, ages.values, color=cols, width=0.8)
axA.axvline(SLA_SOFT_DAYS + 0.5, color=MUTED, ls=(0, (4, 3)), lw=1)
axA.text(SLA_SOFT_DAYS + 0.7, ages.max() * 0.95, "10-day\nlimit", color=MUTED, fontsize=9.5, va="top")
axA.text(7, ages[7] + ages.max() * 0.02, f"{ages[7]:,}\n(exactly 7 days)", ha="center", fontsize=9.5, color=INK)
axA.text(13, ages[13] + ages.max() * 0.02, f"{ages[13]:,}", ha="center", fontsize=9.5, color=INK)
axA.set_ylim(0, ages.max() * 1.2)
axA.set_xticks(range(0, 17, 2))
axA.set_xlabel("calendar days from entering CTRL1 to posting", color=MUTED)
axA.set_title(f"Age at posting ({len(u):,} ZNB1 invoices)", loc="left", fontsize=12, fontweight="bold", color=INK, pad=8)

# B: weekly median age, absences shaded
axB = fig.add_axes([0.47, 0.42, 0.5, 0.3])
for a, b in absences:
    axB.axvspan(a, b + pd.Timedelta(days=1), color=SHADE, lw=0)
axB.plot(wk.index, wk.med, color=ACC, lw=2, marker="o", ms=4.5)
axB.axhline(SLA_SOFT_DAYS, color=MUTED, ls=(0, (4, 3)), lw=1)
axB.text(wk.index[0], SLA_SOFT_DAYS + 0.3, "10-day limit", color=MUTED, fontsize=9)
axB.set_ylim(0, 16)
axB.set_ylabel("median days", color=MUTED)
import matplotlib.dates as mdates
axB.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
axB.set_title("Median age by week of posting (grey = absent 2+ working days)", loc="left", fontsize=12, fontweight="bold",
              color=INK, pad=8)

# C: workload box
bx, by, bw, bh = 0.47, 0.12, 0.5, 0.22
fig.add_artist(plt.Rectangle((bx, by), bw, bh, transform=fig.transFigure, facecolor="#e3efea", edgecolor="none"))
fig.text(bx + 0.015, by + bh - 0.035, f"Workload for the {n_sup} suppliers that always go to CTRL1", fontsize=12.5,
         fontweight="bold", color="#1f6f5c")
kp = [(f"{tot:,}", f"postings by {USER}\n({tot / n_all100:.0%} of {n_all100:,})"),
      (f"{len(days)}", f"active working days\nof {len(workdays)} ({len(days) / len(workdays):.0%})"),
      (f"{pdm:.0f}", f"per active day (mean)\nmedian {pdmed:.0f}, max {mx:.0f}\nmiddle half {q1:.0f}–{q3:.0f}"),
      (f"≈ {per_h:.0f}", f"per logged active hour\n{per_cal:.0f} per working day\nincl. days off")]
for i, (big, small) in enumerate(kp):
    x = bx + 0.015 + i * bw / 4
    fig.text(x, by + 0.095, big, fontsize=21, fontweight="bold", color=INK)
    fig.text(x, by + 0.085, small, fontsize=9, color=MUTED, linespacing=1.3, va="top")

for ax in (axA, axB):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
fig.text(0.03, 0.012,
         f"ZNB1 invoices posted manually by {USER} ({u.day.min():%d %b %Y} – {u.day.max():%d %b %Y}); age in calendar days, entry = first "
         f"routing to CTRL1.\nWorkload: invoices of the {n_sup} suppliers with 100% CTRL1 routing, posted straight from the queue. "
         f"Active working day = Mon–Fri with at least one {USER} event.\nActive hours estimated from gaps between events (lower bound). "
         f"Source: notes/user1341_znb1_pattern.csv",
         fontsize=8.5, color=MUTED, linespacing=1.4)
png = NOTES / "figures" / "user1341_znb1_pattern.png"
fig.savefig(png, dpi=200, facecolor="white")
plt.close(fig)
print("wrote", png)
print(f"ZNB1 postings {len(u)}, median age {u.age.median()}, late {late:.1%}")
print(f"100% suppliers: {n_sup}, invoices {n_all100}, by {USER} {tot} ({tot / n_all100:.1%})")
print(f"active days {len(days)} of {len(workdays)}; per active day mean {pdm:.1f} median {pdmed} q1 {q1} q3 {q3} max {mx}; "
      f"per active hour {per_h:.1f}; per calendar working day {per_cal:.1f}; active hours total {per_day.active_h.sum():.0f}")
print("absences:", [(a.strftime('%d %b'), b.strftime('%d %b')) for a, b in absences])
