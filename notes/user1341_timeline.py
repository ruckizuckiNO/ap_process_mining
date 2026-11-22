"""User1341 over time: when they were absent, how their queue built up, and who took over after they left.

Scope: ZNB1 invoices of the 13 suppliers whose main clerk is User1341 (notes/znb1_supplier_main_clerk.csv).
Queue = from the invoice's first CTRL1 routing to its first posting (never-posted invoices stay open to log end).
Absence = run of >= 2 working days (Mon-Fri) without any User1341 event between their first and last activity.
-> notes/user1341_timeline.csv (weekly), notes/user1341_absences.csv, notes/figures/user1341_timeline.png
Run from the repo root:  PYTHONPATH=. conda run -n datasci python -W ignore notes/user1341_timeline.py
"""
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from common import POSTING_ACTIVITIES, load_events

NOTES = Path(__file__).resolve().parent
K = ["PO_ID", "SAP_PO_No"]
U = "User1341"
BLUE, AQUA, ORANGE = "#2a78d6", "#1baf7a", "#eb6834"
INK, INK2, GRID, SURFACE, SHADE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb", "#efeee9"

ev = load_events()
z = ev[ev.SAP_PO_Type_Code == "ZNB1"]
mc = pd.read_csv(NOTES / "znb1_supplier_main_clerk.csv")
mine = set(mc.loc[mc.main_clerk == U, "Supplier_ID"])

# --- absences ---
u_days = set(ev.loc[ev.User_ID == U, "ts"].dt.normalize())
first, last = min(u_days), max(u_days)
gaps, cur = [], []
for d in pd.bdate_range(first, last):
    if d in u_days:
        if len(cur) >= 2:
            gaps.append((cur[0], cur[-1], len(cur)))
        cur = []
    else:
        cur.append(d)
absences = pd.DataFrame(gaps, columns=["from", "to", "working_days"])
absences.to_csv(NOTES / "user1341_absences.csv", index=False)

# --- queue of User1341's suppliers ---
qin = z[z.SAP_Next_Processing_Type == "CTRL1"].groupby(K).ts.min()
post = z[z.Activity_ID.isin(POSTING_ACTIVITIES)].sort_values("ts").groupby(K).first()
q = pd.DataFrame({"qin": qin, "sup": z.groupby(K).Supplier_ID.first().reindex(qin.index)})
q["out"] = post.ts.reindex(q.index)
q["poster"] = post.User_ID.reindex(q.index)
q = q[q.sup.isin(mine)]
q["age"] = (q.out - q.qin).dt.total_seconds() / 86400

weeks = pd.date_range("2025-07-04", "2026-02-27", freq="W-FRI")
rows = []
for w in weeks:
    s = w + pd.Timedelta(hours=23, minutes=59)
    in_week = (q.out > s - pd.Timedelta(days=7)) & (q.out <= s)
    rows.append({
        "week_ending": w,
        "open_in_queue": int(((q.qin <= s) & ((q.out > s) | q.out.isna())).sum()),
        "entered": int(((q.qin > s - pd.Timedelta(days=7)) & (q.qin <= s)).sum()),
        "posted_by_user1341": int((in_week & (q.poster == U)).sum()),
        "posted_by_others": int((in_week & (q.poster != U)).sum()),
        "median_queue_age_days": q.loc[in_week, "age"].median(),
    })
wk = pd.DataFrame(rows)
wk.to_csv(NOTES / "user1341_timeline.csv", index=False)

# --- chart: two panels, one y-scale each ---
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"]})
fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 7.6), dpi=200, sharex=True, gridspec_kw={"height_ratios": [1, 1]})
fig.patch.set_facecolor(SURFACE)
x = wk.week_ending - pd.Timedelta(days=2)  # centre bars on Mon-Fri
a1.bar(x, wk.posted_by_user1341, width=5, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=2,
       label="Posted by User1341")
a1.bar(x, wk.posted_by_others, width=5, bottom=wk.posted_by_user1341, color=AQUA, edgecolor=SURFACE, linewidth=1.5,
       zorder=2, label="Posted by other clerks")
a2.plot(wk.week_ending, wk.open_in_queue, color=ORANGE, linewidth=2, marker="o", markersize=4,
        markeredgecolor=SURFACE, zorder=3)
for ax in (a1, a2):
    ax.set_facecolor(SURFACE)
    for _, g in absences.iterrows():
        ax.axvspan(g["from"], g["to"] + pd.Timedelta(days=1), color=SHADE, zorder=0)
    ax.axvline(last + pd.Timedelta(days=1), color=INK2, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(axis="both", length=0, colors=INK2, labelsize=9)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color(GRID)

a1.set_ylabel("Invoices posted per week", color=INK2, fontsize=9.5)
a1.set_title("Who posts the invoices of User1341's 13 suppliers", loc="left", fontsize=10.5, color=INK,
             fontweight="bold")
a1.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=INK2)
a2.set_ylabel("Invoices waiting in the queue", color=INK2, fontsize=9.5)
a2.set_title("Queue at the end of each week (same suppliers)", loc="left", fontsize=10.5, color=INK, fontweight="bold")
a2.set_ylim(0, wk.open_in_queue.max() * 1.25)

# annotations
aug = absences[absences["from"].dt.month == 8].iloc[0]
w_aug = wk[wk.week_ending == "2025-08-15"].iloc[0]
w_aug_prev = wk[wk.week_ending == "2025-08-08"].iloc[0]
a2.annotate(f"Absent {aug['from']:%d %b}–{aug['to']:%d %b} ({aug.working_days} working days):\n"
            f"queue {w_aug_prev.open_in_queue} to {w_aug.open_in_queue} in one week,\n"
            f"other clerks post only {w_aug.posted_by_others}",
            xy=(w_aug.week_ending, w_aug.open_in_queue), xytext=(pd.Timestamp("2025-07-04"), 330),
            fontsize=9, color=INK, va="center", arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.8))
peak = wk.loc[wk.open_in_queue.idxmax()]
intake = wk[(wk.week_ending >= "2025-09-19") & (wk.week_ending <= peak.week_ending)].entered.mean()
a2.annotate(f"Peak {peak.open_in_queue} on {peak.week_ending:%d %b}: no absence,\n"
            f"intake jumps to ~{intake:.0f} invoices a week",
            xy=(peak.week_ending, peak.open_in_queue), xytext=(pd.Timestamp("2025-10-20"), 330),
            fontsize=9, color=INK, va="center", arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.8))
pre = q[(q.out >= "2025-10-01") & (q.out <= last + pd.Timedelta(days=1))].age.median()
post_ = q[q.out > last + pd.Timedelta(days=1)]
a2.text(last + pd.Timedelta(days=3), wk.open_in_queue.max() * 1.1,
        f"Last activity {last:%d %b %Y}.\nUser1482 and User1082 take over;\n"
        f"median queue age {pre:.1f} rises to {post_.age.median():.1f} days", fontsize=9, color=INK, va="top")
a2.xaxis.set_major_locator(mdates.MonthLocator())
a2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

fig.suptitle("User1341: the queue builds up when they are away, or when intake surges", x=0.06, ha="left", y=0.975, fontsize=15,
             fontweight="bold", color=INK)
fig.text(0.06, 0.935, "ZNB1 · suppliers whose main clerk is User1341 · grey bands = working days with no User1341 "
         "activity · dashed line = last activity", fontsize=9.5, color=INK2, ha="left", va="top")
fig.text(0.06, 0.01, "Queue = from first CTRL1 routing to first posting. New cases stop on 30 Nov 2025 and the log ends "
         "3 Mar 2026, so volumes fall from December.", fontsize=7.5, color=INK2, ha="left")
fig.subplots_adjust(left=0.06, right=0.98, top=0.86, bottom=0.07, hspace=0.25)
fig.savefig(NOTES / "figures" / "user1341_timeline.png", facecolor=SURFACE)

print(absences.to_string(index=False))
print(wk.to_string(index=False))
print("posted after last day by", post_.poster.value_counts().to_dict(), "median age before/after", pre, post_.age.median())
