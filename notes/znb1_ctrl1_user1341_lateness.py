"""ZNB1 invoices routed to the manual queue (CTRL1) and posted manually: how late, User1341 vs the other clerks.

Scope: posted ZNB1 invoices whose first Initial validation routed them to CTRL1 and whose first posting is
"Posted manually" (all suppliers). Clerk = user of that posting. Days = pipeline clock (time_to_post_days: arrival
in AP to first posting; the few CTRL1 invoices with a WAIT_GR step start at goods receipt).
Buckets: <= 10 days (discount kept), 10-30 days (discount missed, paid on time), > 30 days (past the payment term).
-> notes/figures/znb1_ctrl1_user1341_lateness.png + notes/znb1_ctrl1_user1341_lateness.csv
Run from the repo root:  PYTHONPATH=. conda run -n datasci python -W ignore notes/znb1_ctrl1_user1341_lateness.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import POSTING_ACTIVITIES, SLA_HARD_DAYS, SLA_SOFT_DAYS, load_events, load_invoices

NOTES = Path(__file__).resolve().parent
PNG = NOTES / "figures" / "znb1_ctrl1_user1341_lateness.png"
CSV = NOTES / "znb1_ctrl1_user1341_lateness.csv"
K = ["PO_ID", "SAP_PO_No"]
FOCUS = "User1341"
MIN_POSTINGS = 100  # clerks shown on their own row

BUCKETS = [f"≤ {SLA_SOFT_DAYS} days", f"{SLA_SOFT_DAYS}–{SLA_HARD_DAYS} days", f"> {SLA_HARD_DAYS} days"]
COLORS = ["#2a78d6", "#eb6834", "#a3261b"]
TEXT_ON = ["white", "white", "white"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"

ev = load_events()
z = ev[ev.SAP_PO_Type_Code == "ZNB1"].sort_values(["PO_ID", "ts", "Event_ID"])
route = z[z.Activity_ID == "Initial validation"].groupby(K).SAP_Next_Processing_Type.first().rename("route")
first_post = z[z.Activity_ID.isin(POSTING_ACTIVITIES)].groupby(K).agg(act=("Activity_ID", "first"), user=("User_ID", "first"))
inv = load_invoices()
inv = inv[(inv.po_type == "ZNB1") & inv.posted].set_index(K).join(route).join(first_post)
m = inv[(inv.route == "CTRL1") & (inv.act == "Posted manually")].copy()
m["bucket"] = pd.cut(m.time_to_post_days, [-np.inf, SLA_SOFT_DAYS, SLA_HARD_DAYS, np.inf], labels=BUCKETS)


def row(sub):
    c = sub.bucket.value_counts().reindex(BUCKETS, fill_value=0)
    return pd.Series({"invoices": len(sub), **{b: int(c[b]) for b in BUCKETS},
                      **{f"{b} %": c[b] / len(sub) for b in BUCKETS}, "median_days": sub.time_to_post_days.median()})


others = m[m.user != FOCUS]
big = others.user.value_counts()
big = big[big >= MIN_POSTINGS].index
rows = {FOCUS: row(m[m.user == FOCUS]), "All other clerks": row(others), "All clerks": row(m)}
rows.update({u: row(others[others.user == u]) for u in big})
tab = pd.DataFrame(rows).T
tab.index.name = "clerk"
tab.to_csv(CSV)

# ---------------------------------------------------------------- chart: 100% stacked bars
order = [FOCUS, "All other clerks", None] + list(big)  # None = gap row
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"]})
fig, ax = plt.subplots(figsize=(11.5, 6.6), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
ypos, labels = [], []
y = 0.0
for name in order:
    if name is None:
        y -= 0.5
        continue
    r = tab.loc[name]
    h = 0.72 if name in (FOCUS, "All other clerks") else 0.55
    left = 0.0
    for b, col, tc in zip(BUCKETS, COLORS, TEXT_ON):
        w = r[f"{b} %"] * 100
        ax.barh(y, w, left=left, height=h, color=col, edgecolor=SURFACE, linewidth=1.5, zorder=2)
        if w >= 6:
            ax.text(left + w / 2, y, f"{w:.0f}%", ha="center", va="center", fontsize=9.5 if h > 0.6 else 8.5,
                    color=tc, fontweight="bold")
        left += w
    late = int(r[BUCKETS[1]] + r[BUCKETS[2]])
    ax.text(101.5, y, f"{int(r.invoices):,} invoices · {late:,} late · median {r.median_days:.1f} d",
            va="center", ha="left", fontsize=9 if h > 0.6 else 8.5, color=INK2)
    ypos.append(y)
    labels.append(name)
    y -= 1.0
ax.set_yticks(ypos)
ax.set_yticklabels(labels)
for t in ax.get_yticklabels():
    if t.get_text() in (FOCUS, "All other clerks"):
        t.set_fontweight("bold")
        t.set_fontsize(11)
    else:
        t.set_fontsize(9.5)
    t.set_color(INK)
ax.set_xlim(0, 100)
ax.set_xticks(range(0, 101, 20))
ax.set_xticklabels([f"{v}%" for v in range(0, 101, 20)])
ax.grid(axis="x", color=GRID, zorder=0)
ax.tick_params(axis="both", length=0, colors=INK2)
for s in ax.spines.values():
    s.set_visible(False)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS]
names = [f"≤ {SLA_SOFT_DAYS} days: discount kept", f"{SLA_SOFT_DAYS}–{SLA_HARD_DAYS} days: discount missed",
         f"> {SLA_HARD_DAYS} days: past payment term"]
ax.legend(handles, names, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, fontsize=9.5,
          labelcolor=INK2, handlelength=1.2, borderaxespad=0.3)
f, o = tab.loc[FOCUS], tab.loc["All other clerks"]
fig.suptitle(f"ZNB1 manual queue (CTRL1): {FOCUS} is late more often, but never beyond {SLA_HARD_DAYS} days",
             x=0.01, ha="left", y=0.975, fontsize=13.5, fontweight="bold", color=INK)
fig.text(0.01, 0.905,
         f"{FOCUS}: {int(f.invoices):,} invoices, {f[f'{BUCKETS[1]} %'] + f[f'{BUCKETS[2]} %']:.0%} posted after "
         f"{SLA_SOFT_DAYS} days, {int(f[BUCKETS[2]])} after {SLA_HARD_DAYS} · other clerks: {int(o.invoices):,} invoices, "
         f"{o[f'{BUCKETS[1]} %'] + o[f'{BUCKETS[2]} %']:.0%} after {SLA_SOFT_DAYS} days, {int(o[BUCKETS[2]])} after {SLA_HARD_DAYS}",
         ha="left", fontsize=10, color=INK2)
fig.text(0.01, 0.012, f"Posted ZNB1 invoices routed to CTRL1 at their first initial validation and posted manually "
         f"(all suppliers). Clerk = user of the posting. Days = pipeline clock (arrival in AP to first posting). "
         f"Clerks with ≥ {MIN_POSTINGS} invoices shown on their own row.", ha="left", fontsize=7.5, color=INK2)
fig.subplots_adjust(left=0.15, right=0.79, top=0.8, bottom=0.07)
fig.savefig(PNG, facecolor=SURFACE)
pd.set_option("display.width", 200)
print(tab.round(3).to_string())
print(PNG)
