"""User1341's postings (all fall on or before 12 Jan 2026): days to posting in bands (<=10, 10-30, >30), the weekly
rhythm, the weekday effect and the monthly split. Clock as in the pipeline (ZNB1 from goods receipt where WAIT_GR).
Run from the repo root:  PYTHONPATH=. conda run -n datasci python -W ignore notes/user1341_time_bands.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import load_cases, load_events, load_invoices
from common import use_invoice_proxy_labels

use_invoice_proxy_labels()  # chart text: invoice -> invoice proxy

NOTES = Path(__file__).resolve().parent
K = ["PO_ID", "SAP_PO_No"]
U, CUTOFF = "User1341", "2026-01-13"

ev = load_events()
inv = load_invoices().set_index(K)
po_value = load_cases().set_index("PO_ID").invoice_value_nok
pm = ev[ev.Activity_ID == "Posted manually"].sort_values("ts").drop_duplicates(K).set_index(K)
u = pm[(pm.User_ID == U) & (pm.ts < CUTOFF)][["ts", "Supplier_ID", "SAP_PO_Type_Code"]].copy()
u["days"] = inv.time_to_post_days.reindex(u.index)
u["po_value_nok"] = po_value.reindex(u.index.get_level_values("PO_ID")).values
u["queue_in"] = ev[ev.SAP_Next_Processing_Type == "CTRL1"].groupby(K).ts.min().reindex(u.index)
u["band"] = pd.cut(u.days, [-1, 10, 30, np.inf], labels=["<=10 d", "10-30 d", ">30 d"])


def summary(x):
    b = x.band.value_counts(normalize=True)
    return pd.Series({"invoices": len(x), "pos": x.index.get_level_values("PO_ID").nunique(),
                      "median_po_value_nok": x.reset_index().drop_duplicates("PO_ID").po_value_nok.median(),
                      "within_10d_pct": 100 * b["<=10 d"], "10_to_30d_pct": 100 * b["10-30 d"],
                      "over_30d_pct": 100 * b[">30 d"], "median_days": x.days.median()})


tables = {
    "all": summary(u).to_frame("User1341").T,
    "po_type": u.groupby("SAP_PO_Type_Code").apply(summary),
    "month": u.groupby(u.ts.dt.to_period("M")).apply(summary),
    "queue_weekday": u.groupby(u.queue_in.dt.dayofweek).apply(summary).rename(index=dict(enumerate(
        ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]))),
    "supplier": u.groupby("Supplier_ID").apply(summary).sort_values("invoices", ascending=False),
}
pd.concat(tables, names=["view", "group"]).round(1).to_csv(NOTES / "user1341_time_bands.csv")

# --- Figure -----------------------------------------------------------------------------
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dc"
CON, CLATE, CVLATE = "#2a78d6", "#eb6834", "#8a8980"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 3, figsize=(19, 5.6), gridspec_kw={"width_ratios": [1.5, 0.8, 1]})

ax = axes[0]
counts, edges = np.histogram(u.days.clip(upper=29.99), bins=np.arange(0, 31, 1))
ax.bar(edges[:-1] + 0.5, counts, width=0.85, color=np.where(edges[:-1] < 10, CON, CLATE))
ax.axvline(10, color=INK, linewidth=1, linestyle="--")
ax.text(10.3, counts.max() * 0.93, "10-day limit", color=INK, fontsize=10)
ax.annotate("weekly slot\n(7–9 days)", (8, counts[7]), xytext=(1.0, counts.max() * 0.78), fontsize=10, color=INK,
            arrowprops=dict(arrowstyle="-", color=MUTED))
ax.annotate("missed a week\n(12–14 days)", (13, counts[13]), xytext=(17, counts.max() * 0.5), fontsize=10,
            color=INK, arrowprops=dict(arrowstyle="-", color=MUTED))
s = tables["all"].iloc[0]
ax.text(29.5, counts.max() * 0.93, f"≤10 d: {s.within_10d_pct:.1f}%\n10–30 d: {s['10_to_30d_pct']:.1f}%\n"
        f">30 d: {s.over_30d_pct:.1f}%", ha="right", va="top", fontsize=11, color=INK)
ax.set_xlabel("Days to posting")
ax.set_ylabel("Invoices")
ax.set_title(f"{int(s.invoices):,} invoices, median PO {s.median_po_value_nok / 1e3:.0f}k NOK: a one-week rhythm",
             loc="left", color=INK, fontweight="bold")
ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)

ax = axes[1]
w = tables["queue_weekday"].reindex(["Mon", "Tue", "Wed", "Thu", "Fri"])
late = 100 - w.within_10d_pct
ax.bar(w.index, late, color=np.where(w.index == "Fri", CLATE, CON), width=0.6)
for x, v in zip(w.index, late):
    ax.text(x, v + 1.2, f"{v:.0f}%", ha="center", fontsize=10, color=INK)
ax.set_ylim(0, 70)
ax.set_xlabel("Weekday the invoice entered the queue")
ax.set_ylabel("Posted after more than 10 days (%)")
ax.set_title("Friday arrivals: most are late", loc="left", color=INK, fontweight="bold")
ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)

ax = axes[2]
m = tables["month"]
labels = [p.strftime("%b") for p in m.index]
ax.bar(labels, m.within_10d_pct, color=CON, width=0.65, label="≤10 days")
ax.bar(labels, m["10_to_30d_pct"], bottom=m.within_10d_pct, color=CLATE, width=0.65, label="10–30 days")
ax.bar(labels, m.over_30d_pct, bottom=m.within_10d_pct + m["10_to_30d_pct"], color=CVLATE, width=0.65,
       label=">30 days")
for x, v in zip(labels, m.within_10d_pct):
    ax.text(x, v / 2, f"{v:.0f}%", ha="center", va="center", fontsize=9.5, color="white")
ax.set_ylim(0, 100)
ax.set_ylabel("Share of invoices (%)")
ax.set_title("August: holidays push most past 10 days", loc="left", color=INK, fontweight="bold")
ax.legend(frameon=False, fontsize=9.5, loc="lower right", ncol=3, bbox_to_anchor=(1, -0.2))
fig.tight_layout()
fig.savefig(NOTES / "figures" / "user1341_time_bands.png", dpi=150)

pd.set_option("display.width", 220)
for k, v in tables.items():
    print(f"== {k}"); print(v.round(1).head(12).to_string())
