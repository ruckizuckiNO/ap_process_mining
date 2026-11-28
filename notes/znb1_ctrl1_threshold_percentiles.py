"""ZNB1 manual queue (CTRL1): distribution of the manual postings above each invoice-value threshold (1.5 / 1.75 / 2.0 /
2.25M NOK), i.e. the postings that would stay manual. One histogram per threshold, cut left at the threshold, with
percentile lines (P10, P25, P50, P75, P90).

Scope (same as znb1_ctrl1_financial.py): posted ZNB1 invoices routed to CTRL1 at their first Initial validation and
posted manually (all suppliers). Value = invoice value: the PO value (Case_Invoice_Value_NOK) split evenly over the
PO's posted invoices (pipeline rule; the log has no amount per invoice), one count per posted invoice. Late = posted > 10 days (pipeline clock); missed discount = 2% of the invoice
amount (PO value split evenly over the PO's posted invoices) when late.
-> notes/figures/znb1_ctrl1_threshold_percentiles.png + notes/znb1_ctrl1_threshold_percentiles.csv
Run from the repo root:  PYTHONPATH=. conda run -n datasci python -W ignore notes/znb1_ctrl1_threshold_percentiles.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import POSTING_ACTIVITIES, SLA_SOFT_DAYS, load_cases, load_events, load_invoices
from common import use_invoice_proxy_labels

use_invoice_proxy_labels()  # chart text: invoice -> invoice proxy

NOTES = Path(__file__).resolve().parent
PNG = NOTES / "figures" / "znb1_ctrl1_threshold_percentiles.png"
CSV = NOTES / "znb1_ctrl1_threshold_percentiles.csv"
K = ["PO_ID", "SAP_PO_No"]
CUTS = [1.5e6, 1.75e6, 2.0e6, 2.25e6]
PCTS = [10, 25, 50, 75, 90]
DISCOUNT = 0.02

BLUE = "#2a78d6"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"

# ---------------------------------------------------------------- data
ev = load_events()
z = ev[ev.SAP_PO_Type_Code == "ZNB1"].sort_values(["PO_ID", "ts", "Event_ID"])
route = z[z.Activity_ID == "Initial validation"].groupby(K).SAP_Next_Processing_Type.first().rename("route")
first_post = z[z.Activity_ID.isin(POSTING_ACTIVITIES)].groupby(K).Activity_ID.first().rename("act")
inv = load_invoices()
inv = inv[inv.po_type == "ZNB1"].set_index(K)
po_value = load_cases().set_index("PO_ID").invoice_value_nok
inv["po_value"] = po_value.reindex(inv.index.get_level_values(0)).values
inv["n_posted"] = inv.groupby(level="PO_ID").posted.transform("sum")
inv["amount"] = np.where(inv.posted, inv.po_value / inv.n_posted.where(inv.n_posted > 0), 0)
inv = inv[inv.posted].join(route).join(first_post)
m = inv[(inv.route == "CTRL1") & (inv.act == "Posted manually")].copy()
m["late"] = m.time_to_post_days > SLA_SOFT_DAYS
m["missed"] = DISCOUNT * m.amount * m.late
n_all, missed_all = len(m), m.missed.sum()

rows = []
for c in CUTS:
    a = m[m.amount > c]
    p = np.percentile(a.amount, PCTS)
    rows.append({"threshold": c, "postings": len(a), "share_postings": len(a) / n_all,
                 "pos": a.index.get_level_values("PO_ID").nunique(),
                 **{f"P{q}": v for q, v in zip(PCTS, p)}, "max": a.amount.max(),
                 "late_share": a.late.mean(), "missed": a.missed.sum(), "share_missed": a.missed.sum() / missed_all})
tab = pd.DataFrame(rows)
tab.to_csv(CSV, index=False)

# ---------------------------------------------------------------- chart: 4 small multiples, shared log x-axis
lo, hi = np.log10(CUTS[0]), np.log10(m.amount.max()) + 0.05
bins = np.logspace(lo, hi, 41)


def fmt(v):
    return f"{v / 1e6:,.1f}M" if v < 1e8 else f"{v / 1e6:,.0f}M"


plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"]})
fig, axes = plt.subplots(len(CUTS), 1, figsize=(12, 10.5), dpi=200, sharex=True, sharey=True)
fig.patch.set_facecolor(SURFACE)
ymax = max(np.histogram(m.amount[m.amount > c], bins=bins)[0].max() for c in CUTS) * 1.32
for ax, r in zip(axes, tab.itertuples()):
    a = m.amount[m.amount > r.threshold]
    ax.set_facecolor(SURFACE)
    ax.hist(a, bins=bins, color=BLUE, edgecolor=SURFACE, linewidth=0.8, zorder=2)
    ax.axvspan(bins[0], r.threshold, color=GRID, alpha=0.6, zorder=1)  # area left of the threshold
    ax.axvline(r.threshold, color=INK, lw=1.4, zorder=3)
    same = {}  # percentiles with the same value share one line and label (e.g. "P10/P25")
    for q in PCTS:
        same.setdefault(round(getattr(r, f"P{q}"), 2), []).append(f"P{q}")
    for v, qs in same.items():
        ax.axvline(v, color=INK2, lw=1, ls=(0, (3, 3)), zorder=3)
        ax.text(v, ymax * 0.97, f"{'/'.join(qs)}\n{fmt(v)}", ha="center", va="top", fontsize=8, color=INK,
                bbox=dict(fc=SURFACE, ec="none", pad=1), zorder=4)
    ax.set_xscale("log")
    ax.set_ylim(0, ymax)
    ax.set_title(f"Threshold {r.threshold / 1e6:g}M NOK: {r.postings:,} postings stay manual ({r.share_postings:.0%} "
                 f"of all) on {r.pos:,} POs · {r.late_share:.0%} late > {SLA_SOFT_DAYS} days · "
                 f"{r.missed / 1e6:,.2f}M NOK missed discount ({r.share_missed:.0%})",
                 loc="left", fontsize=10.5, color=INK, pad=6)
    ax.set_ylabel("Postings", color=INK2, fontsize=9.5)
    ax.grid(axis="y", color=GRID, zorder=0)
    ax.tick_params(axis="both", which="both", length=0, colors=INK2)
    for s in ax.spines.values():
        s.set_visible(False)
ticks = [1.5e6, 2e6, 3e6, 5e6, 1e7, 2e7, 5e7, 1e8]
axes[-1].set_xlim(bins[0], bins[-1])
axes[-1].set_xticks([t for t in ticks if bins[0] <= t <= bins[-1]])
axes[-1].xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1e6:g}M"))
axes[-1].xaxis.set_minor_formatter(plt.NullFormatter())
axes[-1].set_xlabel("Invoice value (NOK, log scale)", color=INK2, fontsize=10)
fig.suptitle("ZNB1 manual queue (CTRL1): what stays manual above each invoice-value threshold", x=0.01, ha="left",
             y=0.985, fontsize=14, fontweight="bold", color=INK)
fig.text(0.01, 0.935, f"{n_all:,} ZNB1 invoices routed to CTRL1 and posted manually ({missed_all / 1e6:,.2f}M NOK missed "
         f"discount).\nEach panel shows only the postings above its threshold (solid line, grey = cut away); dashed lines = "
         f"percentiles of those postings.", ha="left", fontsize=9.5, color=INK2)
fig.text(0.01, 0.008, "All suppliers. One count per posted invoice. Invoice value = PO value (Case_Invoice_Value_NOK) split evenly over the "
         "PO's posted invoices; the log has no amount per invoice.\nLate = posted > 10 days after arrival. "
         "Missed discount = 2% of the invoice value.", ha="left", fontsize=7.5, color=INK2)
fig.subplots_adjust(left=0.07, right=0.98, top=0.895, bottom=0.07, hspace=0.38)
fig.savefig(PNG, facecolor=SURFACE)
pd.set_option("display.width", 220)
print(tab.round(3).to_string())
print(PNG)
