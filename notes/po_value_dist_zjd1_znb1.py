"""PO-value distribution for ZJD1 and ZNB1: two histograms on a shared log-scale x-axis.

Value counted once per PO (Case_Invoice_Value_NOK). POs with value <= 0 can't go on a log axis and are left out
(count shown in the footnote). -> notes/figures/po_value_dist_zjd1_znb1.png + notes/po_value_dist_zjd1_znb1.csv
(summary statistics). Run from the repo root:
    conda run -n datasci python -W ignore notes/po_value_dist_zjd1_znb1.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NOTES = Path(__file__).resolve().parent
sys.path.insert(0, str(NOTES.parent))
from common import PO_TYPES, load_cases  # noqa: E402
from common import use_invoice_proxy_labels

use_invoice_proxy_labels()  # chart text: invoice -> invoice proxy

PNG = NOTES / "figures" / "po_value_dist_zjd1_znb1.png"
CSV = NOTES / "po_value_dist_zjd1_znb1.csv"
TYPES = ["ZJD1", "ZNB1"]

BLUE = "#2a78d6"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"


def fmt_nok(v):
    return f"{v / 1e6:,.2f}M" if v >= 1e6 else f"{v / 1e3:,.0f}k"


cases = load_cases()
cases = cases[cases.po_type.isin(TYPES)]
pos_vals = cases[cases.invoice_value_nok > 0]

stats = cases.groupby("po_type").invoice_value_nok.describe(percentiles=[.1, .25, .5, .75, .9, .99])
stats["total"] = cases.groupby("po_type").invoice_value_nok.sum()
stats["zero_or_neg"] = cases.groupby("po_type").invoice_value_nok.apply(lambda x: int((x <= 0).sum()))
stats.to_csv(CSV)

# A handful of POs below 500 NOK would stretch the axis over three empty decades: left out of the plot, counted
# in the footnote (and still in the stats).
X_MIN = 500
small = pos_vals[pos_vals.invoice_value_nok < X_MIN].groupby("po_type").size().reindex(TYPES, fill_value=0)
pos_vals = pos_vals[pos_vals.invoice_value_nok >= X_MIN]
bins = np.logspace(np.log10(X_MIN), np.log10(2e8), 51)

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"]})
fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), dpi=200, sharex=True, sharey=True)
fig.patch.set_facecolor(SURFACE)

for ax, t in zip(axes, TYPES):
    v = pos_vals.loc[pos_vals.po_type == t, "invoice_value_nok"]
    s = stats.loc[t]
    ax.set_facecolor(SURFACE)
    ax.hist(v, bins=bins, color=BLUE, edgecolor=SURFACE, linewidth=0.6, zorder=2)
    ax.set_xscale("log")
    for x, lab, ls, side in [(s["50%"], f"median {fmt_nok(s['50%'])}", "-", "right"),
                             (s["mean"], f"mean {fmt_nok(s['mean'])}", "--", "left")]:
        ax.axvline(x, color=INK, lw=1.2, ls=ls, zorder=3)
        ax.annotate(lab, (x, 1), xycoords=("data", "axes fraction"), xytext=(-5 if side == "right" else 5, -4),
                    textcoords="offset points", va="top", ha=side, fontsize=9, color=INK,
                    bbox=dict(fc=SURFACE, ec="none", pad=1))
    ax.set_title(f"{t} · {PO_TYPES[t]}   {int(s['count']):,} POs, total {s['total'] / 1e9:,.2f}B NOK, "
                 f"P10–P90 {fmt_nok(s['10%'])}–{fmt_nok(s['90%'])}, max {fmt_nok(s['max'])}",
                 loc="left", fontsize=11, color=INK, pad=8)
    ax.set_ylabel("POs", color=INK2, fontsize=10)
    ax.grid(axis="y", color=GRID, zorder=0)
    ax.tick_params(axis="both", length=0, colors=INK2)
    for sp in ax.spines.values():
        sp.set_visible(False)

axes[-1].set_xlim(bins[0], bins[-1])
ticks = [1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8]
axes[-1].set_xticks([t for t in ticks if bins[0] <= t <= bins[-1]])
axes[-1].xaxis.set_major_formatter(plt.FuncFormatter(
    lambda v, _: f"{v / 1e6:,.0f}M" if v >= 1e6 else (f"{v / 1e3:,.0f}k" if v >= 1e3 else f"{v:,.0f}")))
axes[-1].set_xlabel("PO value (NOK, log scale)", color=INK2, fontsize=10)
fig.suptitle("PO-value distribution, ZJD1 vs ZNB1", x=0.01, ha="left", color=INK, fontsize=13, fontweight="bold")
zeros = ", ".join(f"{t}: {int(stats.loc[t, 'zero_or_neg'])}" for t in TYPES)
fig.text(0.01, 0.01, f"Value counted once per PO (Case_Invoice_Value_NOK). Not plotted: POs with value 0 "
         f"({zeros}) and POs under {X_MIN} NOK ("
         + ", ".join(f"{t}: {small[t]}" for t in TYPES) + "); all are included in the stats in the panel titles.", color=INK2, fontsize=8)
fig.tight_layout(rect=(0, 0.03, 1, 0.97))
fig.savefig(PNG, facecolor=SURFACE)
print(stats.T.to_string())
print(PNG)
