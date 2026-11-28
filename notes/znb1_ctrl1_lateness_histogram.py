"""How late are ZNB1 invoice proxies on the manual queue (CTRL1)? Histogram of days to first posting, > 10 and > 30 days.

Scope: ZNB1 invoice proxies (PO_ID x SAP_PO_No) routed to CTRL1 at their first Initial validation and posted (any posting
activity, all suppliers). Deleted and never-posted ones have no clock stop and are left out.
Clock: the invoice clock (pipeline): goods receipt (first WAIT_GR) if there is one, else arrival in AP; almost no CTRL1
invoice proxy has WAIT_GR, so this is in practice arrival in AP -> first posting.
Amount = PO value (Case_Invoice_Value_NOK) split evenly over the PO's posted invoice proxies (pipeline rule).
  missed discount = 2% x amount                              if days > 10
  late interest   = amount x 12.5% p.a. x (days - 30) / 365  if days > 30  (payment assumed on the posting day)
Left panel: 0-60 days in 1-day bins (longer waits in a final "60+" bar). Right panel: the > 30-day tail in 5-day bins.
-> notes/figures/znb1_ctrl1_lateness_histogram.png + notes/znb1_ctrl1_lateness_histogram.csv
Run from the repo root:  PYTHONPATH=. conda run -n datasci python -W ignore notes/znb1_ctrl1_lateness_histogram.py
"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (SLA_HARD_DAYS, SLA_SOFT_DAYS, load_cases, load_events, load_invoices, use_invoice_proxy_labels)

use_invoice_proxy_labels()  # chart text: invoice -> invoice proxy

NOTES = Path(__file__).resolve().parent
PNG = NOTES / "figures" / "znb1_ctrl1_lateness_histogram.png"
CSV = NOTES / "znb1_ctrl1_lateness_histogram.csv"
K = ["PO_ID", "SAP_PO_No"]
DISCOUNT, INTEREST = 0.02, 0.125
CAP = 60          # left panel: last bar collects everything >= CAP days
TAIL_BIN = 5      # right panel bin width (days)

BLUE, ORANGE, RED = "#2a78d6", "#eb6834", "#a3261b"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"]})

# ---------------------------------------------------------------- data
ev = load_events()
z = ev[ev.SAP_PO_Type_Code == "ZNB1"].sort_values(["PO_ID", "ts", "Event_ID"])
route = z[z.Activity_ID == "Initial validation"].groupby(K).SAP_Next_Processing_Type.first().rename("route")
inv = load_invoices()
inv = inv[inv.po_type == "ZNB1"].set_index(K).join(route)
po_value = load_cases().set_index("PO_ID").invoice_value_nok
inv["n_posted"] = inv.groupby(level="PO_ID").posted.transform("sum")
inv["amount"] = np.where(inv.posted, po_value.reindex(inv.index.get_level_values(0)).values
                         / inv.n_posted.where(inv.n_posted > 0), np.nan)
ctrl_all = inv[inv.route == "CTRL1"]
m = ctrl_all[ctrl_all.posted].copy()
d = m.time_to_post_days
m["band"] = np.select([d <= SLA_SOFT_DAYS, d <= SLA_HARD_DAYS], ["≤ 10 days", "10–30 days"], "> 30 days")
m["discount"] = DISCOUNT * m.amount * (d > SLA_SOFT_DAYS)
m["interest"] = m.amount * INTEREST * (d - SLA_HARD_DAYS).clip(lower=0) / 365

n = len(m)
band = m.groupby("band").agg(invoices=("amount", "size"), amount=("amount", "sum"),
                             discount=("discount", "sum"), interest=("interest", "sum"))
band = band.reindex(["≤ 10 days", "10–30 days", "> 30 days"])
band["share"] = band.invoices / n
late10, late30 = m[d > SLA_SOFT_DAYS], m[d > SLA_HARD_DAYS]
band.to_csv(CSV)

# ---------------------------------------------------------------- chart
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 6.4), dpi=200, gridspec_kw={"width_ratios": [1.7, 1], "wspace": 0.12})
fig.patch.set_facecolor(SURFACE)


def color_for(day):
    return BLUE if day < SLA_SOFT_DAYS else ORANGE if day < SLA_HARD_DAYS else RED


# Left: 0..CAP days, 1-day bins; bin k = (k, k+1] days, last bin = >= CAP
days = d.clip(upper=CAP)
counts = np.bincount(np.minimum(np.floor(days).astype(int), CAP), minlength=CAP + 1)
x = np.arange(CAP + 1)
a1.bar(x + 0.5, counts, width=0.8, color=[color_for(k) for k in x], zorder=2)
for xv, lab in ((SLA_SOFT_DAYS, "10 days: discount lost"), (SLA_HARD_DAYS, "30 days: payment deadline")):
    a1.axvline(xv, color=INK2, lw=1, ls="--", zorder=3)
    a1.text(xv + 0.4, counts.max() * 1.12, lab, fontsize=9, color=INK2, va="top")
a1.set_xlim(0, CAP + 1.2)
a1.set_xticks([0, 5, 10, 15, 20, 25, 30, 40, 50, CAP + 0.5], ["0", "5", "10", "15", "20", "25", "30", "40", "50", f"{CAP}+"])
a1.set_ylim(0, counts.max() * 1.52)
a1.set_yticks(np.arange(0, counts.max() + 1, 200))
a1.set_xlabel("Days to first posting", color=INK2, fontsize=10)
a1.set_ylabel("Invoices", color=INK2, fontsize=10)
a1.set_title(f"All {n:,} CTRL1 invoices, one bar per day", loc="left", fontsize=11.5, color=INK, pad=10)

# band summaries placed over each zone
ymax = counts.max()
for (lab, r), xpos, col in zip(band.iterrows(), (5, 20, 45), (BLUE, ORANGE, RED)):
    extra = ("discount kept" if lab == "≤ 10 days" else
             f"{r.discount / 1e6:.2f}M NOK discount missed" if lab == "10–30 days" else
             f"{r.discount / 1e6:.2f}M discount + {r.interest / 1e3:,.0f}k interest")
    a1.text(xpos, ymax * 1.47, f"{lab}\n{int(r.invoices):,} invoices ({r.share:.0%})\n{extra}",
            ha="center", va="top", fontsize=9.5, color=INK, linespacing=1.4,
            bbox=dict(boxstyle="round,pad=0.4", fc=SURFACE, ec=col, lw=1.2))

# Right: tail beyond 30 days, TAIL_BIN-day bins
t = late30.time_to_post_days
edges = np.arange(SLA_HARD_DAYS, np.ceil(t.max() / TAIL_BIN) * TAIL_BIN + TAIL_BIN, TAIL_BIN)
tc, _ = np.histogram(t, bins=edges)
a2.bar(edges[:-1] + TAIL_BIN / 2, tc, width=TAIL_BIN * 0.82, color=RED, zorder=2)
a2.set_xlim(SLA_HARD_DAYS, edges[-1])
a2.set_xticks(np.arange(SLA_HARD_DAYS, edges[-1] + 1, 15))
a2.set_xlabel("Days to first posting", color=INK2, fontsize=10)
a2.set_title(f"Zoom: the {len(late30):,} invoices posted after > 30 days", loc="left", fontsize=11.5, color=INK, pad=10)
a2.text(0.97, 0.95,
        f"Median {t.median():.0f} days, longest {t.max():.0f}\n"
        f"Amount {late30.amount.sum() / 1e6:,.1f}M NOK\n"
        f"Late interest {late30.interest.sum() / 1e3:,.0f}k NOK\n"
        f"(12.5% p.a. from day 30)", transform=a2.transAxes, ha="right", va="top", fontsize=9.5, color=INK,
        linespacing=1.4)

for ax in (a1, a2):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, zorder=0)
    ax.tick_params(axis="both", length=0, colors=INK2, labelsize=9)
    for s in ax.spines.values():
        s.set_visible(False)

fig.suptitle(f"ZNB1 manual queue (CTRL1): {len(late10) / n:.0%} of invoices are posted after more than 10 days, "
             f"{len(late30) / n:.1%} after more than 30", x=0.02, ha="left", y=0.975, fontsize=14.5, color=INK)
fig.text(0.02, 0.905,
         f"{n:,} posted invoices · {len(late10):,} late > 10 days: {late10.discount.sum() / 1e6:.2f}M NOK missed 2% discount · "
         f"{len(late30):,} late > 30 days: {late30.interest.sum() / 1e3:,.0f}k NOK late interest · "
         f"total {(m.discount.sum() + m.interest.sum()) / 1e6:.2f}M NOK", fontsize=10, color=INK2, ha="left")
fig.text(0.02, 0.012,
         "ZNB1 invoices routed to CTRL1 at their first initial validation and posted (all suppliers); "
         f"{int((~ctrl_all.posted).sum()):,} deleted or never-posted ones left out. Clock: goods receipt (WAIT_GR) if present, "
         "else arrival in AP, to first posting.\nAmount = PO value split evenly over the PO's posted invoices. "
         "Late interest assumes payment on the posting day.", fontsize=7.5, color=INK2, ha="left")
fig.subplots_adjust(left=0.06, right=0.98, top=0.8, bottom=0.15)
fig.savefig(PNG, facecolor=SURFACE)
plt.close(fig)

print(band.round(2).to_string())
print(f"n={n}, late10={len(late10)}, late30={len(late30)}, discount={m.discount.sum():,.0f}, interest={m.interest.sum():,.0f}")
