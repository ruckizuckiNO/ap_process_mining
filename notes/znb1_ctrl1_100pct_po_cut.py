"""Share of the missed 2% discount of the ZNB1 100%-manual suppliers that sits below / above a PO-value cut
(1.5, 1.75, 2.0 and 2.25M NOK).

Reading: with a cut at X, POs with PO value <= X would be handled automatically (posted within 10 days), so the
discount missed on them today could be claimed; POs above X stay manual and keep today's result.

Same scope and numbers as znb1_ctrl1_100pct_exposure.py (35 suppliers, all posted invoices, deleted and never-posted
left out, pipeline clock, invoice amount = PO value split evenly over the PO's posted invoices, total 9.97M NOK).
PO value = Case_Invoice_Value_NOK (the PO total).
-> notes/znb1_ctrl1_100pct_po_cut.csv + notes/figures/znb1_ctrl1_100pct_po_cut.png
Run from the repo root:  PYTHONPATH=. conda run -n datasci python -W ignore notes/znb1_ctrl1_100pct_po_cut.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import POSTING_ACTIVITIES, SLA_SOFT_DAYS, load_cases, load_events, load_invoices
from common import use_invoice_proxy_labels

use_invoice_proxy_labels()  # chart text: invoice -> invoice proxy

NOTES = Path(__file__).resolve().parent
PNG = NOTES / "figures" / "znb1_ctrl1_100pct_po_cut.png"
CSV = NOTES / "znb1_ctrl1_100pct_po_cut.csv"
K = ["PO_ID", "SAP_PO_No"]
DISCOUNT = 0.02
CUTS = [1.5e6, 1.75e6, 2.0e6, 2.25e6]

BLUE, GREY = "#2a78d6", "#c9c8c2"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"

# ---------------------------------------------------------------- selection (as in znb1_ctrl1_100pct_exposure.py)
ev = load_events()
ev = ev[ev.SAP_PO_Type_Code == "ZNB1"].sort_values(["PO_ID", "ts", "Event_ID"])
ev["nxt"] = ev.groupby(K).Activity_ID.shift(-1)
iv = ev[ev.Activity_ID == "Initial validation"].groupby(K)
route, nxt = iv.SAP_Next_Processing_Type.first(), iv.nxt.first()

inv = load_invoices()
inv = inv[inv.po_type == "ZNB1"].set_index(K)
inv["supplier"] = ev.groupby(K).Supplier_ID.first()
inv["route"] = route
share = inv[inv.route.notna()].groupby("supplier").route.agg(lambda x: (x == "CTRL1").mean())
s100 = share.index[share == 1]
direct = inv.posted & nxt.reindex(inv.index).isin(POSTING_ACTIVITIES)
s100 = s100.intersection(inv.loc[direct, "supplier"].unique())

po_value = load_cases().set_index("PO_ID").invoice_value_nok
inv["po_value"] = po_value.reindex(inv.index.get_level_values(0)).values
inv["n_posted"] = inv.groupby(level="PO_ID").posted.transform("sum")
inv["amount"] = np.where(inv.posted, inv.po_value / inv.n_posted.where(inv.n_posted > 0), 0)
sel = inv[inv.supplier.isin(s100) & inv.posted].copy()
sel["late"] = sel.time_to_post_days > SLA_SOFT_DAYS
sel["missed"] = DISCOUNT * sel.amount * sel.late
total = sel.missed.sum()
n_pos = sel.index.get_level_values("PO_ID").nunique()

# ---------------------------------------------------------------- per cut
rows = []
for c in CUTS:
    b = sel[sel.po_value <= c]
    rows.append({
        "cut_mnok": c / 1e6,
        "pos_below": b.index.get_level_values("PO_ID").nunique(), "pos_share": b.index.get_level_values("PO_ID").nunique() / n_pos,
        "invoices_below": len(b), "invoice_share": len(b) / len(sel),
        "amount_below_nok": b.amount.sum(), "amount_share": b.amount.sum() / sel.amount.sum(),
        "late_below": int(b.late.sum()),
        "missed_below_nok": b.missed.sum(), "missed_share_below": b.missed.sum() / total,
        "missed_above_nok": total - b.missed.sum(), "missed_share_above": 1 - b.missed.sum() / total,
    })
res = pd.DataFrame(rows)
res.to_csv(CSV, index=False)

# ---------------------------------------------------------------- chart: one stacked bar per cut (below | above)
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"]})
fig, ax = plt.subplots(figsize=(11, 5.2), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
y = np.arange(len(res))[::-1]
below, above = res.missed_below_nok / 1e6, res.missed_above_nok / 1e6
ax.barh(y, below, height=0.58, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=2,
        label="PO value ≤ cut: could be automated and claimed")
ax.barh(y, above, left=below, height=0.58, color=GREY, edgecolor=SURFACE, linewidth=2, zorder=2,
        label="PO value > cut: stays manual")
for yi, r in zip(y, res.itertuples()):
    ax.text(r.missed_below_nok / 1e6 / 2, yi, f"{r.missed_below_nok / 1e6:.2f}M  ({r.missed_share_below:.0%})",
            ha="center", va="center", fontsize=9.5, color="white", fontweight="bold")
    ax.text((r.missed_below_nok + r.missed_above_nok / 2) / 1e6, yi,
            f"{r.missed_above_nok / 1e6:.2f}M  ({r.missed_share_above:.0%})", ha="center", va="center",
            fontsize=9.5, color=INK)
    ax.text(total / 1e6 * 1.01, yi, f"{r.pos_share:.0%} of POs\n{r.invoice_share:.0%} of invoices",
            ha="left", va="center", fontsize=8.5, color=INK2)
ax.set_yticks(y)
ax.set_yticklabels([f"Cut at {c:.2f}M NOK".replace(".00M", ".0M").replace(".50M", ".5M") for c in res.cut_mnok],
                   fontsize=10, color=INK)
ax.set_xlim(0, total / 1e6 * 1.22)
ax.set_xlabel("Missed 2% cash discount (M NOK)", color=INK2, fontsize=10)
ax.grid(axis="x", color=GRID, zorder=0)
ax.tick_params(axis="both", length=0, colors=INK2)
for s in ax.spines.values():
    s.set_visible(False)
ax.legend(loc="upper left", bbox_to_anchor=(0, 1.1), ncol=2, frameon=False, fontsize=9, labelcolor=INK2,
          borderaxespad=0)
fig.suptitle(f"ZNB1 100%-manual suppliers: share of the {total / 1e6:.2f}M NOK missed discount below each PO-value cut",
             x=0.01, ha="left", y=0.97, fontsize=13, fontweight="bold", color=INK)
fig.text(0.01, 0.015, f"{len(s100)} suppliers with 100% of invoices routed to CTRL1 · {len(sel):,} posted invoices on "
         f"{n_pos:,} POs · deleted and never-posted invoices left out · late = posted > {SLA_SOFT_DAYS} days after "
         "arrival.\nPO value = Case_Invoice_Value_NOK (PO total). Right: share of POs / invoices at or below the cut.",
         ha="left", fontsize=7.5, color=INK2)
fig.subplots_adjust(left=0.15, right=0.98, top=0.8, bottom=0.17)
fig.savefig(PNG, facecolor=SURFACE)
pd.set_option("display.width", 200)
print(res.round(3).to_string())
print("total", total, "suppliers", len(s100), "invoices", len(sel), "POs", n_pos)
print(PNG)
