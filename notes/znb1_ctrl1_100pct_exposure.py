"""Missed 2% cash discount exposure of the ZNB1 suppliers that go 100% to the manual queue (CTRL1).

Selection
  - suppliers: 100% of their ZNB1 invoices routed to CTRL1 at the first Initial validation, and at least one invoice
    posted straight from the queue (same 35 suppliers as znb1_ctrl1_100pct_suppliers.py; this drops Supplier524,
    whose only invoice went via approval)
  - invoices: all *posted* invoices of those suppliers, whatever path follows the queue (direct manual post or via
    approval); deleted and never-posted invoices are left out
  - late = first posting more than 10 days after the clock start (pipeline clock: arrival in AP; only 2 of the
    ~3,300 invoices have a WAIT_GR step and start at goods receipt, neither late, ~0.9M NOK)
  - invoice amount = PO value split evenly over the PO's posted invoices (pipeline rule)
  - exposure = 2% of the amount of the late invoices
-> notes/figures/znb1_ctrl1_100pct_exposure.png + notes/znb1_ctrl1_100pct_exposure.csv
Run from the repo root:  PYTHONPATH=. conda run -n datasci python -W ignore notes/znb1_ctrl1_100pct_exposure.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import POSTING_ACTIVITIES, SLA_SOFT_DAYS, load_cases, load_events, load_invoices

NOTES = Path(__file__).resolve().parent
PNG = NOTES / "figures" / "znb1_ctrl1_100pct_exposure.png"
CSV = NOTES / "znb1_ctrl1_100pct_exposure.csv"
K = ["PO_ID", "SAP_PO_No"]
DISCOUNT = 0.02
TOP = 10

BLUE, MUTED = "#2a78d6", "#9aa7a1"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"

# ---------------------------------------------------------------- data
ev = load_events()
ev = ev[ev.SAP_PO_Type_Code == "ZNB1"].sort_values(["PO_ID", "ts", "Event_ID"])
route = ev[ev.Activity_ID == "Initial validation"].groupby(K).SAP_Next_Processing_Type.first()
ev["nxt"] = ev.groupby(K).Activity_ID.shift(-1)
nxt = ev[ev.Activity_ID == "Initial validation"].groupby(K).nxt.first()
supplier = ev.groupby(K).Supplier_ID.first()

inv = load_invoices()
inv = inv[inv.po_type == "ZNB1"].set_index(K)
inv["supplier"] = supplier
inv["route"] = route
v = inv[inv.route.notna()]
share = v.groupby("supplier").route.agg(lambda x: (x == "CTRL1").mean())
s100 = share.index[share == 1]
direct = inv.posted & nxt.reindex(inv.index).isin(POSTING_ACTIVITIES)
s100 = s100.intersection(inv.loc[direct, "supplier"].unique())

po_value = load_cases().set_index("PO_ID").invoice_value_nok
inv["n_posted"] = inv.groupby(level="PO_ID").posted.transform("sum")
inv["amount"] = np.where(inv.posted, po_value.reindex(inv.index.get_level_values(0)).values
                         / inv.n_posted.where(inv.n_posted > 0), 0)

sel = inv[inv.supplier.isin(s100) & inv.posted].copy()
print("with WAIT_GR:", sel.goods_receipt.notna().sum(), "of", len(sel))
sel["late"] = sel.time_to_post_days > SLA_SOFT_DAYS
sel["exposure"] = DISCOUNT * sel.amount * sel.late

sup = sel.groupby("supplier").agg(invoices=("late", "size"), late=("late", "sum"), amount=("amount", "sum"),
                                  late_amount=("amount", lambda a: a[sel.loc[a.index, "late"]].sum()),
                                  exposure=("exposure", "sum"))
sup["late_share"] = sup.late / sup.invoices
sup = sup.sort_values(["exposure", "amount"], ascending=False)
sup.to_csv(CSV)
tot = sup.sum()
n_sup_exposed = int((sup.exposure > 0).sum())

# ---------------------------------------------------------------- chart
top, rest = sup.head(TOP), sup.iloc[TOP:]
labels = list(top.index) + [f"Other {len(rest)} suppliers"]
expo = list(top.exposure / 1e6) + [rest.exposure.sum() / 1e6]
late = list(top.late) + [rest.late.sum()]
ninv = list(top.invoices) + [rest.invoices.sum()]
colors = [BLUE] * len(top) + [MUTED]

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"]})
fig, ax = plt.subplots(figsize=(11, 6.4), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
y = np.arange(len(labels))[::-1]
ax.barh(y, expo, height=0.62, color=colors, zorder=2)
xmax = max(expo)
for yi, e, l, n in zip(y, expo, late, ninv):
    ax.text(e + xmax * 0.01, yi, f"{e:,.2f}M NOK  ({e / (tot.exposure / 1e6):.0%})  ·  {l:,} of {n:,} invoices late",
            va="center", ha="left", fontsize=9, color=INK)
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=10, color=INK)
ax.set_xlim(0, xmax * 1.45)
ax.set_xlabel("Missed 2% cash discount (M NOK)", color=INK2, fontsize=10)
ax.grid(axis="x", color=GRID, zorder=0)
ax.tick_params(axis="both", length=0, colors=INK2)
for s in ax.spines.values():
    s.set_visible(False)
fig.suptitle(f"ZNB1 100%-manual suppliers: {tot.exposure / 1e6:,.2f}M NOK cash discount missed on invoices "
             f"posted after > {SLA_SOFT_DAYS} days", x=0.01, ha="left", y=0.975, fontsize=13, fontweight="bold",
             color=INK)
fig.text(0.01, 0.905,
         f"{len(sup)} suppliers · {int(tot.invoices):,} posted invoices worth {tot.amount / 1e6:,.0f}M NOK · "
         f"{int(tot.late):,} late ({tot.late / tot.invoices:.0%}) worth {tot.late_amount / 1e6:,.0f}M NOK · "
         f"exposure = 2% of that · {n_sup_exposed} suppliers with any exposure",
         ha="left", fontsize=9.5, color=INK2)
fig.text(0.01, 0.015, "Suppliers with 100% of invoices routed to CTRL1 at initial validation. Deleted and never-posted "
         "invoices left out. Clock: arrival in AP to first posting (pipeline clock). Invoice amount = PO value split evenly over the "
         "PO's posted invoices.", ha="left", fontsize=7.5, color=INK2)
fig.subplots_adjust(left=0.17, right=0.98, top=0.85, bottom=0.12)
fig.savefig(PNG, facecolor=SURFACE)
print(sup.head(TOP + 3).to_string())
print(tot.to_string())
print("suppliers", len(sup), "exposed", n_sup_exposed)
print(PNG)
