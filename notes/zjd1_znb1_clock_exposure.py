"""ZJD1 vs ZNB1 per invoice on the invoice clock: invoices > 10 and > 30 days, missed discount and late-interest exposure.

Scope: every invoice (PO_ID x SAP_PO_No) of ZJD1 and ZNB1, all routes at validation (WAITGR, CTRL1, ZWMGRW, DELE,
none) and all posting kinds. Posted invoices are measured; never-posted ones are counted separately (no clock stop).
Clock (as in 10_report_invoice_clock.py, both types alike): start = first WAIT_GR event of the invoice, else its first
event; stop = first posting. Side view: ZJD1 on the pipeline clock (arrival, GR_CLOCK_PO_TYPES = ZNB1 only), and both
types on the arrival clock (upper bound).
Amount per invoice = PO value (Case_Invoice_Value_NOK) split evenly over the PO's posted invoices (pipeline rule).
  missed discount = 2% x amount                                   if days > 10
  late interest   = amount x 12.5% p.a. x (days - 30) / 365       if days > 30  (payment assumed on the posting day)
  exposure > 30 d = amount of the invoices posted after more than 30 days (value paid past the legal deadline)
-> notes/zjd1_znb1_clock_exposure.csv
Run from the repo root:  PYTHONPATH=. conda run -n datasci python -W ignore notes/zjd1_znb1_clock_exposure.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from common import POSTING_ACTIVITIES, SLA_HARD_DAYS, SLA_SOFT_DAYS, load_cases, load_events, load_invoices

CSV = Path(__file__).resolve().parent / "zjd1_znb1_clock_exposure.csv"
TYPES = ["ZJD1", "ZNB1"]
K = ["PO_ID", "SAP_PO_No"]
DISCOUNT, INTEREST = 0.02, 0.125

ev = load_events()
ev = ev[ev.SAP_PO_Type_Code.isin(TYPES)]
route = (ev[ev.Activity_ID == "Initial validation"].sort_values(["ts", "Event_ID"])
         .groupby(K).SAP_Next_Processing_Type.first().fillna("(none)").rename("route"))
post_act = (ev[ev.Activity_ID.isin(POSTING_ACTIVITIES)].sort_values(["ts", "Event_ID"])
            .groupby(K).Activity_ID.first().rename("post_activity"))

inv = load_invoices()
inv = inv[inv.po_type.isin(TYPES)].set_index(K).join(route).join(post_act)
inv["route"] = inv.route.fillna("(no validation)")
po_value = load_cases().set_index("PO_ID").invoice_value_nok
inv["n_posted_on_po"] = inv.groupby(level="PO_ID").posted.transform("sum")
inv["amount"] = np.where(inv.posted, po_value.reindex(inv.index.get_level_values(0)).values
                         / inv.n_posted_on_po.where(inv.n_posted_on_po > 0), np.nan)
inv["has_gr"] = inv.goods_receipt.notna()


def measure(d, start):
    """Add days/late flags/costs for clock start column `start`."""
    d = d.copy()
    d["days"] = (d.first_post - d[start]).dt.total_seconds() / 86400
    d["late10"], d["late30"] = d.days > SLA_SOFT_DAYS, d.days > SLA_HARD_DAYS
    d["discount"] = DISCOUNT * d.amount * d.late10
    d["interest"] = d.amount * INTEREST * (d.days - SLA_HARD_DAYS).clip(lower=0) / 365
    return d


def summary(d, n_unposted):
    return pd.Series({
        "posted": len(d), "not_posted": n_unposted, "amount_mnok": d.amount.sum() / 1e6,
        "median_days": d.days.median(), "late10": int(d.late10.sum()), "share10": d.late10.mean(),
        "late30": int(d.late30.sum()), "share30": d.late30.mean(),
        "amount_late10_mnok": d.amount[d.late10].sum() / 1e6, "exposure30_mnok": d.amount[d.late30].sum() / 1e6,
        "missed_discount_mnok": d.discount.sum() / 1e6, "late_interest_mnok": d.interest.sum() / 1e6,
        "total_cost_mnok": (d.discount.sum() + d.interest.sum()) / 1e6,
    })


inv["clock_invoice"] = inv.goods_receipt.fillna(inv.start)        # WAIT_GR, else first event (both types)
posted = inv[inv.posted]
rows = {}
for t in TYPES:
    d = posted[posted.po_type == t]
    nu = int((~inv[inv.po_type == t].posted).sum())
    rows[(t, "invoice clock", "all routes")] = summary(measure(d, "clock_invoice"), nu)
    rows[(t, "arrival clock (upper bound)", "all routes")] = summary(measure(d, "start"), nu)
    if t == "ZJD1":
        rows[(t, "pipeline clock (arrival)", "all routes")] = summary(measure(d, "clock_start"), nu)
    m = measure(d, "clock_invoice")
    for r, g in m.groupby("route"):
        rows[(t, "invoice clock", f"route {r}")] = summary(g, int((~inv[(inv.po_type == t) & (inv.route == r)].posted).sum()))
    for h, g in m.groupby("has_gr"):
        rows[(t, "invoice clock", "with WAIT_GR" if h else "without WAIT_GR")] = summary(g, np.nan)
tab = pd.DataFrame(rows).T
tab.index.names = ["po_type", "clock", "subset"]
tab.to_csv(CSV)

# Check: ZNB1 on the invoice clock is the pipeline headline (1,687 late / 15.9M NOK)
z = tab.loc[("ZNB1", "invoice clock", "all routes")]
print(f"check ZNB1: {int(z.late10)} late, {z.missed_discount_mnok:.2f}M missed discount")
pd.set_option("display.width", 250)
print(tab.round(3).to_string())
