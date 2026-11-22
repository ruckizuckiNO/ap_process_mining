"""ZNB1 deep dive: which suppliers miss the 10- and 30-day limits, prioritised by PO value and manual handling.

Unit is the invoice cycle (PO_ID x SAP_PO_No), as in 04_prioritise.py: SLA clock = first event -> first posting.
Run from the repo root:  conda run -n datasci python -W ignore notes/znb1_supplier_priority.py
"""
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = Path(os.environ.get("AP_EVENT_LOG", ROOT.parent / "log.csv"))
OUT = ROOT / "notes"
BOTS = ["User1587", "User280"]
POSTING = ["MM Posted Automatically", "Posted manually", "MM Autopost after manual process"]
APPROVAL = ["Sent for approval", "Approved MM", "Approved MM (accept if price diff)", "Approval rejected", "Approved"]

df = pd.read_csv(RAW, low_memory=False)
df = df[df.SAP_PO_Type_Code == "ZNB1"].copy()
df["ts"] = pd.to_datetime(df.Timestamp_DDMMYYYY, format="%d/%m/%Y %H:%M:%S")
df = df.sort_values(["PO_ID", "ts", "Event_ID"])
df["human"] = ~df.User_ID.isin(BOTS)
log_end = df.ts.max()

g = df.groupby(["PO_ID", "SAP_PO_No"])
inv = pd.DataFrame({
    "supplier": g.Supplier_ID.first(),
    "po_value_nok": g.Case_Invoice_Value_NOK.first(),
    "start": g.ts.min(),
    "goods_receipt": df[(df.Activity_ID == "Wait for goods receipt") & (df.SAP_Object_Code == "WAIT_GR")]
    .groupby(["PO_ID", "SAP_PO_No"]).ts.min(),
    "first_post": df[df.Activity_ID.isin(POSTING)].groupby(["PO_ID", "SAP_PO_No"]).ts.min(),
    "human_events": g.human.sum(),
    "posted_manually": g.Activity_ID.agg(lambda a: (a == "Posted manually").any()),
    "ctrl1": g.SAP_Next_Processing_Type.agg(lambda a: (a == "CTRL1").any()),
    "approval": g.Activity_ID.agg(lambda a: a.isin(APPROVAL).any()),
    "deleted": g.Activity_ID.agg(lambda a: (a == "Invoice document deleted").any()),
}).reset_index()
inv["posted"] = inv.first_post.notna()
# ZNB1 clock starts at goods receipt where the invoice waits for one (same rule as GR_CLOCK_PO_TYPES in common.py)
inv["clock_start"] = inv.goods_receipt.fillna(inv.start)
inv["days_to_post"] = (inv.first_post - inv.clock_start).dt.total_seconds() / 86400
# Open (never posted, not deleted) invoices count as missed once they are older than the limit at log end
age_open = (log_end - inv.clock_start).dt.total_seconds() / 86400
clock = inv.days_to_post.where(inv.posted, age_open.where(~inv.deleted))
inv["miss10"] = clock > 10
inv["miss30"] = clock > 30
inv["manual"] = inv.human_events > 0
# PO value is only known per PO: split evenly over the PO's invoices (same rule as 04_prioritise.py)
n_posted = inv.groupby("PO_ID").posted.transform("sum").clip(lower=1)
inv["value_nok"] = inv.po_value_nok / n_posted * inv.posted   # split over posted invoices; unposted carry no value
hv_cut = inv.drop_duplicates("PO_ID").po_value_nok.quantile(0.75)
inv["high_value_po"] = inv.po_value_nok >= hv_cut
inv["focus"] = inv.manual & inv.miss10                         # manual and late
inv["missed_discount_nok"] = inv.value_nok * 0.02 * (inv.posted & inv.miss10)
inv = inv[~inv.deleted | inv.posted]                            # deleted, never-posted invoices are not paid

s = inv.groupby("supplier")
sup = pd.DataFrame({
    "pos": s.PO_ID.nunique(),
    "high_value_pos": s.apply(lambda x: x.loc[x.high_value_po, "PO_ID"].nunique()),
    "po_value_nok": s.apply(lambda x: x.drop_duplicates("PO_ID").po_value_nok.sum()),
    "max_po_value_nok": s.po_value_nok.max(),
    "invoices": s.size(),
    "within_10d_pct": 100 * (1 - s.miss10.mean()),
    "within_30d_pct": 100 * (1 - s.miss30.mean()),
    "missed_10d": s.miss10.sum(),
    "missed_30d": s.miss30.sum(),
    "median_days_to_post": s.days_to_post.median(),
    "manual_pct": 100 * s.manual.mean(),
    "approval_invoices": s.approval.sum(),
    "manual_and_late": s.focus.sum(),
    "manual_late_value_nok": s.apply(lambda x: x.loc[x.focus, "value_nok"].sum()),
    "manual_late_highvalue_nok": s.apply(lambda x: x.loc[x.focus & x.high_value_po, "value_nok"].sum()),
    "missed_discount_nok": s.missed_discount_nok.sum(),
})
sup["share_of_znb1_late_value_pct"] = 100 * sup.manual_late_value_nok / sup.manual_late_value_nok.sum()
sup = sup.sort_values(["manual_late_highvalue_nok", "manual_late_value_nok"], ascending=False)
sup["cum_share_pct"] = sup.share_of_znb1_late_value_pct.cumsum()


def tier(r):
    if r.manual_late_highvalue_nok > 0 and r.missed_30d > 0:
        return "A: high-value, manual, misses 30d"
    if r.manual_late_highvalue_nok > 0:
        return "B: high-value, manual, misses 10d"
    if r.manual_and_late > 0:
        return "C: manual and late, smaller POs"
    return "D: on time or late without manual work"


sup["tier"] = sup.apply(tier, axis=1)
sup.insert(0, "rank", range(1, len(sup) + 1))
sup = sup.round({c: 1 for c in sup.columns if c.endswith("pct") or c.startswith("median")}).round(
    {c: 0 for c in sup.columns if c.endswith("nok")})
sup.to_csv(OUT / "znb1_supplier_priority.csv")
inv.to_csv(OUT / "znb1_supplier_priority_invoices.csv", index=False)

print(f"high-value PO cut (top quartile): {hv_cut:,.0f} NOK; log end {log_end}")
print(f"invoices {len(inv)}, within 10d {100 * (1 - inv.miss10.mean()):.1f}%, within 30d {100 * (1 - inv.miss30.mean()):.1f}%, "
      f"manual {100 * inv.manual.mean():.1f}%, manual&late {inv.focus.sum()}")
print(f"late among manual {100 * inv[inv.manual].miss10.mean():.1f}% vs automatic {100 * inv[~inv.manual].miss10.mean():.1f}%")
print(sup.tier.value_counts().sort_index().to_string())
pd.set_option("display.width", 250)
print(sup.head(25).drop(columns=["share_of_znb1_late_value_pct"]).to_string())
