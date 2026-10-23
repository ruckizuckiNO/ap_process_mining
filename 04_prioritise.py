"""Step 4 input: one table that combines volume, cost (manual work), SLA risk and control status per PO type,
plus the routing decision at validation and who posts on the direct manual path."""
import numpy as np
import pandas as pd

from common import (OUT, PO_TYPES, BOT_USERS, POSTING_ACTIVITIES, SLA_SOFT_DAYS, SLA_HARD_DAYS, load_events, load_cases,
                    load_invoices)

ev = load_events()
cases = load_cases().set_index("PO_ID")
cap = pd.read_csv(OUT / "capability.csv", index_col="po_type")
ctrl = pd.read_csv(OUT / "control_chart_results.csv")

# Behaviour flags per case
act = ev.groupby(["PO_ID", "Activity_ID"]).size().unstack(fill_value=0)
# Rework = an activity repeated within the SAME invoice cycle; repeats across cycles are just more invoices
act_cycle = ev.groupby(["PO_ID", "SAP_PO_No", "Activity_ID"]).size()
cases["has_rework"] = (act_cycle > 1).groupby(level="PO_ID").any()
cases["invoice_cycles"] = ev.groupby("PO_ID").SAP_PO_No.nunique()
cases["auto_post_failed"] = act.get("Auto posting failed", 0) > 0
cases["approval"] = act.get("Sent for approval", 0) > 0
cases["deleted"] = act.get("Invoice document deleted", 0) > 0
cases["touchless"] = cases.manual_count == 0
cases["breach10"] = cases.throughput_days > SLA_SOFT_DAYS
cases["breach30"] = cases.throughput_days > SLA_HARD_DAYS
# Missed 2% cash discount, counted per invoice: an invoice (SAP_PO_No) posted more than 10 days after its first
# event misses 2% of its value. The value is only known per PO (the sum of its invoices), so it is split evenly over
# the PO's posted invoices. Invoices never posted (deleted, open) are not paid and are left out.
inv = load_invoices()
inv = inv[inv.posted].copy()
inv["value_nok"] = inv.PO_ID.map(cases.invoice_value_nok) / inv.groupby("PO_ID").SAP_PO_No.transform("size")
inv["late"] = inv.time_to_post_days > SLA_SOFT_DAYS
inv["missed_discount_nok"] = inv.value_nok * 0.02 * inv.late
cases["late_invoices"] = inv.groupby("PO_ID").late.sum().reindex(cases.index, fill_value=0)
cases["missed_discount_nok"] = inv.groupby("PO_ID").missed_discount_nok.sum().reindex(cases.index, fill_value=0)
# Upper bound for comparison: the whole PO value once the PO's last posting comes after more than 10 days
last_post = ev[ev.Activity_ID.isin(POSTING_ACTIVITIES)].groupby("PO_ID").ts.max()
po_lead_days = (last_post.reindex(cases.index) - cases.start).dt.total_seconds() / 86400
cases["missed_discount_po_nok"] = cases.invoice_value_nok * 0.02 * (po_lead_days > SLA_SOFT_DAYS)

g = cases.groupby("po_type")
p = pd.DataFrame({
    "name": pd.Series(PO_TYPES),
    "cases": g.size(),
    "touchless_rate": g.touchless.mean(),
    "mean_manual": g.manual_count.mean(),
    "total_manual": g.manual_count.sum(),
    "mean_trace_length": g.trace_length.mean(),
    "median_tp_days": g.throughput_days.median(),
    "rework_rate": g.has_rework.mean(),
    "mean_invoice_cycles": g.invoice_cycles.mean(),
    "multi_cycle_rate": g.invoice_cycles.agg(lambda s: (s > 1).mean()),
    "auto_post_failed_rate": g.auto_post_failed.mean(),
    "approval_rate": g.approval.mean(),
    "deleted_rate": g.deleted.mean(),
    "breach10_rate": g.breach10.mean(),
    "breach30_rate": g.breach30.mean(),
    "breach10_cases": g.breach10.sum(),
    "breach30_cases": g.breach30.sum(),
    "median_invoice_lead_days": inv.groupby("po_type").time_to_post_days.median(),
    "mean_invoice_lead_days": inv.groupby("po_type").time_to_post_days.mean(),
    "late_invoices": g.late_invoices.sum(),
    "missed_discount_nok": g.missed_discount_nok.sum(),
    "missed_discount_po_nok": g.missed_discount_po_nok.sum(),
    "invoice_value_nok": g.invoice_value_nok.sum(),
    "suppliers": g.supplier.nunique(),
})
p["case_share"] = p.cases / p.cases.sum()
p["manual_share"] = p.total_manual / p.total_manual.sum()
p["breach10_share_of_all"] = p.breach10_cases / p.breach10_cases.sum()
p = p.join(cap[["Cpu_soft10", "Cpu_hard30"]])

strict = ctrl.groupby("po_type").in_control.all()
weekly = ctrl.groupby("po_type").weekly_I_in_control.agg(lambda s: int((s == True).sum()))
p["xbar_in_control_kpis"] = ctrl.groupby("po_type").in_control.sum().astype(int)
p["weekly_means_in_control_kpis"] = weekly
p = p.sort_values("cases", ascending=False)
p.index.name = "po_type"
p.to_csv(OUT / "priority.csv")

# Monthly drift of the case population (explains many control-chart signals)
m = cases.groupby([cases.start.dt.to_period("M"), "po_type"]).agg(
    n=("trace_length", "size"), median_tp=("throughput_days", "median"),
    breach10=("breach10", "mean"), manual=("manual_count", "mean")).reset_index()
m.to_csv(OUT / "monthly_by_type.csv", index=False)

# Routing at validation: SAP_Next_Processing_Type on Initial validation picks the road (WAITGR = goods-receipt
# check before auto-posting, CTRL1 = manual control queue, ZWMGRW = goods receipt from WM first, DELE = deletion)
ev["next_activity"] = ev.groupby(["PO_ID", "SAP_PO_No"]).Activity_ID.shift(-1).fillna("(end)")
iv = ev[ev.Activity_ID == "Initial validation"]
codes = iv.groupby("SAP_Next_Processing_Type").agg(validations=("PO_ID", "size"))
codes["share"] = codes.validations / codes.validations.sum()
nxt = iv.groupby("SAP_Next_Processing_Type").next_activity.value_counts(normalize=True)
codes["top_next_activity"] = nxt.groupby(level=0).idxmax().str[1]
codes["top_next_share"] = nxt.groupby(level=0).max()
codes.index.name = "code"
codes.sort_values("validations", ascending=False).to_csv(OUT / "routing_codes.csv")
# Per PO type, counted once per invoice: the first validation of each SAP_PO_No
first_iv = iv.groupby(["PO_ID", "SAP_PO_No"]).first()
routing = pd.crosstab(first_iv.SAP_PO_Type_Code, first_iv.SAP_Next_Processing_Type, normalize="index")
routing["invoices"] = first_iv.groupby("SAP_PO_Type_Code").size()
routing = routing.reindex(p.index)
routing.index.name = "po_type"
routing.to_csv(OUT / "routing.csv")

# Direct path (validation straight to manual posting): who posts, and how long after validation.
# User280 (nightly batch) does many of these "manual" postings.
DIRECT = ("Data management", "Initial validation", "Posted manually")
trace = ev.groupby("PO_ID").Activity_ID.agg(tuple)
d = ev[ev.PO_ID.isin(trace.index[trace == DIRECT])]
t = d.pivot(index="PO_ID", columns="Activity_ID", values="ts")
post = d[d.Activity_ID == "Posted manually"].set_index("PO_ID")
post["days"] = (t["Posted manually"] - t["Initial validation"]).dt.total_seconds() / 86400
post["by_bot"] = post.User_ID.isin(BOT_USERS)


def direct_stats(x):
    bot, hum = x[x.by_bot].days, x[~x.by_bot].days
    return pd.Series({"cases": len(x), "bot_share": x.by_bot.mean(), "median_days_bot": bot.median(),
                      "median_days_human": hum.median(), "p90_days_human": hum.quantile(0.9)})


direct = post.groupby("SAP_PO_Type_Code").apply(direct_stats).reindex(p.index)
direct.loc["ALL"] = direct_stats(post)
direct.index.name = "po_type"
direct.to_csv(OUT / "direct_posting.csv")

# ZNB1 invoice lead time and missed 2% discount by top-5 case variant
LT_TYPE = "ZNB1"
li = inv[inv.po_type == LT_TYPE].copy()
variant = ev[ev.SAP_PO_Type_Code == LT_TYPE].groupby("PO_ID").Activity_ID.agg(" → ".join)
rank = variant.map({v: i for i, v in enumerate(variant.value_counts().index, 1)})
li["variant_rank"] = li.PO_ID.map(rank.where(rank <= 5, 0))  # 0 = all other variants


def lead_stats(x):
    return pd.Series({"cases": x.PO_ID.nunique(), "invoices": len(x),
                      "median_days": x.time_to_post_days.median(), "mean_days": x.time_to_post_days.mean(),
                      "late_invoices": x.late.sum(), "missed_discount_nok": x.missed_discount_nok.sum(),
                      "missed_discount_po_nok": cases.loc[x.PO_ID.unique(), "missed_discount_po_nok"].sum()})


lt = li.groupby("variant_rank").apply(lead_stats)
lt.loc["ALL"] = lead_stats(li)
lt.index.name = "variant_rank"
lt.to_csv(OUT / "lead_time_variants.csv")

pd.set_option("display.width", 250)
print(p.drop(columns=["name"]).round(3).to_string())
print("total missed discount NOK", p.missed_discount_nok.sum().round(0), "of invoice value", p.invoice_value_nok.sum().round(0))
print("overall touchless", cases.touchless.mean(), "manual events", cases.manual_count.sum())
print(cases.groupby(cases.start.dt.to_period("M")).agg(n=("touchless", "size"), touchless=("touchless", "mean"),
      breach10=("breach10", "mean"), completed=("completed", "mean")).round(3))
print(codes.round(3).to_string())
print(routing.round(2).to_string())
print(direct.round(2).to_string())
print(lt.round(2).to_string())
