"""Step 4 input: one table that combines volume, cost (manual work), SLA risk and control status per PO type."""
import numpy as np
import pandas as pd

from common import OUT, PO_TYPES, SLA_SOFT_DAYS, SLA_HARD_DAYS, load_events, load_cases

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
    # Upper bound: assumes every invoice had 2%/10-day terms
    "discount_at_risk_nok": cases[cases.breach10].groupby("po_type").invoice_value_nok.sum() * 0.02,
    "invoice_value_nok": g.invoice_value_nok.sum(),
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

pd.set_option("display.width", 250)
print(p.drop(columns=["name"]).round(3).to_string())
print("total discount at risk NOK", p.discount_at_risk_nok.sum().round(0), "of invoice value", p.invoice_value_nok.sum().round(0))
print("overall touchless", cases.touchless.mean(), "manual events", cases.manual_count.sum())
print(cases.groupby(cases.start.dt.to_period("M")).agg(n=("touchless", "size"), touchless=("touchless", "mean"),
      breach10=("breach10", "mean"), completed=("completed", "mean")).round(3))
