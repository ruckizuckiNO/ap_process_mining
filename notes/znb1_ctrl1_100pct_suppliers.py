"""ZNB1 suppliers whose invoices go 100% to the manual queue (CTRL1), invoices that go from CTRL1 straight to posting.

Outputs (for further use in the report / slides):
  notes/figures/znb1_ctrl1_100pct_suppliers.png   three panels: invoice amount, missed 2% discount, share > 10 days
  notes/figures/znb1_ctrl1_100pct_suppliers_amount.png   invoice amount only
  notes/znb1_ctrl1_100pct_suppliers.xlsx          sheets Suppliers (formulas over Invoices), Invoices, Notes

Selection
  - ZNB1 invoices (PO_ID x SAP_PO_No) with a routing code at their first Initial validation
  - suppliers for which 100% of those invoices were routed to CTRL1
  - invoices whose next step after that validation is a posting (in practice always "Posted manually")
Clock: arrival in AP (first event, Data management) -> first posting; none of these invoices has a WAIT_GR step.
Invoice value: PO value (Case_Invoice_Value_NOK) split evenly over the PO's posted invoices (same rule as the pipeline).

Run from the repo root:  PYTHONPATH=. conda run -n datasci python -W ignore notes/znb1_ctrl1_100pct_suppliers.py
"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from openpyxl import Workbook
from openpyxl.comments import Comment
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from common import POSTING_ACTIVITIES, SLA_HARD_DAYS, SLA_SOFT_DAYS, load_cases, load_events, load_invoices
from common import use_invoice_proxy_labels

use_invoice_proxy_labels()  # chart text: invoice -> invoice proxy

NOTES = Path(__file__).resolve().parent
FIGS = NOTES / "figures"
K = ["PO_ID", "SAP_PO_No"]
DISCOUNT = 0.02
TOP = 10

# ---------------------------------------------------------------- data
ev = load_events()
ev = ev[ev.SAP_PO_Type_Code == "ZNB1"].sort_values(["PO_ID", "ts", "Event_ID"])
ev["nxt"] = ev.groupby(K).Activity_ID.shift(-1)
fv = ev[ev.Activity_ID == "Initial validation"].groupby(K).agg(route=("SAP_Next_Processing_Type", "first"), nxt=("nxt", "first"))
post = ev[ev.Activity_ID.isin(POSTING_ACTIVITIES)].groupby(K).User_ID.first().rename("poster")
inv = load_invoices()
inv = inv[inv.po_type == "ZNB1"].set_index(K).join(fv).join(post)
inv["supplier"] = ev.groupby(K).Supplier_ID.first()
cases = load_cases().set_index("PO_ID")
inv["n_posted"] = inv.groupby(level="PO_ID").posted.transform("sum")
inv["value"] = np.where(inv.posted, cases.invoice_value_nok.reindex(inv.index.get_level_values(0)).values
                        / inv.n_posted.where(inv.n_posted > 0), 0)

v = inv[inv.route.notna()]
share = v.groupby("supplier").route.agg(lambda x: (x == "CTRL1").mean())
s100 = share.index[share == 1]
sel = inv[inv.supplier.isin(s100) & inv.posted & inv.nxt.isin(POSTING_ACTIVITIES)].copy()
assert sel.goods_receipt.isna().all()
sel["days"] = (sel.first_post - sel.start).dt.total_seconds() / 86400
sel = sel.reset_index().sort_values(["supplier", "start"])
sel["late10"], sel["late30"] = sel.days > SLA_SOFT_DAYS, sel.days > SLA_HARD_DAYS
sel["missed"] = DISCOUNT * sel.value * sel.late10
all_amount = inv.loc[inv.posted, "value"].sum()
all_missed = DISCOUNT * inv.loc[inv.posted & (inv.time_to_post_days > SLA_SOFT_DAYS), "value"].sum()

sup = sel.groupby("supplier").agg(invoices=("days", "size"), pos=("PO_ID", "nunique"), amount=("value", "sum"),
                                  median=("days", "median"), p90=("days", lambda x: x.quantile(0.9)),
                                  n10=("late10", "sum"), n30=("late30", "sum"), missed=("missed", "sum"))
sup = sup.sort_values("amount", ascending=False)
sup["s10"] = sup.n10 / sup.invoices
tot = dict(n=len(sel), pos=sel.PO_ID.nunique(), sup=len(sup), amount=sel.value.sum(), missed=sel.missed.sum(),
           n10=int(sel.late10.sum()), n30=int(sel.late30.sum()), med=sel.days.median(), p90=sel.days.quantile(0.9))

# ---------------------------------------------------------------- chart: three panels, same supplier order
top = sup.head(TOP)
rest = sup.iloc[TOP:]
labels = list(top.index) + [f"Other {len(rest)} suppliers"]
amount = list(top.amount / 1e6) + [rest.amount.sum() / 1e6]
missed = list(top.missed / 1e6) + [rest.missed.sum() / 1e6]
s10 = list(top.s10 * 100) + [rest.n10.sum() / rest.invoices.sum() * 100]
ninv = list(top.invoices) + [rest.invoices.sum()]

INK, MUTED, GRID, BAR, OTHER = "#16201c", "#5a6a63", "#d9e1dd", "#2a78d6", "#9aa7a1"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": INK})
fig, axes = plt.subplots(1, 3, figsize=(14, 6.2), sharey=True, gridspec_kw={"wspace": 0.08})
y = np.arange(len(labels))[::-1]
colors = [BAR] * len(top) + [OTHER]
panels = [(amount, "Invoice amount (M NOK)", "{:,.0f}"), (missed, "Missed 2% discount (M NOK)", "{:,.2f}"),
          (s10, f"Invoices posted after > {SLA_SOFT_DAYS} days (%)", "{:.0f}%")]
for ax, (vals, title, fmt) in zip(axes, panels):
    ax.barh(y, vals, color=colors, height=0.62)
    top_v = max(vals)
    for yi, val in zip(y, vals):
        ax.text(val + top_v * 0.015, yi, fmt.format(val), va="center", fontsize=9, color=INK)
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
    ax.set_xlim(0, top_v * 1.22)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
axes[0].set_yticks(y)
axes[0].set_yticklabels([f"{l}  ({n:,})" for l, n in zip(labels, ninv)])
fig.suptitle("ZNB1: suppliers whose invoices always go to the manual queue (CTRL1), posted manually straight from the queue",
             x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
fig.text(0.01, 0.905,
         f"{tot['sup']} suppliers · {tot['n']:,} invoices on {tot['pos']:,} POs · invoice amount {tot['amount'] / 1e6:,.0f}M NOK "
         f"({tot['amount'] / all_amount:.0%} of posted ZNB1) · missed discount {tot['missed'] / 1e6:.2f}M NOK "
         f"({tot['missed'] / all_missed:.0%} of ZNB1) · > {SLA_HARD_DAYS} days: {tot['n30']} invoice ({tot['n30'] / tot['n']:.2%})",
         ha="left", fontsize=10, color=MUTED)
fig.text(0.01, 0.015, "Number of invoices in brackets. Sorted by invoice amount. Clock: arrival in AP → first posting.\n"
         "Invoice amount = PO value split evenly over the PO's posted invoices. Missed discount = 2% of late invoices (> 10 days).",
         ha="left", fontsize=8.5, color=MUTED)
fig.subplots_adjust(left=0.17, right=0.98, top=0.84, bottom=0.115)
png = FIGS / "znb1_ctrl1_100pct_suppliers.png"
fig.savefig(png, dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- chart: invoice amount only
fig, ax = plt.subplots(figsize=(11, 6.2))
ax.barh(y, amount, color=colors, height=0.62)
top_v = max(amount)
for yi, val, n in zip(y, amount, ninv):
    ax.text(val + top_v * 0.01, yi, f"{val:,.0f}M NOK  ({val / (tot['amount'] / 1e6):.0%})", va="center", fontsize=9,
            color=INK)
ax.set_xlim(0, top_v * 1.25)
ax.set_xlabel("Invoice amount (M NOK)")
ax.grid(axis="x", color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
for s_ in ["top", "right", "left"]:
    ax.spines[s_].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.set_yticks(y)
ax.set_yticklabels([f"{l}  ({n:,})" for l, n in zip(labels, ninv)])
fig.suptitle("ZNB1: invoice amount of suppliers that always go to the manual queue (CTRL1)",
             x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
fig.text(0.01, 0.905,
         f"{tot['sup']} suppliers · {tot['n']:,} invoices on {tot['pos']:,} POs · "
         f"{tot['amount'] / 1e6:,.0f}M NOK in total ({tot['amount'] / all_amount:.0%} of posted ZNB1 value)",
         ha="left", fontsize=10, color=MUTED)
fig.text(0.01, 0.015, "Number of invoices in brackets; % = share of this group's amount. Invoices posted straight from "
         "the queue.\nInvoice amount = PO value split evenly over the PO's posted invoices.",
         ha="left", fontsize=8.5, color=MUTED)
fig.subplots_adjust(left=0.2, right=0.97, top=0.84, bottom=0.15)
png_amount = FIGS / "znb1_ctrl1_100pct_suppliers_amount.png"
fig.savefig(png_amount, dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- workbook
F = "Arial"
H_FILL = PatternFill("solid", start_color="1F6F5C")
H_FONT = Font(name=F, bold=True, color="FFFFFF")
B_FONT, BOLD = Font(name=F), Font(name=F, bold=True)
INPUT = Font(name=F, color="0000FF")
THIN = Border(top=Side(style="thin", color="999999"))
NOK, PCT, DAYS, INT = "#,##0;(#,##0);-", "0.0%;(0.0%);-", "0.0", "#,##0;(#,##0);-"

wb = Workbook()
ws_s = wb.active
ws_s.title = "Suppliers"
ws_i = wb.create_sheet("Invoices")
ws_n = wb.create_sheet("Notes")

# Notes (assumptions referenced by formulas)
notes = [
    ("ZNB1 suppliers 100% via CTRL1: invoices posted straight from the queue", None),
    ("", None),
    ("Discount rate", DISCOUNT),
    ("Soft limit (days, cash discount)", SLA_SOFT_DAYS),
    ("Hard limit (days, payment deadline)", SLA_HARD_DAYS),
    ("", None),
    ("Selection", "ZNB1 invoices (PO_ID x SAP_PO_No); suppliers with 100% of invoices routed to CTRL1 at their first Initial validation; "
                  "only invoices whose next step after validation is a posting (Posted manually)."),
    ("Clock", "Arrival in AP (first event, Data management) to first posting, calendar days. None of these invoices has a WAIT_GR step."),
    ("Invoice amount", "Estimated: PO value (Case_Invoice_Value_NOK) split evenly over the PO's posted invoices. The log has no value per invoice."),
    ("Missed discount", "Discount rate x invoice amount for invoices posted after more than the soft limit. Assumes 2/10 terms for every invoice (not in the log)."),
    ("Median / 90th pct days", "Computed in Python from the Invoices sheet (static values)."),
    ("Source", "Event log via output/events.pkl, invoices.csv, cases.csv; script notes/znb1_ctrl1_100pct_suppliers.py"),
    ("Reference: all posted ZNB1 invoices", None),
    ("Invoice amount (NOK)", round(all_amount, 2)),
    ("Missed 2% discount, invoice clock (NOK)", round(all_missed, 2)),
]
for r, (a, b) in enumerate(notes, 1):
    ws_n.cell(r, 1, a).font = BOLD if r in (1, 13) else B_FONT
    if b is not None:
        c = ws_n.cell(r, 2, b)
        c.font = INPUT if r in (3, 4, 5) else B_FONT
        c.alignment = Alignment(wrap_text=True, vertical="top")
ws_n["B3"].number_format = "0%"
ws_n["B14"].number_format = ws_n["B15"].number_format = NOK
ws_n["C3"] = "Blue = input you can change; all sheets recalculate."
ws_n["C3"].font = Font(name=F, italic=True, color="5A6A63")
ws_n.column_dimensions["A"].width = 40
ws_n.column_dimensions["B"].width = 110
RATE, SOFT, HARD = "Notes!$B$3", "Notes!$B$4", "Notes!$B$5"

# Invoices (one row per invoice; flags and discount as formulas)
ih = ["Supplier", "PO_ID", "SAP_PO_No", "Arrival in AP", "First posting", "Posted by", "Days to posting",
      "Invoice amount (NOK, est.)", "> soft limit", "> hard limit", "Missed discount (NOK)"]
for j, h in enumerate(ih, 1):
    c = ws_i.cell(1, j, h)
    c.font, c.fill, c.alignment = H_FONT, H_FILL, Alignment(wrap_text=True, vertical="center")
for i, r in enumerate(sel.itertuples(index=False), 2):
    ws_i.cell(i, 1, r.supplier)
    ws_i.cell(i, 2, r.PO_ID)
    ws_i.cell(i, 3, int(r.SAP_PO_No))
    ws_i.cell(i, 4, r.start.to_pydatetime()).number_format = "yyyy-mm-dd hh:mm"
    ws_i.cell(i, 5, r.first_post.to_pydatetime()).number_format = "yyyy-mm-dd hh:mm"
    ws_i.cell(i, 6, r.poster)
    ws_i.cell(i, 7, f"=E{i}-D{i}").number_format = "0.00"
    ws_i.cell(i, 8, round(float(r.value), 2)).number_format = NOK
    ws_i.cell(i, 9, f"=IF(G{i}>{SOFT},1,0)")
    ws_i.cell(i, 10, f"=IF(G{i}>{HARD},1,0)")
    ws_i.cell(i, 11, f"=I{i}*{RATE}*H{i}").number_format = NOK
    for j in range(1, 12):
        ws_i.cell(i, j).font = B_FONT
last = len(sel) + 1
ws_i.freeze_panes = "A2"
ws_i.auto_filter.ref = f"A1:K{last}"
for j, w in enumerate([13, 15, 12, 17, 17, 11, 10, 16, 9, 9, 14], 1):
    ws_i.column_dimensions[get_column_letter(j)].width = w

# Suppliers (aggregates by formula over the Invoices sheet)
sh = ["Supplier", "Invoices", "POs", "Invoice amount (NOK)", "Share of amount", "Median days", "90th pct days",
      "Invoices > 10 d", "Share > 10 d", "Invoices > 30 d", "Share > 30 d", "Missed 2% discount (NOK)", "Share of missed"]
for j, h in enumerate(sh, 1):
    c = ws_s.cell(1, j, h)
    c.font, c.fill, c.alignment = H_FONT, H_FILL, Alignment(wrap_text=True, vertical="center")
ws_s["H1"].comment = Comment("Posted after more than the soft limit (Notes!B4, 10 days).", "analysis")
ws_s["J1"].comment = Comment("Posted after more than the hard limit (Notes!B5, 30 days).", "analysis")
ws_s["F1"].comment = Comment("Static values computed in Python from the Invoices sheet.", "analysis")
rng = lambda col: f"Invoices!${col}$2:${col}${last}"
n_s = len(sup)
tr = n_s + 2
for i, (s, r) in enumerate(sup.iterrows(), 2):
    ws_s.cell(i, 1, s)
    ws_s.cell(i, 2, f"=COUNTIF({rng('A')},A{i})").number_format = INT
    ws_s.cell(i, 3, int(r.pos)).number_format = INT
    ws_s.cell(i, 4, f"=SUMIF({rng('A')},A{i},{rng('H')})").number_format = NOK
    ws_s.cell(i, 5, f"=IF($D${tr}=0,0,D{i}/$D${tr})").number_format = PCT
    ws_s.cell(i, 6, round(float(r["median"]), 2)).number_format = DAYS
    ws_s.cell(i, 7, round(float(r.p90), 2)).number_format = DAYS
    ws_s.cell(i, 8, f"=SUMIF({rng('A')},A{i},{rng('I')})").number_format = INT
    ws_s.cell(i, 9, f"=IF(B{i}=0,0,H{i}/B{i})").number_format = PCT
    ws_s.cell(i, 10, f"=SUMIF({rng('A')},A{i},{rng('J')})").number_format = INT
    ws_s.cell(i, 11, f"=IF(B{i}=0,0,J{i}/B{i})").number_format = PCT
    ws_s.cell(i, 12, f"=SUMIF({rng('A')},A{i},{rng('K')})").number_format = NOK
    ws_s.cell(i, 13, f"=IF($L${tr}=0,0,L{i}/$L${tr})").number_format = PCT
    for j in range(1, 14):
        ws_s.cell(i, j).font = B_FONT
ws_s.cell(tr, 1, "Total")
ws_s.cell(tr, 2, f"=SUM(B2:B{tr - 1})").number_format = INT
ws_s.cell(tr, 3, int(tot["pos"])).number_format = INT
ws_s.cell(tr, 4, f"=SUM(D2:D{tr - 1})").number_format = NOK
ws_s.cell(tr, 5, f"=SUM(E2:E{tr - 1})").number_format = PCT
ws_s.cell(tr, 6, round(float(tot["med"]), 2)).number_format = DAYS
ws_s.cell(tr, 7, round(float(tot["p90"]), 2)).number_format = DAYS
ws_s.cell(tr, 8, f"=SUM(H2:H{tr - 1})").number_format = INT
ws_s.cell(tr, 9, f"=IF(B{tr}=0,0,H{tr}/B{tr})").number_format = PCT
ws_s.cell(tr, 10, f"=SUM(J2:J{tr - 1})").number_format = INT
ws_s.cell(tr, 11, f"=IF(B{tr}=0,0,J{tr}/B{tr})").number_format = PCT
ws_s.cell(tr, 12, f"=SUM(L2:L{tr - 1})").number_format = NOK
ws_s.cell(tr, 13, f"=SUM(M2:M{tr - 1})").number_format = PCT
for j in range(1, 14):
    ws_s.cell(tr, j).font, ws_s.cell(tr, j).border = BOLD, THIN
ws_s.cell(tr, 3).comment = Comment("Distinct POs over all suppliers (static).", "analysis")
# Context rows: share of all posted ZNB1
ws_s.cell(tr + 2, 1, "Share of all posted ZNB1").font = Font(name=F, italic=True)
ws_s.cell(tr + 2, 4, f"=D{tr}/Notes!$B$14").number_format = PCT
ws_s.cell(tr + 2, 12, f"=L{tr}/Notes!$B$15").number_format = PCT
for j in (4, 12):
    ws_s.cell(tr + 2, j).font = Font(name=F, italic=True)
ws_s.cell(tr + 3, 1, "Sorted by invoice amount. Definitions and inputs on the Notes sheet; formulas aggregate the Invoices sheet.").font = \
    Font(name=F, italic=True, color="5A6A63")
ws_s.freeze_panes = "B2"
for j, w in enumerate([14, 10, 8, 18, 11, 10, 10, 11, 10, 11, 10, 18, 11], 1):
    ws_s.column_dimensions[get_column_letter(j)].width = w
ws_s.row_dimensions[1].height = 32
ws_i.row_dimensions[1].height = 32

xlsx = NOTES / "znb1_ctrl1_100pct_suppliers.xlsx"
wb.calculation.fullCalcOnLoad = True  # no cached values from openpyxl: let Excel compute everything on open
wb.save(xlsx)
print("wrote", png, "and", xlsx)
print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in tot.items()}, "| suppliers 100%:", len(s100))
