"""ZNB1 suppliers whose invoices always go straight to the manual queue (CTRL1) at initial validation, whatever the
PO amount -> notes/znb1_always_manual_suppliers.xlsx (list, PO-value evidence, definitions).

"Always manual" = >= 5 POs and >= 95% of invoices routed to CTRL1 at initial validation. Clock as in the pipeline.
Run from the repo root:  PYTHONPATH=. conda run -n datasci python -W ignore notes/znb1_always_manual_suppliers.py
"""
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from common import SLA_SOFT_DAYS, load_cases, load_events, load_invoices

NOTES = Path(__file__).resolve().parent
OUT = NOTES / "znb1_always_manual_suppliers.xlsx"
K = ["PO_ID", "SAP_PO_No"]
MIN_POS, MIN_DIRECT = 5, 0.95
# Assumption (2026-11-19): leave out every ZNB1 PO that has at least one never-posted invoice (deleted, rejected at
# validation, moved to a new number or still open). Set to False for the previous version (only those invoices out).
EXCLUDE_POS_WITH_UNPOSTED = True

ev = load_events()
z = ev[ev.SAP_PO_Type_Code == "ZNB1"]
po_value = load_cases().set_index("PO_ID").invoice_value_nok
inv = load_invoices().set_index(K)
all_pos = inv[inv.po_type == "ZNB1"].index.get_level_values("PO_ID").unique()
bad_pos = inv[(inv.po_type == "ZNB1") & ~inv.posted].index.get_level_values("PO_ID").unique()
if EXCLUDE_POS_WITH_UNPOSTED:
    z = z[~z.PO_ID.isin(bad_pos)]
    inv = inv[~inv.index.get_level_values("PO_ID").isin(bad_pos)]
SCOPE = (f"Scope: ZNB1 POs without never-posted invoices ({len(all_pos) - len(bad_pos):,} of {len(all_pos):,} POs; "
         f"{len(bad_pos)} POs with at least one never-posted invoice left out)." if EXCLUDE_POS_WITH_UNPOSTED
         else f"Scope: all {len(all_pos):,} ZNB1 POs; never-posted invoices carry no value.")

iv = z[z.Activity_ID == "Initial validation"].sort_values("ts").drop_duplicates(K).set_index(K)
t = pd.DataFrame({"supplier": iv.Supplier_ID, "direct": iv.SAP_Next_Processing_Type == "CTRL1"})
t["po_value"] = po_value.reindex(t.index.get_level_values("PO_ID")).values
t["posted"] = inv.posted.reindex(t.index).fillna(False).astype(bool)
t["late"] = t.posted & (inv.time_to_post_days.reindex(t.index) > SLA_SOFT_DAYS)
n_posted = t.groupby(level="PO_ID").posted.transform("sum").clip(lower=1)
t["inv_value"] = t.po_value / n_posted * t.posted
t["missed_discount"] = 0.02 * t.inv_value * t.late
pm = z[z.Activity_ID == "Posted manually"].sort_values("ts").drop_duplicates(K).set_index(K)
t["poster"] = pm.User_ID.reindex(t.index)

g = t.groupby("supplier")
pos = t.reset_index().drop_duplicates("PO_ID").groupby("supplier").po_value
s = pd.DataFrame({
    "invoices": g.size(), "pos": pos.size(), "direct": g.direct.mean(),
    "min_po": pos.min(), "median_po": pos.median(), "max_po": pos.max(),
    "main_clerk": g.poster.agg(lambda u: u.value_counts().index[0] if u.notna().any() else ""),
    "main_clerk_share": g.poster.agg(lambda u: u.value_counts(normalize=True).iloc[0] if u.notna().any() else 0),
    "late_share": g.apply(lambda x: x.late.sum() / max(x.posted.sum(), 1)),
    "missed_discount": g.missed_discount.sum(),
})
hv_cut = t.reset_index().drop_duplicates("PO_ID").po_value.quantile(0.75)
lst = s[(s.pos >= MIN_POS) & (s.direct >= MIN_DIRECT)].sort_values("invoices", ascending=False)
lst["segment"] = (lst.median_po >= hv_cut).map({True: "Large POs", False: "Small POs"})

t["decile"] = pd.qcut(t.po_value.rank(method="first"), 10, labels=range(1, 11))
dec = t.groupby("decile", observed=True).agg(invoices=("direct", "size"), min_po=("po_value", "min"),
                                             max_po=("po_value", "max"), direct=("direct", "mean"))
mixed = s[(s.direct > 0.05) & (s.direct < 0.95) & (s.invoices >= 20)].index
mx = t[t.supplier.isin(mixed)].copy()
mx["q"] = pd.cut(mx.groupby("supplier").po_value.rank(pct=True), [0, 0.25, 0.5, 0.75, 1],
                 labels=["Smallest 25% of its POs", "25–50%", "50–75%", "Largest 25% of its POs"])
qt = mx.groupby("q", observed=True).agg(invoices=("direct", "size"), direct=("direct", "mean"))

# --- Workbook ------------------------------------------------------------------------------
F = "Arial"
H_FILL = PatternFill("solid", fgColor="14213D")
T_FILL = PatternFill("solid", fgColor="ECEAE3")
thin = Side(style="thin", color="CFCBC0")
NOK = '#,##0;(#,##0);"-"'
PCT = '0.0%;(0.0%);"-"'


def header(ws, row, cols):
    for c, h in enumerate(cols, 1):
        cell = ws.cell(row, c, h)
        cell.font = Font(name=F, bold=True, color="FFFFFF")
        cell.fill = H_FILL
        cell.alignment = Alignment(wrap_text=True, vertical="center")
    ws.row_dimensions[row].height = 32


def title(ws, text, sub):
    ws["A1"] = text
    ws["A1"].font = Font(name=F, bold=True, size=14, color="14213D")
    ws["A2"] = sub
    ws["A2"].font = Font(name=F, size=10, color="5F6672")


wb = Workbook()
ws = wb.active
ws.title = "Always manual"
title(ws, "ZNB1 suppliers always routed to the manual queue",
      f"{len(lst)} suppliers with ≥ {MIN_POS} POs and ≥ {MIN_DIRECT:.0%} of invoices sent straight to the manual "
      "queue (CTRL1) at initial validation, whatever the PO amount. Definitions on sheet 'Notes'. " + SCOPE)
cols = ["Supplier", "Invoices", "POs", "Straight to queue", "Smallest PO (NOK)", "Median PO (NOK)",
        "Largest PO (NOK)", "Largest / smallest PO", "Segment", "Main clerk", "Main clerk share",
        "Late > 10 days (share of posted)", "Missed 2% discount (NOK)"]
header(ws, 4, cols)
r0 = 5
for i, (sup, r) in enumerate(lst.iterrows()):
    row = r0 + i
    vals = [sup, int(r.invoices), int(r.pos), r.direct, r.min_po, r.median_po, r.max_po, None, r.segment,
            r.main_clerk, r.main_clerk_share, r.late_share, r.missed_discount]
    for c, v in enumerate(vals, 1):
        ws.cell(row, c, v).font = Font(name=F)
    ws.cell(row, 8, f"=IF(E{row}>0,G{row}/E{row},\"-\")").font = Font(name=F)
last = r0 + len(lst) - 1
tot = last + 1
ws.cell(tot, 1, "Total / weighted").font = Font(name=F, bold=True)
ws.cell(tot, 2, f"=SUM(B{r0}:B{last})")
ws.cell(tot, 3, f"=SUM(C{r0}:C{last})")
ws.cell(tot, 4, f"=SUMPRODUCT(B{r0}:B{last},D{r0}:D{last})/B{tot}")
ws.cell(tot, 13, f"=SUM(M{r0}:M{last})")
for c in range(1, len(cols) + 1):
    cell = ws.cell(tot, c)
    cell.font = Font(name=F, bold=True)
    cell.fill = T_FILL
    cell.border = Border(top=thin)
for row in range(r0, tot + 1):
    ws.cell(row, 2).number_format = "#,##0"
    ws.cell(row, 3).number_format = "#,##0"
    ws.cell(row, 4).number_format = PCT
    for c in (5, 6, 7, 13):
        ws.cell(row, c).number_format = NOK
    ws.cell(row, 8).number_format = '#,##0"x"'
    ws.cell(row, 11).number_format = PCT
    ws.cell(row, 12).number_format = PCT
note = tot + 2
ws.cell(note, 1, f"Share of all ZNB1 invoices: ").font = Font(name=F, color="5F6672")
ws.cell(note, 2, f"=B{tot}/{len(t)}").number_format = PCT
ws.cell(note, 3, f"(of {len(t):,} ZNB1 invoices)").font = Font(name=F, color="5F6672")
ws.cell(note + 1, 1, f"Segment: 'Large POs' = median PO ≥ {hv_cut:,.0f} NOK (top quarter of ZNB1 POs).").font = \
    Font(name=F, color="5F6672")
widths = [14, 10, 8, 12, 15, 15, 16, 13, 12, 12, 12, 15, 16]
for c, w in enumerate(widths, 1):
    ws.column_dimensions[get_column_letter(c)].width = w
ws.freeze_panes = "B5"
ws.auto_filter.ref = f"A4:{get_column_letter(len(cols))}{last}"

ws2 = wb.create_sheet("PO value evidence")
title(ws2, "Does the PO amount decide the routing?",
      "Share of ZNB1 invoices sent straight to the manual queue, by PO value.")
ws2["A4"] = "A. All ZNB1 invoices, by PO-value tenth"
ws2["A4"].font = Font(name=F, bold=True)
header(ws2, 5, ["Tenth", "Invoices", "Smallest PO (NOK)", "Largest PO (NOK)", "Straight to queue"])
for i, (d, r) in enumerate(dec.iterrows()):
    row = 6 + i
    for c, v in enumerate([int(d), int(r.invoices), r.min_po, r.max_po, r.direct], 1):
        ws2.cell(row, c, v).font = Font(name=F)
    ws2.cell(row, 3).number_format = NOK
    ws2.cell(row, 4).number_format = NOK
    ws2.cell(row, 5).number_format = PCT
b0 = 6 + len(dec) + 2
ws2.cell(b0, 1, f"B. Within the {len(mixed)} 'mixed' suppliers (≥ 20 invoices, 5–95% straight to queue), by each "
                "supplier's own PO size").font = Font(name=F, bold=True)
header(ws2, b0 + 1, ["PO size within the supplier", "Invoices", "Straight to queue"])
for i, (q, r) in enumerate(qt.iterrows()):
    row = b0 + 2 + i
    for c, v in enumerate([str(q), int(r.invoices), r.direct], 1):
        ws2.cell(row, c, v).font = Font(name=F)
    ws2.cell(row, 3).number_format = PCT
c0 = b0 + 2 + len(qt) + 1
for i, line in enumerate([
    "Reading: across all of ZNB1, larger POs go to the manual queue more often (A), but within the same supplier the "
    "amount barely matters (B).",
    "So the routing is mainly decided by the supplier; the amount effect in A comes from which suppliers have the "
    "large POs. A limit for the very largest POs (> 8M NOK) cannot be ruled out.",
]):
    ws2.cell(c0 + i, 1, line).font = Font(name=F, color="5F6672")
for c, w in enumerate([30, 10, 18, 18, 16], 1):
    ws2.column_dimensions[get_column_letter(c)].width = w

# --- PO-value quantile sheets (one value per PO) ----------------------------------------------
po_tab = t.reset_index().drop_duplicates("PO_ID")[["supplier", "PO_ID", "po_value"]]
groups = [(f"All {len(lst)} suppliers above (pooled)", po_tab[po_tab.supplier.isin(lst.index)].po_value),
          ("Other ZNB1 suppliers", po_tab[~po_tab.supplier.isin(lst.index)].po_value),
          ("All ZNB1", po_tab.po_value)]


def quantile_sheet(name, qs, position, reading, extra_iqr=False):
    wq = wb.create_sheet(name, position)
    step = qs[1] - qs[0]
    title(wq, f"PO value quantiles in {step:g} steps",
          "One value per PO (Case_Invoice_Value_NOK). 0 = smallest PO, 0.5 = median, 1 = largest. "
          "Linear interpolation between POs. Computed from the event log.")
    labels = [f"{q:g}" + (" (min)" if q == 0 else " (median)" if q == 0.5 else " (max)" if q == 1 else "")
              for q in qs]
    cols = ["Supplier", "POs"] + labels + (["Interquartile range (0.75 − 0.25)"] if extra_iqr else [])
    header(wq, 4, cols)
    rows = [(sup, po_tab[po_tab.supplier == sup].po_value) for sup in lst.index]
    q25, q75 = (qs.index(0.25) + 3, qs.index(0.75) + 3) if extra_iqr else (None, None)
    r = 5
    for block, bold in ((rows, False), (groups, True)):
        for label, v in block:
            vals = [label, len(v), *v.quantile(qs).values]
            for c, val in enumerate(vals, 1):
                cell = wq.cell(r, c, val)
                cell.font = Font(name=F, bold=bold)
                if bold:
                    cell.fill = T_FILL
                if c > 2:
                    cell.number_format = NOK
            if extra_iqr:
                c = len(vals) + 1
                cell = wq.cell(r, c, f"={get_column_letter(q75)}{r}-{get_column_letter(q25)}{r}")
                cell.font = Font(name=F, bold=bold)
                cell.number_format = NOK
                if bold:
                    cell.fill = T_FILL
            r += 1
        r += 1
    wq.cell(r, 1, reading).font = Font(name=F, color="5F6672")
    wq.column_dimensions["A"].width = 34
    wq.column_dimensions["B"].width = 8
    for c in range(3, len(cols) + 1):
        wq.column_dimensions[get_column_letter(c)].width = 15
    wq.freeze_panes = "C5"


quantile_sheet("PO value quantiles 0.1", [round(q / 10, 1) for q in range(11)], 1,
               "Reading: each always-manual supplier spans small and very large POs (compare the 0.1 and 0.9 "
               "columns), yet all its invoices take the same manual route.")
quantile_sheet("PO value quartiles", [0, 0.25, 0.5, 0.75, 1], 2,
               "Reading: the middle half of each supplier's POs lies between the 0.25 and 0.75 columns; the "
               "interquartile range shows how wide that middle half is.", extra_iqr=True)

ws3 = wb.create_sheet("Notes")
title(ws3, "Definitions and assumptions", "Source: INN354 AP event log (POs started 1 Jul – 30 Nov 2025, "
                                          "followed to 3 Mar 2026); notes/znb1_always_manual_suppliers.py")
notes = [
    ("Invoice", "One SAP_PO_No within a PO_ID (a PO can have several invoices)."),
    ("Straight to queue", "Initial validation routes the invoice to the manual control queue "
                          "(SAP_Next_Processing_Type = CTRL1), skipping the goods-receipt check."),
    ("Always manual", f"Supplier with ≥ {MIN_POS} POs and ≥ {MIN_DIRECT:.0%} of its invoices straight to queue."),
    ("PO value", "Case_Invoice_Value_NOK, one value per PO. Smallest / median / largest are over the supplier's POs."),
    ("Main clerk", "User who made most of the supplier's manual postings; share = their part of them."),
    ("Late", "Posted more than 10 days after clock start. ZNB1 clock starts at goods receipt (WAIT_GR step) where "
             "there is one, else at the invoice's first event. Share of posted invoices."),
    ("Missed 2% discount", "2% of the value of late invoices. Invoice value = PO value split evenly over the PO's "
                           "posted invoices; 2/10 terms assumed for all. A potential, not a proven loss."),
    ("Interpretation", "Routing that ignores the amount points to a supplier- or PO-level setting (e.g. a manual-check "
                       "flag, or POs without goods receipt). To be confirmed with the business."),
    ("Never-posted invoices", "Invoices never posted (deleted, sent to deletion at validation, moved to a new invoice "
                              "number, or still open on 3 Mar 2026). "
                              + ("Every PO with at least one such invoice is left out entirely, on all sheets. "
                                 if EXCLUDE_POS_WITH_UNPOSTED else "They carry no value; their PO stays in. ")
                              + SCOPE),
    ("Full assumptions", "docs/assumptions.html"),
]
header(ws3, 4, ["Term", "Definition"])
for i, (a, b) in enumerate(notes):
    ws3.cell(5 + i, 1, a).font = Font(name=F, bold=True)
    ws3.cell(5 + i, 2, b).font = Font(name=F)
    ws3.cell(5 + i, 2).alignment = Alignment(wrap_text=True, vertical="top")
    ws3.cell(5 + i, 1).alignment = Alignment(vertical="top")
ws3.column_dimensions["A"].width = 20
ws3.column_dimensions["B"].width = 110

wb.save(OUT)
print(f"wrote {OUT} with {len(lst)} suppliers, {int(lst.invoices.sum())} invoices, "
      f"missed discount {lst.missed_discount.sum():,.0f} NOK")
