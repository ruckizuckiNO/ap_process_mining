"""Benchmark for the ZNB1 manual queue: what if ZNB1 invoices were posted as fast as ZJD1 posts manually after goods receipt?

Reference: ZJD1 invoices with a goods-receipt step (WAIT_GR) that were then posted manually; time = goods receipt -> first
posting. Target: ZNB1 invoices routed straight to CTRL1 at their first validation, posted; time = arrival in AP -> first
posting (no goods-receipt step on this route; assumes suppliers invoice after delivery). Also the subset of suppliers whose
invoices go 100% to CTRL1, posted straight from the queue.
Simulation: each ZNB1 invoice gets a time drawn from the ZJD1 reference (bootstrap of actual values, B runs); its own
invoice value is kept (PO value split evenly over the PO's posted invoices). 95% intervals = 2.5th-97.5th percentile of runs.

Outputs:
  notes/figures/znb1_zjd1_benchmark_slide.png   16:9 overview slide (chart + table + takeaways)
  notes/znb1_zjd1_benchmark.xlsx                percentiles and simulation results
Run from the repo root:  PYTHONPATH=. conda run -n datasci python -W ignore notes/znb1_zjd1_benchmark.py
"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from common import POSTING_ACTIVITIES, SLA_HARD_DAYS, SLA_SOFT_DAYS, load_cases, load_events, load_invoices

import argparse
from common import use_invoice_proxy_labels

use_invoice_proxy_labels()  # chart text: invoice -> invoice proxy

NOTES = Path(__file__).resolve().parent
_ap = argparse.ArgumentParser()
_ap.add_argument("--max-po-value", type=float, default=None,
                 help="keep only invoices on POs with Case_Invoice_Value_NOK below this (NOK); applies to reference and targets")
MAXV = _ap.parse_args().max_po_value
TAG = "" if MAXV is None else f"_po_lt_{MAXV / 1e6:g}m".replace(".", "p")
FILTER_TXT = "" if MAXV is None else f"PO value < {MAXV / 1e6:g}M NOK"
K = ["PO_ID", "SAP_PO_No"]
B, SEED, DISCOUNT = 5000, 0, 0.02
QS = [0.5, 0.75, 0.8, 0.9, 0.95, 0.99]

# ---------------------------------------------------------------- data
ev = load_events()
ev = ev[ev.SAP_PO_Type_Code.isin(["ZJD1", "ZNB1"])].sort_values(["PO_ID", "ts", "Event_ID"])
ev["nxt"] = ev.groupby(K).Activity_ID.shift(-1)
fv = ev[ev.Activity_ID == "Initial validation"].groupby(K).agg(route=("SAP_Next_Processing_Type", "first"),
                                                               nxt=("nxt", "first"), supplier=("Supplier_ID", "first"))
post = ev[ev.Activity_ID.isin(POSTING_ACTIVITIES)].groupby(K).Activity_ID.first().rename("post_act")
inv = load_invoices()
inv = inv[inv.po_type.isin(["ZJD1", "ZNB1"])].set_index(K).join(fv).join(post)
cases = load_cases().set_index("PO_ID")
inv["n_posted"] = inv.groupby(level="PO_ID").posted.transform("sum")
inv["value"] = np.where(inv.posted, cases.invoice_value_nok.reindex(inv.index.get_level_values(0)).values
                        / inv.n_posted.where(inv.n_posted > 0), 0)
p = inv[inv.posted].copy()
p["gr_post"] = (p.first_post - p.goods_receipt).dt.total_seconds() / 86400
p["arr_post"] = (p.first_post - p.start).dt.total_seconds() / 86400

p["po_value"] = cases.invoice_value_nok.reindex(p.index.get_level_values("PO_ID")).values
if MAXV is not None:
    p = p[p.po_value < MAXV]
ref = p[(p.po_type == "ZJD1") & p.goods_receipt.notna() & (p.post_act == "Posted manually")].gr_post.to_numpy()
z_all = p[(p.po_type == "ZNB1") & (p.route == "CTRL1")]
zv = p[(p.po_type == "ZNB1") & p.route.notna()]
sh = inv[(inv.po_type == "ZNB1") & inv.route.notna()].groupby("supplier").route.agg(lambda x: (x == "CTRL1").mean())
z_100 = z_all[z_all.supplier.isin(sh.index[sh == 1]) & z_all.nxt.isin(POSTING_ACTIVITIES)]
GROUPS = {"ZNB1 straight to CTRL1": z_all, "ZNB1 100%-CTRL1 suppliers": z_100}

# ---------------------------------------------------------------- simulation
rng = np.random.default_rng(SEED)
SIM, ACT = {}, {}
for name, g in GROUPS.items():
    vals, n = g.value.to_numpy(), len(g)
    draws = ref[rng.integers(0, len(ref), size=(B, n))]
    l10, l30 = draws > SLA_SOFT_DAYS, draws > SLA_HARD_DAYS
    SIM[name] = {"median": np.median(draws, axis=1), "p90": np.quantile(draws, 0.9, axis=1), "s10": l10.mean(1),
                 "s30": l30.mean(1), "missed": DISCOUNT * (l10 * vals).sum(1)}
    a = g.arr_post
    ACT[name] = {"n": n, "median": a.median(), "p90": a.quantile(0.9), "s10": (a > SLA_SOFT_DAYS).mean(),
                 "s30": (a > SLA_HARD_DAYS).mean(), "missed": DISCOUNT * g.value[a > SLA_SOFT_DAYS].sum(),
                 "amount": g.value.sum()}
REF = {"n": len(ref), "median": np.median(ref), "p90": np.quantile(ref, 0.9), "s10": (ref > SLA_SOFT_DAYS).mean(),
       "s30": (ref > SLA_HARD_DAYS).mean()}


def mid(a):
    return float(np.percentile(a, 50))


def ci(a):
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


M = "ZNB1 straight to CTRL1"
a, s = ACT[M], SIM[M]
saving = a["missed"] - mid(s["missed"])

# ---------------------------------------------------------------- slide (16:9)
INK, MUTED, GRID, SURF = "#16201c", "#5a6a63", "#d9e1dd", "#ffffff"
C_REF, C_ZNB = "#2a78d6", "#eb6834"          # categorical slots 1 and 2
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": GRID,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": MUTED})
fig = plt.figure(figsize=(13.333, 7.5), facecolor=SURF)
fig.text(0.04, 0.93, "ZNB1 manual queue vs ZJD1 benchmark" + (f" · {FILTER_TXT}" if MAXV else ""),
         fontsize=24 if MAXV is None else 21, fontweight="bold", color=INK)
fig.text(0.04, 0.855, f"If ZNB1 invoices sent straight to the manual queue (CTRL1) were posted as fast as ZJD1 posts manually after\n"
         f"goods receipt, invoices over 10 days would fall from {a['s10']:.0%} to about {mid(s['s10']):.0%}.",
         fontsize=13, color=MUTED, linespacing=1.4)

# Left: cumulative share posted by day
ax = fig.add_axes([0.06, 0.2, 0.47, 0.6])
x = np.linspace(0, 30, 601)
cdf_ref = np.searchsorted(np.sort(ref), x, side="right") / len(ref)
za = np.sort(z_all.arr_post.to_numpy())
cdf_z = np.searchsorted(za, x, side="right") / len(za)
ax.plot(x, cdf_ref * 100, color=C_REF, lw=2.2)
ax.plot(x, cdf_z * 100, color=C_ZNB, lw=2.2)
ax.axvline(SLA_SOFT_DAYS, color=MUTED, lw=1, ls=(0, (4, 3)))
ax.text(SLA_SOFT_DAYS + 0.4, 4, "10-day discount limit", color=MUTED, fontsize=10)
r10, z10 = 1 - REF["s10"], 1 - a["s10"]
for val, col in [(r10, C_REF), (z10, C_ZNB)]:
    ax.plot([SLA_SOFT_DAYS], [val * 100], "o", ms=8, color=col, mec=SURF, mew=2, zorder=5)
    ax.text(SLA_SOFT_DAYS - 0.5, val * 100 + 2.5, f"{val:.0%}", color=INK, fontsize=11, ha="right", fontweight="bold")
ax.text(15, 80, f"ZJD1, manual after goods receipt\n(reference, {REF['n']:,} invoices)", color=C_REF, fontsize=11, fontweight="bold")
ax.text(13.5, 52, f"ZNB1 straight to CTRL1\n(actual, {a['n']:,} invoices)", color=C_ZNB, fontsize=11, fontweight="bold")
ax.set_xlim(0, 30)
ax.set_ylim(0, 102)
ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
ax.set_yticks([0, 25, 50, 75, 100])
ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
ax.set_xlabel("Days to posting")
ax.set_title("Share of invoices posted within … days", loc="left", fontsize=13, color=INK, fontweight="bold", pad=10)
ax.grid(color=GRID, lw=0.8)
ax.set_axisbelow(True)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)

# Right: KPI table
lo, hi = ci(s["s10"])
mlo, mhi = ci(s["missed"])
rows = [
    ("", "ZNB1 actual", "ZNB1 simulated*"),
    ("Median days to posting", f"{a['median']:.1f}", f"{mid(s['median']):.1f}"),
    ("90% posted within (days)", f"{a['p90']:.1f}", f"{mid(s['p90']):.1f}"),
    ("Invoices > 10 days", f"{a['s10']:.1%}", f"{mid(s['s10']):.1%}  ({lo:.1%}–{hi:.1%})"),
    ("Invoices > 30 days", f"{a['s30']:.1%}", f"{mid(s['s30']):.1%}"),
    ("Missed 2% discount", f"{a['missed'] / 1e6:.1f}M NOK", f"{mid(s['missed']) / 1e6:.1f}M  ({mlo / 1e6:.1f}–{mhi / 1e6:.1f}M)"),
]
tx, ty, rh = 0.585, 0.78, 0.062
fig.text(tx, ty + 0.02, f"{M} · {a['n']:,} invoices", fontsize=13, fontweight="bold", color=INK)
for i, (l, c1, c2) in enumerate(rows):
    yy = ty - (i + 0.6) * rh
    w = "bold" if i == 0 else "normal"
    col = MUTED if i == 0 else INK
    fig.text(tx, yy, l, fontsize=12, color=col, fontweight=w)
    fig.text(tx + 0.215, yy, c1, fontsize=12, color=C_ZNB if i else MUTED, fontweight="bold" if i else w, ha="right")
    fig.text(tx + 0.375, yy, c2, fontsize=12, color=C_REF if i else MUTED, fontweight="bold" if i else w, ha="right")
    if i:
        fig.add_artist(plt.Line2D([tx, tx + 0.375], [yy + rh * 0.62] * 2, color=GRID, lw=0.8))

# Takeaway box
bx, by, bw, bh = 0.585, 0.2, 0.375, 0.16
fig.add_artist(plt.Rectangle((bx, by), bw, bh, transform=fig.transFigure, facecolor="#e3efea", edgecolor="none"))
fig.text(bx + 0.015, by + bh - 0.04, f"Potential: about {saving / 1e6:.1f}M NOK less missed discount",
         fontsize=12.5, fontweight="bold", color="#1f6f5c")
s100, a100 = SIM["ZNB1 100%-CTRL1 suppliers"], ACT["ZNB1 100%-CTRL1 suppliers"]
fig.text(bx + 0.015, by + 0.025,
         f"The {z_100.supplier.nunique()} always-CTRL1 suppliers alone: {a100['s10']:.0%} → {mid(s100['s10']):.0%} late,\n"
         f"{a100['missed'] / 1e6:.1f}M → {mid(s100['missed']) / 1e6:.1f}M NOK. Same clerks post both types;\n"
         f"the gap is the one-week routine in the ZNB1 queue.", fontsize=10.5, color=INK, linespacing=1.4)

fig.text(0.04, 0.03,
         f"* Each ZNB1 invoice gets a time drawn at random from the ZJD1 reference (goods receipt → manual posting); {B:,} runs, "
         f"median with 95% interval; own invoice values kept.\nZNB1 clock: arrival in AP → first posting (assumes suppliers invoice "
         f"after delivery). Invoice value = PO value split evenly over the PO's posted invoices. Missed discount = 2% of invoices\n"
         f"posted after > 10 days. Capacity is not modelled: the ZJD1 reference queue is smaller ({REF['n']:,} vs {a['n']:,} invoices).",
         fontsize=8.5, color=MUTED, linespacing=1.4)
png = NOTES / "figures" / f"znb1_zjd1_benchmark{TAG}_slide.png"
fig.savefig(png, dpi=200, facecolor=SURF)
plt.close(fig)

# ---------------------------------------------------------------- workbook
F = "Arial"
H_FILL, H_FONT, BOLD, NORM = PatternFill("solid", start_color="1F6F5C"), Font(name=F, bold=True, color="FFFFFF"), \
    Font(name=F, bold=True), Font(name=F)
wb = Workbook()


def header(ws, cols, widths):
    for j, (h, w) in enumerate(zip(cols, widths), 1):
        c = ws.cell(1, j, h)
        c.font, c.fill, c.alignment = H_FONT, H_FILL, Alignment(wrap_text=True, vertical="center")
        ws.column_dimensions[get_column_letter(j)].width = w
    ws.row_dimensions[1].height = 32
    ws.freeze_panes = "B2"


ws = wb.active
ws.title = "Percentiles"
header(ws, ["Group", "Time measured", "Invoices"] + [f"P{int(q * 100)} (days)" for q in QS] + ["> 10 d", "> 30 d"],
       [40, 34, 10] + [11] * len(QS) + [9, 9])
dists = [("ZJD1 manual after goods receipt (reference)", "goods receipt → first posting", pd.Series(ref))] + \
        [(n, "arrival in AP → first posting", g.arr_post) for n, g in GROUPS.items()]
for i, (n, t, d) in enumerate(dists, 2):
    ws.cell(i, 1, n), ws.cell(i, 2, t), ws.cell(i, 3, len(d))
    for j, q in enumerate(QS, 4):
        ws.cell(i, j, round(float(d.quantile(q)), 2)).number_format = "0.00"
    ws.cell(i, 4 + len(QS), float((d > SLA_SOFT_DAYS).mean())).number_format = "0.0%"
    ws.cell(i, 5 + len(QS), float((d > SLA_HARD_DAYS).mean())).number_format = "0.0%"
    for j in range(1, 6 + len(QS)):
        ws.cell(i, j).font = NORM

ws2 = wb.create_sheet("Simulation")
header(ws2, ["Group", "Measure", "Actual", "Simulated median", "95% interval low", "95% interval high", "Difference (actual − simulated)"],
       [30, 26, 14, 16, 16, 16, 18])
r = 2
fmt = {"median": "0.00", "p90": "0.00", "s10": "0.0%", "s30": "0.0%", "missed": "#,##0"}
lab = {"median": "Median days", "p90": "90th percentile days", "s10": "Share > 10 days", "s30": "Share > 30 days",
       "missed": "Missed 2% discount (NOK)"}
for n in GROUPS:
    for m in ["median", "p90", "s10", "s30", "missed"]:
        lo_, hi_ = ci(SIM[n][m])
        vals = [n, lab[m], float(ACT[n][m]), mid(SIM[n][m]), lo_, hi_]
        for j, v in enumerate(vals, 1):
            c = ws2.cell(r, j, v)
            c.font = NORM
            if j >= 3:
                c.number_format = fmt[m]
        ws2.cell(r, 7, f"=C{r}-D{r}").number_format = fmt[m]
        ws2.cell(r, 7).font = NORM
        r += 1
    ws2.cell(r, 1, n).font = NORM
    ws2.cell(r, 2, "Invoices / invoice amount (NOK)").font = NORM
    ws2.cell(r, 3, ACT[n]["n"]).number_format = "#,##0"
    ws2.cell(r, 4, float(ACT[n]["amount"])).number_format = "#,##0"
    r += 2

ws3 = wb.create_sheet("Notes")
notes = [("Reference", "ZJD1 invoices with a goods-receipt step (WAIT_GR) that were posted manually; goods receipt → first posting."),
         ("Target groups", "ZNB1 invoices routed straight to CTRL1 at their first validation (posted); and the subset from suppliers with "
                           "100% CTRL1, posted straight from the queue. Clock: arrival in AP → first posting."),
         ("Assumption", "Suppliers invoice after delivery, so arrival in AP is comparable to goods receipt."),
         ("Simulation", f"Bootstrap: each ZNB1 invoice gets a time drawn at random from the ZJD1 reference, {B:,} runs, seed {SEED}. "
                        "Simulated median = median over runs; 95% interval = 2.5th–97.5th percentile over runs."),
         ("Invoice value", "PO value (Case_Invoice_Value_NOK) split evenly over the PO's posted invoices."),
         ("Missed discount", "2% of the value of invoices posted after more than 10 days (2/10 terms assumed for every invoice)."),
         ("Not modelled", "Capacity: the ZJD1 manual queue is smaller than the ZNB1 one; equal speed at ZNB1 volume is an assumption."),
         ("Values", "All numbers are static outputs of notes/znb1_zjd1_benchmark.py (simulation results can't be recomputed in Excel); "
                    "only the difference column is a formula.")]
for i, (a_, b_) in enumerate(notes, 1):
    ws3.cell(i, 1, a_).font = BOLD
    c = ws3.cell(i, 2, b_)
    c.font, c.alignment = NORM, Alignment(wrap_text=True, vertical="top")
ws3.column_dimensions["A"].width = 18
ws3.column_dimensions["B"].width = 120
wb.calculation.fullCalcOnLoad = True
if MAXV is not None:
    ws3.cell(len(notes) + 1, 1, "Filter").font = BOLD
    ws3.cell(len(notes) + 1, 2, f"{FILTER_TXT}: only invoices on POs with Case_Invoice_Value_NOK below {MAXV:,.0f} NOK, "
                                "applied to the ZJD1 reference and to both ZNB1 groups. The supplier group (100% CTRL1) is "
                                "defined on all ZNB1 invoices, before the filter.").font = NORM
    ws3.cell(len(notes) + 1, 2).alignment = Alignment(wrap_text=True, vertical="top")
xlsx = NOTES / f"znb1_zjd1_benchmark{TAG}.xlsx"
wb.save(xlsx)

print("wrote", png, "and", xlsx)
for n in GROUPS:
    print(n, {k: round(float(v), 4) for k, v in ACT[n].items()},
          {k: (round(mid(v), 4), tuple(round(x, 4) for x in ci(v))) for k, v in SIM[n].items()})
print("REF", {k: round(float(v), 4) for k, v in REF.items()}, "saving", round(saving / 1e6, 2))
