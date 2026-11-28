"""Total PO value per SAP PO type (ZJD1, ZNB1, ...) as a horizontal bar chart.

Value is counted once per PO (Case_Invoice_Value_NOK = PO total), same rule as the overview table in
06_report_znb1.py. -> notes/figures/po_value_by_type.png + notes/po_value_by_type.csv.
Run from the repo root:  conda run -n datasci python -W ignore notes/po_value_by_type.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt

NOTES = Path(__file__).resolve().parent
sys.path.insert(0, str(NOTES.parent))
from common import PO_TYPES, load_cases  # noqa: E402
from common import use_invoice_proxy_labels

use_invoice_proxy_labels()  # chart text: invoice -> invoice proxy

PNG = NOTES / "figures" / "po_value_by_type.png"
CSV = NOTES / "po_value_by_type.csv"

BLUE = "#2a78d6"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"

cases = load_cases()
val = cases.groupby("po_type").invoice_value_nok.agg(pos="size", total_nok="sum")
val["share_pct"] = 100 * val.total_nok / val.total_nok.sum()
val["name"] = val.index.map(PO_TYPES)
val = val.sort_values("total_nok", ascending=False)
val.to_csv(CSV)

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"]})
fig, ax = plt.subplots(figsize=(11, 6), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

plot = val.iloc[::-1]  # largest on top
y = range(len(plot))
ax.barh(y, plot.total_nok / 1e6, height=0.7, color=BLUE, zorder=2)
ax.set_yticks(list(y))
ax.set_yticklabels([f"{c}  ·  {n}" for c, n in zip(plot.index, plot.name)], color=INK, fontsize=10)
xmax = plot.total_nok.max() / 1e6
for yi, (tot, sh, n) in zip(y, zip(plot.total_nok, plot.share_pct, plot.pos)):
    ax.text(tot / 1e6 + xmax * 0.01, yi, f"{tot / 1e6:,.1f}M NOK  ({sh:.1f}%, {n:,} POs)" if sh >= 0.05 else f"{tot / 1e6:,.1f}M NOK  (<0.1%, {n:,} POs)",
            va="center", ha="left", fontsize=9, color=INK2)
ax.set_xlim(0, xmax * 1.32)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
ax.set_xlabel("Total PO value (million NOK)", color=INK2, fontsize=10)
ax.grid(axis="x", color=GRID, zorder=0)
ax.tick_params(axis="both", length=0, colors=INK2)
for s in ax.spines.values():
    s.set_visible(False)
total = val.total_nok.sum()
ax.set_title(f"Total PO value per SAP PO type  (all types: {total / 1e9:,.2f}B NOK, {val.pos.sum():,} POs)",
             loc="left", color=INK, fontsize=13, fontweight="bold", pad=12)
fig.text(0.01, 0.01, "Value counted once per PO (Case_Invoice_Value_NOK holds the PO total on every row).",
         color=INK2, fontsize=8)
fig.tight_layout(rect=(0, 0.03, 1, 1))
PNG.parent.mkdir(exist_ok=True)
fig.savefig(PNG, facecolor=SURFACE)
print(val.to_string())
print(PNG)
