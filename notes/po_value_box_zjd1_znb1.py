"""PO-value boxplots for ZJD1 and ZNB1 on a log scale (companion to po_value_dist_zjd1_znb1.py).

Value counted once per PO (Case_Invoice_Value_NOK). The box statistics are computed on log10(value), so box and
whiskers (1.5 x IQR) are symmetric on the log axis the chart is drawn on; POs with value 0 can't be logged and are
left out. -> notes/figures/po_value_box_zjd1_znb1.png
Run from the repo root:  conda run -n datasci python -W ignore notes/po_value_box_zjd1_znb1.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

NOTES = Path(__file__).resolve().parent
sys.path.insert(0, str(NOTES.parent))
from common import PO_TYPES, load_cases  # noqa: E402

PNG = NOTES / "figures" / "po_value_box_zjd1_znb1.png"
TYPES = ["ZJD1", "ZNB1"]

BLUE, BLUE_LIGHT = "#2a78d6", "#cfe1f7"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"


def fmt_nok(v):
    return f"{v / 1e6:,.2f}M" if v >= 1e6 else (f"{v / 1e3:,.0f}k" if v >= 1e3 else f"{v:,.0f}")


cases = load_cases()
cases = cases[cases.po_type.isin(TYPES)]
zeros = cases[cases.invoice_value_nok <= 0].groupby("po_type").size().reindex(TYPES, fill_value=0)
vals = {t: cases.loc[(cases.po_type == t) & (cases.invoice_value_nok > 0), "invoice_value_nok"].to_numpy()
        for t in TYPES}

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"]})
fig, ax = plt.subplots(figsize=(11, 4.6), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

pos = [1, 0]  # ZJD1 on top
logs = [np.log10(vals[t]) for t in TYPES]
bp = ax.boxplot(logs, positions=pos, vert=False, widths=0.5, patch_artist=True, whis=1.5,
                medianprops=dict(color=INK, lw=2),
                boxprops=dict(facecolor=BLUE_LIGHT, edgecolor=BLUE, lw=1.4),
                whiskerprops=dict(color=BLUE, lw=1.4), capprops=dict(color=BLUE, lw=1.4),
                flierprops=dict(marker="o", markersize=3, markerfacecolor=BLUE, markeredgecolor="none", alpha=0.35),
                zorder=3)
# Mean as a marker
for p, t in zip(pos, TYPES):
    m = vals[t].mean()
    ax.plot(np.log10(m), p, marker="D", color=INK, markersize=6, zorder=4)
    q1, med, q3 = np.percentile(vals[t], [25, 50, 75])
    ax.annotate(f"median {fmt_nok(med)}", (np.log10(med), p + 0.25), xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontsize=9, color=INK)
    ax.annotate(f"mean {fmt_nok(m)}", (np.log10(m), p - 0.25), xytext=(0, -4), textcoords="offset points",
                ha="center", va="top", fontsize=9, color=INK)
    ax.text(1.01, p, f"IQR {fmt_nok(q1)}–{fmt_nok(q3)}\nmax {fmt_nok(vals[t].max())}", transform=ax.get_yaxis_transform(),
            va="center", ha="left", fontsize=9, color=INK2)

ax.set_yticks(pos)
ax.set_yticklabels([f"{t} · {PO_TYPES[t]}\n{len(vals[t]):,} POs" for t in TYPES], color=INK, fontsize=10)
ax.set_ylim(-0.6, 1.6)
lo, top = np.floor(min(l.min() for l in logs)), max(l.max() for l in logs)
ax.set_xlim(lo - 0.1, top + 0.25)
ax.set_xticks(np.arange(lo, np.floor(top) + 1))
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: fmt_nok(10 ** v).replace(".00", "")))
ax.set_xlabel("PO value (NOK, log scale)", color=INK2, fontsize=10)
ax.grid(axis="x", color=GRID, zorder=0)
ax.tick_params(axis="both", length=0, colors=INK2)
for sp in ax.spines.values():
    sp.set_visible(False)
ax.set_title("PO-value distribution, ZJD1 vs ZNB1 (boxplot)", loc="left", color=INK, fontsize=13,
             fontweight="bold", pad=12)
fig.text(0.01, 0.01, "Box = middle 50% of POs, line = median, diamond = mean, whiskers = 1.5 × IQR on the log scale, "
         "dots = POs beyond. Value counted once per PO; POs with value 0 left out "
         f"(ZJD1: {zeros['ZJD1']}, ZNB1: {zeros['ZNB1']}).", color=INK2, fontsize=8)
fig.tight_layout(rect=(0, 0.04, 0.88, 1))
fig.savefig(PNG, facecolor=SURFACE)
print(PNG)
