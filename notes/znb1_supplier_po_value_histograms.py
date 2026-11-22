"""PO-value histograms for three ZNB1 always-manual suppliers (Supplier381, Supplier361, Supplier378).

One value per PO (Case_Invoice_Value_NOK). Scope as in znb1_always_manual_suppliers.xlsx: ZNB1 POs without
never-posted invoices. PO values span 1k-150M NOK, so the bins are log-spaced (1-2.5-5 steps). Each bar is split into
POs where every invoice was posted within 10 days and POs with at least one invoice later than that (GR clock).
-> notes/figures/znb1_supplier_po_value_histograms.png (+ one PNG per supplier) and
   notes/znb1_supplier_po_value_histograms.csv
Run from the repo root:  PYTHONPATH=. conda run -n datasci python -W ignore notes/znb1_supplier_po_value_histograms.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import SLA_SOFT_DAYS, load_cases, load_invoices

NOTES = Path(__file__).resolve().parent
FIG = NOTES / "figures"
SUPPLIERS = ["Supplier381", "Supplier361", "Supplier378"]
EDGES = np.array([1e3, 2.5e3, 5e3, 1e4, 2.5e4, 5e4, 1e5, 2.5e5, 5e5, 1e6, 2.5e6, 5e6, 1e7, 2.5e7, 5e7, 1e8, 2.5e8])
THRESHOLD = 1.5e6  # PO-value triage line discussed for ZNB1 (2026-11-19)
SHOW_TRIAGE = True  # off for now (2026-11-22); True draws the dashed line and the ">=1.5M" counts


BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"


def fmt(v):
    return f"{v / 1e6:g}M" if v >= 1e6 else f"{v / 1e3:g}k"


inv = load_invoices()
inv = inv[inv.po_type == "ZNB1"]
bad_pos = set(inv.loc[~inv.posted.astype(bool), "PO_ID"])
inv = inv[~inv.PO_ID.isin(bad_pos)]
inv["late"] = inv.time_to_post_days > SLA_SOFT_DAYS
late_po = inv.groupby("PO_ID").late.any()

cases = load_cases()
po = cases[(cases.po_type == "ZNB1") & cases.supplier.isin(SUPPLIERS) & ~cases.PO_ID.isin(bad_pos)].copy()
po["late"] = po.PO_ID.map(late_po).fillna(False).astype(bool)
assert po.invoice_value_nok.between(EDGES[0], EDGES[-1]).all(), po.invoice_value_nok.describe()
po["bin"] = pd.cut(po.invoice_value_nok, EDGES, right=False)

table = (po.groupby(["supplier", "bin"], observed=False)
         .agg(pos=("PO_ID", "size"), pos_with_late_invoice=("late", "sum"), po_value_nok=("invoice_value_nok", "sum"))
         .reset_index())
table["bin_from_nok"] = table.bin.map(lambda b: b.left)
table["bin_to_nok"] = table.bin.map(lambda b: b.right)
table.drop(columns="bin").to_csv(NOTES / "znb1_supplier_po_value_histograms.csv", index=False)

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"]})
ymax = table.pos.max() + 4


def panel(ax, sup, show_xlabels=True):
    d = po[po.supplier == sup]
    ok = np.histogram(d.loc[~d.late, "invoice_value_nok"], EDGES)[0]
    late = np.histogram(d.loc[d.late, "invoice_value_nok"], EDGES)[0]
    left, width = EDGES[:-1], np.diff(EDGES)
    # Bars drawn on log x: shrink each bar a little inside its bin so neighbours keep a surface gap.
    l2, w2 = left * 1.06, width - left * 0.12
    ax.bar(l2, ok, width=w2, align="edge", color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=2,
           label="All invoices posted ≤10 days")
    ax.bar(l2, late, width=w2, bottom=ok, align="edge", color=ORANGE, edgecolor=SURFACE, linewidth=1.5, zorder=2,
           label="At least one invoice >10 days")
    for x, n in zip(np.sqrt(EDGES[:-1] * EDGES[1:]), ok + late):
        if n:
            ax.text(x, n + 0.4, str(n), ha="center", va="bottom", fontsize=8, color=INK, zorder=4,
                    bbox=dict(boxstyle="square,pad=0.1", facecolor=SURFACE, edgecolor="none"))
    med = d.invoice_value_nok.median()
    ax.axvline(med, color=INK, linewidth=1, linestyle=(0, (1, 2)), zorder=3)
    if SHOW_TRIAGE:
        ax.axvline(THRESHOLD, color=INK2, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    above = (d.invoice_value_nok >= THRESHOLD).sum()
    ax.set_xscale("log")
    ax.set_xlim(EDGES[0], EDGES[-1])
    ax.set_ylim(0, ymax)
    ax.set_xticks(EDGES)
    ax.set_xticklabels([fmt(e) for e in EDGES] if show_xlabels else [], fontsize=8, color=INK2)
    ax.minorticks_off()
    ax.tick_params(axis="both", length=0, colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color(GRID)
    ax.set_facecolor(SURFACE)
    n_late = int(d.late.sum())
    ax.set_title(f"{sup} · {len(d)} POs · {d.invoice_value_nok.sum() / 1e6:.0f}M NOK · "
                 f"{n_late} POs ({n_late / len(d):.0%}) with a late invoice · "
                 f"median {med / 1e6:.2f}M" + (f" · {above} POs ≥1.5M" if SHOW_TRIAGE else ""),
                 loc="left", fontsize=10.5, color=INK, fontweight="bold")


def finish(fig, title, subtitle, ax0, y=(0.985, 0.955, 0.94)):
    fig.patch.set_facecolor(SURFACE)
    fig.suptitle(title, x=0.07, ha="left", y=y[0], fontsize=14, fontweight="bold", color=INK)
    fig.text(0.07, y[1], subtitle, fontsize=9.5, color=INK2, ha="left", va="top")
    h, l = ax0.get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.065, y[2]), ncol=2, frameon=False, fontsize=9,
               labelcolor=INK2, handlelength=1.6)
    fig.text(0.07, 0.008, "Scope: ZNB1 POs without never-posted invoices. PO value = Case_Invoice_Value_NOK (one per PO). "
             "Late = posted more than 10 days after the clock start (goods receipt). Bins are log-spaced.",
             fontsize=7, color=INK2, ha="left")


SUB = ("Number of POs per PO-value bin (NOK, log scale) · dotted = supplier median"
       + (" · dashed = 1.5M NOK triage line" if SHOW_TRIAGE else ""))
fig, axes = plt.subplots(3, 1, figsize=(11, 10.5), dpi=200, sharex=True)
for i, (ax, sup) in enumerate(zip(axes, SUPPLIERS)):
    panel(ax, sup, show_xlabels=True)
    ax.set_ylabel("POs", color=INK2, fontsize=9)
axes[-1].tick_params(labelbottom=True)
axes[-1].set_xlabel("PO value (NOK)", color=INK2, fontsize=10)
finish(fig, "PO-value distribution of three always-manual ZNB1 suppliers", SUB, axes[0])
fig.subplots_adjust(left=0.07, right=0.98, top=0.89, bottom=0.07, hspace=0.3)
FIG.mkdir(exist_ok=True)
fig.savefig(FIG / "znb1_supplier_po_value_histograms.png", facecolor=SURFACE)

for sup in SUPPLIERS:
    fig, ax = plt.subplots(figsize=(11, 4.6), dpi=200)
    panel(ax, sup)
    ax.set_title("", loc="left")
    ax.set_ylabel("POs", color=INK2, fontsize=9)
    ax.set_xlabel("PO value (NOK)", color=INK2, fontsize=10)
    d = po[po.supplier == sup]
    finish(fig, f"{sup}: PO-value distribution (ZNB1)",
           f"{len(d)} POs · {d.invoice_value_nok.sum() / 1e6:.0f}M NOK · {int(d.late.sum())} POs with a late invoice · "
           f"median {d.invoice_value_nok.median() / 1e6:.2f}M (dotted)"
           + (f" · {(d.invoice_value_nok >= THRESHOLD).sum()} POs ≥1.5M (dashed line)" if SHOW_TRIAGE else ""), ax, y=(0.975, 0.895, 0.86))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.78, bottom=0.17)
    fig.savefig(FIG / f"znb1_po_value_histogram_{sup}.png", facecolor=SURFACE)
    plt.close(fig)

print(po.groupby("supplier").agg(pos=("PO_ID", "size"), late=("late", "sum"),
                                 median=("invoice_value_nok", "median"), total=("invoice_value_nok", "sum"),
                                 ge_1_5M=("invoice_value_nok", lambda s: (s >= THRESHOLD).sum())))
