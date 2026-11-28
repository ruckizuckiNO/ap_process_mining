"""Pareto of invoice amount for the ZNB1 suppliers whose invoices go 100% to the manual queue (CTRL1): the "vital few"
suppliers needed to reach 80% of the group's amount (the one that crosses 80% included).

Same group and numbers as notes/figures/znb1_ctrl1_100pct_suppliers_amount.png: read from the Invoices sheet of
notes/znb1_ctrl1_100pct_suppliers.xlsx (built by znb1_ctrl1_100pct_suppliers.py, run that first). That is: suppliers
with 100% of invoices routed to CTRL1 at their first Initial validation; invoices posted straight from the queue;
invoice amount = PO value split evenly over the PO's posted invoices. Main clerk = most frequent poster.
-> notes/figures/znb1_ctrl1_100pct_pareto_amount.png + notes/znb1_ctrl1_100pct_pareto_amount.csv
Run from the repo root:  conda run -n datasci python -W ignore notes/znb1_ctrl1_100pct_pareto_amount.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter
from common import use_invoice_proxy_labels

use_invoice_proxy_labels()  # chart text: invoice -> invoice proxy

NOTES = Path(__file__).resolve().parent
SRC = NOTES / "znb1_ctrl1_100pct_suppliers.xlsx"
CUTOFF = 80.0
PNG = NOTES / "figures" / "znb1_ctrl1_100pct_pareto_amount.png"
CSV = NOTES / "znb1_ctrl1_100pct_pareto_amount.csv"

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"

inv = pd.read_excel(SRC, sheet_name="Invoices")
# Days and missed discount are Excel formulas in the workbook (pandas reads no cached value): recompute them here
# with the same rule: days = arrival in AP -> first posting; missed = 2% of the amount when > 10 days.
days = (pd.to_datetime(inv["First posting"]) - pd.to_datetime(inv["Arrival in AP"])).dt.total_seconds() / 86400
inv["missed"] = 0.02 * inv["Invoice amount (NOK, est.)"] * (days > 10)
g = inv.groupby("Supplier")
df = pd.DataFrame({
    "main_clerk": g["Posted by"].agg(lambda u: u.value_counts().index[0]),
    "pos": g.PO_ID.nunique(), "invoices": g.size(),
    "amount": g["Invoice amount (NOK, est.)"].sum(), "missed": g.missed.sum(),
}).rename_axis("Supplier").reset_index()
df = df.sort_values("amount", ascending=False).reset_index(drop=True)
total = df.amount.sum()
df["share_pct"] = 100 * df.amount / total
df["cum_pct"] = df.share_pct.cumsum()
# Vital few = suppliers needed to reach the cutoff, the one that crosses it included.
TOP_N = int((df.cum_pct < CUTOFF).sum()) + 1
df["vital_few"] = df.index < TOP_N
df.to_csv(CSV, index=False)

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"]})
t = df.head(TOP_N)
fig, ax = plt.subplots(figsize=(12, 6.4), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
x = range(TOP_N)
ax.bar(x, t.share_pct, width=0.6, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=2,
       label="Share of invoice amount (bar)")
ax.plot(x, t.cum_pct, color=ORANGE, linewidth=2, marker="o", markersize=7, markeredgecolor=SURFACE,
        markeredgewidth=1.5, zorder=3, label="Cumulative share (line)")
ax.axhline(CUTOFF, color=INK2, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
ax.text(TOP_N - 0.45, CUTOFF + 1.5, f"{CUTOFF:.0f}%", ha="right", va="bottom", color=INK2, fontsize=9)
for i, r in t.iterrows():
    inside = r.share_pct > 15
    ax.text(i, r.share_pct - 2 if inside else r.share_pct + 1.2,
            f"{r.amount / 1e6:,.0f}M NOK\n{r.share_pct:.0f}%", ha="center", va="top" if inside else "bottom",
            fontsize=9, color="white" if inside else INK, fontweight="bold" if inside else "normal")
    if i > 0:
        ax.text(i, r.cum_pct + 2.5, f"{r.cum_pct:.1f}%" if round(r.cum_pct) == CUTOFF else f"{r.cum_pct:.0f}%", ha="center", va="bottom", fontsize=9, color=INK)
ax.set_ylim(0, 100)
ax.set_xlim(-0.5, TOP_N - 0.5)
ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
ax.set_ylabel("Share of the group's total invoice amount", color=INK2, fontsize=10)
ax.set_xticks(list(x))
ax.set_xticklabels([f"{s}\n{c} · {p:.0f} POs" for s, c, p in zip(t.Supplier, t.main_clerk, t.pos)],
                   fontsize=9, color=INK)
ax.tick_params(axis="both", length=0, colors=INK2)
ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.spines["bottom"].set_visible(True)
ax.spines["bottom"].set_color(GRID)
fig.suptitle(f"{TOP_N} of {len(df)} suppliers hold {t.cum_pct.iloc[-1]:.0f}% of the 100%-CTRL1 invoice amount",
             x=0.07, ha="left", y=0.965, fontsize=14, fontweight="bold", color=INK)
ax.set_title(f"ZNB1 · {t.amount.sum() / 1e6:,.0f}M of {total / 1e6:,.0f}M NOK across the {len(df)} suppliers that "
             f"always go to CTRL1 · label under each name = main clerk, POs", loc="left", fontsize=9.5, color=INK2, pad=26)
h, l = ax.get_legend_handles_labels()
ax.legend(h[::-1], l[::-1], loc="upper left", bbox_to_anchor=(0, 1.07), ncol=2, frameon=False, fontsize=9,
          labelcolor=INK2, handlelength=1.6, borderaxespad=0)
fig.text(0.07, 0.01, "Invoices posted straight from the manual queue. Invoice amount = PO value split evenly over "
         "the PO's posted invoices. Main clerk = most frequent poster.", fontsize=7.5, color=INK2, ha="left")
fig.subplots_adjust(left=0.07, right=0.98, top=0.84, bottom=0.13)
fig.savefig(PNG, facecolor=SURFACE)
print(df[["Supplier", "main_clerk", "pos", "invoices", "amount", "share_pct", "cum_pct", "missed"]].head(TOP_N + 2).to_string())
print("total", total, "suppliers", len(df))
print(PNG)
