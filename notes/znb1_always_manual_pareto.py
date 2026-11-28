"""Pareto chart of missed 2% discount for the ZNB1 'always manual' suppliers (>= 5 POs, >= 95% of invoices straight
to CTRL1). Reads the list from notes/znb1_always_manual_suppliers.xlsx (same scope: POs with never-posted invoices
left out, GR clock) -> notes/figures/znb1_always_manual_pareto.png + notes/znb1_always_manual_pareto.csv.

Bars and line share one axis (% of the group's missed discount), so the chart needs no second y-scale.
Run from the repo root:  conda run -n datasci python -W ignore notes/znb1_always_manual_pareto.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter
from common import use_invoice_proxy_labels

use_invoice_proxy_labels()  # chart text: invoice -> invoice proxy

NOTES = Path(__file__).resolve().parent
SRC = NOTES / "znb1_always_manual_suppliers.xlsx"
PNG = NOTES / "figures" / "znb1_always_manual_pareto.png"
CSV = NOTES / "znb1_always_manual_pareto.csv"
CUTOFF = 80.0

BLUE, ORANGE, MUTED = "#2a78d6", "#eb6834", "#c9c8c2"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"

df = pd.read_excel(SRC, sheet_name="Always manual", header=3)
df = df[df.Supplier.astype(str).str.startswith("Supplier")]
df = df.rename(columns={"Missed 2% discount (NOK)": "missed", "Invoices": "invoices", "POs": "pos",
                        "Main clerk": "main_clerk", "Late > 10 days (share of posted)": "late_share"})
df = df.sort_values("missed", ascending=False).reset_index(drop=True)
total = df.missed.sum()
df["share_pct"] = 100 * df.missed / total
df["cum_pct"] = df.share_pct.cumsum()
# "Vital few" = the suppliers needed to reach the cutoff (the one that crosses it included).
n_vital = int((df.cum_pct < CUTOFF).sum()) + 1
df["vital_few"] = df.index < n_vital
df[["Supplier", "pos", "invoices", "main_clerk", "late_share", "missed", "share_pct", "cum_pct", "vital_few"]].to_csv(
    CSV, index=False)

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"]})
fig, ax = plt.subplots(figsize=(12, 6.2), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

x = range(len(df))
ax.bar(x, df.share_pct, width=0.72, color=[BLUE if v else MUTED for v in df.vital_few],
       edgecolor=SURFACE, linewidth=2, zorder=2, label="Share of missed discount (bar)")
ax.plot(x, df.cum_pct, color=ORANGE, linewidth=2, marker="o", markersize=5, markeredgecolor=SURFACE,
        markeredgewidth=1.5, zorder=3, label="Cumulative share (line)")
ax.axhline(CUTOFF, color=INK2, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
ax.text(len(df) - 0.5, CUTOFF + 1.5, f"{CUTOFF:.0f}%", ha="right", va="bottom", color=INK2, fontsize=9)
ax.axvline(n_vital - 0.5, color=INK2, linewidth=1, linestyle=(0, (1, 3)), zorder=1)

# Selective labels: NOK amount on the vital-few bars only, cumulative % where the line crosses the cutoff.
for i, r in df[df.vital_few].iterrows():
    if r.share_pct > 15:  # tall bar: label inside, clear of the cumulative line that starts at its top
        ax.text(i, r.share_pct - 2, f"{r.missed / 1e6:.1f}M", ha="center", va="top", fontsize=8.5,
                color="white", fontweight="bold")
    else:
        ax.text(i, r.share_pct + 1.2, f"{r.missed / 1e6:.1f}M", ha="center", va="bottom", fontsize=8.5, color=INK)
r = df.iloc[n_vital - 1]
ax.annotate(f"{n_vital} of {len(df)} suppliers = {r.cum_pct:.0f}%", xy=(n_vital - 1, r.cum_pct),
            xytext=(n_vital + 1.2, r.cum_pct - 16), fontsize=9.5, color=INK,
            arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.8))

ax.set_ylim(0, 105)
ax.set_xlim(-0.6, len(df) - 0.4)
ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
ax.set_ylabel("Share of the group's missed 2% discount", color=INK2, fontsize=10)
ax.set_xticks(list(x))
ax.set_xticklabels(df.Supplier.str.replace("Supplier", "S", regex=False), rotation=0, fontsize=8.5, color=INK2)
ax.set_xlabel("Supplier (S381 = Supplier381), sorted by missed discount", color=INK2, fontsize=10)
ax.tick_params(axis="both", length=0, colors=INK2)
ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.spines["bottom"].set_visible(True)
ax.spines["bottom"].set_color(GRID)

fig.suptitle("ZNB1 always-manual suppliers: where the missed 2% discount sits", x=0.06, ha="left", y=0.965,
             fontsize=15, fontweight="bold", color=INK)
ax.set_title(f"{len(df)} suppliers with ≥95% of invoices straight to the manual queue · "
             f"{total / 1e6:.1f}M NOK missed discount in total · blue = the vital few up to {CUTOFF:.0f}%",
             loc="left", fontsize=10, color=INK2, pad=26)
h, l = ax.get_legend_handles_labels()
ax.legend(h[::-1], l[::-1], loc="upper left", bbox_to_anchor=(0, 1.07), ncol=2, frameon=False, fontsize=9, labelcolor=INK2,
          handlelength=1.6, borderaxespad=0)
fig.text(0.06, 0.01, "Scope: ZNB1 POs without never-posted invoices; payment clock from goods receipt. "
         "Missed discount = 2% of invoices posted after 10 days. Source: notes/znb1_always_manual_suppliers.xlsx",
         fontsize=7.5, color=INK2, ha="left")
fig.subplots_adjust(left=0.06, right=0.98, top=0.84, bottom=0.12)
PNG.parent.mkdir(exist_ok=True)
fig.savefig(PNG, facecolor=SURFACE)
print(df[["Supplier", "missed", "share_pct", "cum_pct"]].head(n_vital + 2).to_string())
print("total", total, "vital", n_vital, "of", len(df))

# --- Top-4 version: only the suppliers that together make up about 80% (shares stay relative to all 25) ---
TOP_N = 4
PNG_TOP = NOTES / "figures" / f"znb1_always_manual_pareto_top{TOP_N}.png"
t = df.head(TOP_N)
fig, ax = plt.subplots(figsize=(9, 6), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
x = range(TOP_N)
ax.bar(x, t.share_pct, width=0.6, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=2,
       label="Share of missed discount (bar)")
ax.plot(x, t.cum_pct, color=ORANGE, linewidth=2, marker="o", markersize=7, markeredgecolor=SURFACE,
        markeredgewidth=1.5, zorder=3, label="Cumulative share (line)")
ax.axhline(CUTOFF, color=INK2, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
ax.text(TOP_N - 0.45, CUTOFF + 1.5, f"{CUTOFF:.0f}%", ha="right", va="bottom", color=INK2, fontsize=9)
for i, r in t.reset_index(drop=True).iterrows():
    inside = r.share_pct > 15
    ax.text(i, r.share_pct - 2 if inside else r.share_pct + 1.2,
            f"{r.missed / 1e6:.1f}M NOK\n{r.share_pct:.0f}%", ha="center", va="top" if inside else "bottom",
            fontsize=9, color="white" if inside else INK, fontweight="bold" if inside else "normal")
    if i > 0:
        ax.text(i, r.cum_pct + 2.5, f"{r.cum_pct:.0f}%", ha="center", va="bottom", fontsize=9, color=INK)
ax.set_ylim(0, 100)
ax.set_xlim(-0.5, TOP_N - 0.5)
ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
ax.set_ylabel("Share of the group's missed 2% discount", color=INK2, fontsize=10)
ax.set_xticks(list(x))
ax.set_xticklabels([f"{s}\n{c} · {p:.0f} POs" for s, c, p in zip(t.Supplier, t.main_clerk, t.pos)],
                   fontsize=9.5, color=INK)
ax.tick_params(axis="both", length=0, colors=INK2)
ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.spines["bottom"].set_visible(True)
ax.spines["bottom"].set_color(GRID)
fig.suptitle(f"{TOP_N} suppliers cause {t.cum_pct.iloc[-1]:.0f}% of the always-manual missed discount",
             x=0.09, ha="left", y=0.965, fontsize=14, fontweight="bold", color=INK)
ax.set_title(f"ZNB1 · {t.missed.sum() / 1e6:.1f}M of {total / 1e6:.1f}M NOK across the {len(df)} always-manual "
             f"suppliers · label under each name = main clerk, POs", loc="left", fontsize=9.5, color=INK2, pad=26)
h, l = ax.get_legend_handles_labels()
ax.legend(h[::-1], l[::-1], loc="upper left", bbox_to_anchor=(0, 1.07), ncol=2, frameon=False, fontsize=9, labelcolor=INK2,
          handlelength=1.6, borderaxespad=0)
fig.text(0.09, 0.01, "Scope: ZNB1 POs without never-posted invoices; payment clock from goods receipt. "
         "Missed discount = 2% of invoices posted after 10 days.", fontsize=7.5, color=INK2, ha="left")
fig.subplots_adjust(left=0.09, right=0.97, top=0.84, bottom=0.13)
fig.savefig(PNG_TOP, facecolor=SURFACE)
print(t[["Supplier", "main_clerk", "pos", "missed", "share_pct", "cum_pct"]].to_string())
