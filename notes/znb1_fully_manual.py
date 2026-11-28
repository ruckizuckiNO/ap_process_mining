"""ZNB1 suppliers whose invoices are ALL posted manually: who handles them, how fast, what it costs.

Uses the pipeline outputs (run 01_prepare.py first), so the clock is the pipeline's: for ZNB1 it starts at goods
receipt where a WAIT_GR step exists, else at the first event.
Run from the repo root:  conda run -n datasci python -W ignore notes/znb1_fully_manual.py
"""
from pathlib import Path

import pandas as pd

from common import SLA_SOFT_DAYS, load_cases, load_events, load_invoices

NOTES = Path(__file__).resolve().parent
K = ["PO_ID", "SAP_PO_No"]

inv = load_invoices()
inv = inv[(inv.po_type == "ZNB1") & inv.posted].set_index(K)
ev = load_events()
z = ev[ev.SAP_PO_Type_Code == "ZNB1"]
posted_by = z[z.Activity_ID == "Posted manually"].drop_duplicates(K).set_index(K).User_ID
inv["posted_by"] = posted_by.reindex(inv.index)
inv["manual"] = inv.posted_by.notna()
inv["direct_to_queue"] = z[(z.Activity_ID == "Initial validation") & (z.SAP_Next_Processing_Type == "CTRL1")] \
    .groupby(K).size().reindex(inv.index).notna()
inv["supplier"] = z.groupby(K).Supplier_ID.first().reindex(inv.index)
po_value = load_cases().set_index("PO_ID").invoice_value_nok
inv["value_nok"] = po_value.reindex(inv.index.get_level_values("PO_ID")).values / inv.groupby(level="PO_ID").manual.transform("size")
inv["late"] = inv.time_to_post_days > SLA_SOFT_DAYS
inv["missed_discount_nok"] = inv.value_nok * 0.02 * inv.late

g = inv.groupby("supplier")
sup = pd.DataFrame({
    "invoices": g.size(),
    "manual_pct": 100 * g.manual.mean(),
    "direct_to_queue_pct": 100 * g.direct_to_queue.mean(),
    "median_days_to_post": g.time_to_post_days.median(),
    "late_pct": 100 * g.late.mean(),
    "late_invoices": g.late.sum(),
    "missed_discount_nok": g.missed_discount_nok.sum(),
    "po_value_nok": inv.reset_index().drop_duplicates("PO_ID").groupby("supplier").PO_ID.apply(
        lambda p: po_value.reindex(p).sum()),
    "main_clerk": g.posted_by.agg(lambda u: u.value_counts().index[0] if u.notna().any() else None),
    "main_clerk_pct": g.posted_by.agg(lambda u: 100 * u.value_counts(normalize=True).iloc[0] if u.notna().any() else 0),
})
# PO values per supplier (one value per PO); "large-PO" = median PO value in ZNB1's top quartile
cases = load_cases()
cases = cases[(cases.po_type == "ZNB1") & (cases.invoice_value_nok > 0)]
hv_cut = cases.invoice_value_nok.quantile(0.75)
pv = cases.groupby("supplier").invoice_value_nok
sup["pos"] = pv.size()
sup["median_po_value_nok"] = pv.median()
sup["mean_po_value_nok"] = pv.mean()
sup["max_po_value_nok"] = pv.max()
sup["segment"] = (sup.median_po_value_nok >= hv_cut).map({True: "large POs", False: "small POs"})
full = sup[sup.manual_pct == 100].sort_values("missed_discount_nok", ascending=False).round(1)
full.insert(0, "rank", range(1, len(full) + 1))
full.to_csv(NOTES / "znb1_fully_manual_suppliers.csv")

f = inv[inv.supplier.isin(full.index)]
by_clerk = f.groupby("posted_by").agg(invoices=("late", "size"), median_days_to_post=("time_to_post_days", "median"),
                                      late_pct=("late", "mean"), missed_discount_nok=("missed_discount_nok", "sum"))
by_clerk["late_pct"] *= 100
by_clerk = by_clerk.sort_values("invoices", ascending=False).round(1)
by_clerk.to_csv(NOTES / "znb1_fully_manual_by_clerk.csv")

# --- Overview figure ------------------------------------------------------------
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from common import use_invoice_proxy_labels

use_invoice_proxy_labels()  # chart text: invoice -> invoice proxy

INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dc"
C1341, COTHER, CGREY = "#2a78d6", "#eb6834", "#8a8980"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 4, figsize=(23, 5.8), gridspec_kw={"width_ratios": [1.15, 1.2, 0.85, 0.9]})
is1341 = full.main_clerk == "User1341"

# 1: missed discount by supplier (top 15), coloured by main clerk, segment in the label
ax = axes[0]
t = full.head(15).iloc[::-1]
ax.barh(t.index, t.missed_discount_nok / 1e6, color=np.where(t.main_clerk == "User1341", C1341, COTHER), height=0.65)
for y, (s, r) in enumerate(t.iterrows()):
    ax.text(r.missed_discount_nok / 1e6 + 0.05, y, f"{r.late_pct:.0f}% late · {r.segment} · {r.main_clerk}",
            va="center", fontsize=8.5, color=INK)
ax.set_xlim(0, t.missed_discount_nok.max() / 1e6 * 1.9)
ax.set_xlabel("Missed 2% discount (M NOK)")
ax.set_title("Top 15 fully manual suppliers by missed discount", loc="left", color=INK, fontweight="bold")
ax.grid(axis="x", color=GRID); ax.set_axisbelow(True)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=C1341), plt.Rectangle((0, 0), 1, 1, color=COTHER)],
          labels=["Main clerk User1341", "Other clerks"], frameon=False, loc="lower right", fontsize=9)

# 2: median PO value vs median days to posting, one dot per supplier (size = invoices)
ax = axes[1]
for mask, col, lab in [(is1341, C1341, "Main clerk User1341"), (~is1341, COTHER, "Other clerks")]:
    d = full[mask]
    ax.scatter(d.median_po_value_nok, d.median_days_to_post, s=20 + np.sqrt(d.invoices) * 9, color=col, alpha=0.8,
               edgecolor="white", linewidth=1.5, label=lab)
for s in ["Supplier381", "Supplier361", "Supplier7", "Supplier377", "Supplier282", "Supplier281", "Supplier199"]:
    if s in full.index:
        r = full.loc[s]
        ax.annotate(s.replace("Supplier", "S"), (r.median_po_value_nok, r.median_days_to_post),
                    xytext=(6, 4), textcoords="offset points", fontsize=8.5, color=INK)
ax.axhline(10, color=INK, linewidth=1, linestyle="--")
ax.text(full.median_po_value_nok.min() * 1.1, 10.4, "10-day limit", fontsize=9, color=INK)
ax.axvline(hv_cut, color=MUTED, linewidth=1, linestyle=":")
ax.text(hv_cut * 1.1, ax.get_ylim()[1] * 0.93, "large-PO cut (937k)", fontsize=9, color=MUTED)
ax.set_xscale("log")
ax.set_xlabel("Median PO value (NOK, log scale)")
ax.set_ylabel("Median days to posting")
ax.set_title("Same ~8 days at User1341, whatever the PO value", loc="left", color=INK, fontweight="bold")
ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.grid(color=GRID); ax.set_axisbelow(True)

# 3: PO value distribution, fully manual vs other suppliers
ax = axes[2]
grp = cases.supplier.isin(full.index)
qs = [0.25, 0.5, 0.75, 0.9]
vals = pd.DataFrame({"37 fully manual": cases[grp].invoice_value_nok.quantile(qs).values,
                     "Other 179": cases[~grp].invoice_value_nok.quantile(qs).values},
                    index=["25th pct", "Median", "75th pct", "90th pct"]) / 1e6
x = np.arange(len(vals))
for off, col, c in [(-0.2, "37 fully manual", C1341), (0.2, "Other 179", CGREY)]:
    ax.bar(x + off, vals[col], width=0.38, color=c, label=col)
    for xi, v in zip(x + off, vals[col]):
        ax.text(xi, v + 0.08, f"{v:.2f}", ha="center", fontsize=8.5, color=INK)
ax.set_xticks(x, vals.index)
ax.set_ylabel("PO value (M NOK)")
ax.set_title("Typical PO alike, long tail of large POs", loc="left", color=INK, fontweight="bold")
ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)

# 4: by clerk, median days and late share (fully manual suppliers only)
ax = axes[3]
bc = by_clerk[by_clerk.invoices >= 20].iloc[::-1]
ax.barh(bc.index, bc.median_days_to_post, color=np.where(bc.index == "User1341", C1341, COTHER), height=0.6)
for y, (u, r) in enumerate(bc.iterrows()):
    ax.text(r.median_days_to_post + 0.3, y, f"{r.median_days_to_post:.1f} d · {r.late_pct:.0f}% late · {r.invoices:.0f} inv",
            va="center", fontsize=8.5, color=INK)
ax.axvline(10, color=INK, linewidth=1, linestyle="--")
ax.set_xlim(0, bc.median_days_to_post.max() * 1.9)
ax.set_xlabel("Median days to posting")
ax.set_title("Queue speed depends on the clerk", loc="left", color=INK, fontweight="bold")
ax.grid(axis="x", color=GRID); ax.set_axisbelow(True)

fig.tight_layout()
(NOTES / "figures").mkdir(exist_ok=True)
fig.savefig(NOTES / "figures" / "znb1_fully_manual_overview.png", dpi=140)

pd.set_option("display.width", 220)
print(full.groupby("segment").agg(suppliers=("invoices", "size"), invoices=("invoices", "sum"),
                                  late=("late_invoices", "sum"), disc=("missed_discount_nok", "sum")).round(0))
print(f"fully manual suppliers: {len(full)} of {len(sup)}; invoices {len(f)} of {len(inv)} "
      f"({100 * len(f) / len(inv):.0f}%); late {f.late.sum()} of {inv.late.sum()}; "
      f"missed discount {f.missed_discount_nok.sum() / 1e6:.1f}M of {inv.missed_discount_nok.sum() / 1e6:.1f}M NOK; "
      f"PO value {full.po_value_nok.sum() / 1e6:.0f}M of {sup.po_value_nok.sum() / 1e6:.0f}M; "
      f"direct to queue {100 * f.direct_to_queue.mean():.0f}%")
print(full.head(20).to_string())
print(by_clerk.to_string())
