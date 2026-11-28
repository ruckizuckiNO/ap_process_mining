"""Always-manual ZNB1 suppliers (the 28 in znb1_always_manual_suppliers.xlsx): average throughput time per week and
month for invoices on POs worth more than 1.5M NOK.

Throughput time = days from the invoice's clock start to its first posting (pipeline clock; these suppliers skip the
goods-receipt wait, so it is arrival in AP -> posting). Grouped by the week / month the invoice arrived.
Run from the repo root:  PYTHONPATH=. conda run -n datasci python -W ignore notes/znb1_always_manual_large_po_trend.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import SLA_SOFT_DAYS, load_cases, load_events, load_invoices
from common import use_invoice_proxy_labels

use_invoice_proxy_labels()  # chart text: invoice -> invoice proxy

NOTES = Path(__file__).resolve().parent
K = ["PO_ID", "SAP_PO_No"]
MIN_PO = 1_500_000
MIN_POS, MIN_DIRECT = 5, 0.95

ev = load_events()
z = ev[ev.SAP_PO_Type_Code == "ZNB1"]
po_value = load_cases().set_index("PO_ID").invoice_value_nok
inv = load_invoices().set_index(K)

# same supplier list as the Excel file
iv = z[z.Activity_ID == "Initial validation"].sort_values("ts").drop_duplicates(K).set_index(K)
r = pd.DataFrame({"supplier": iv.Supplier_ID, "direct": iv.SAP_Next_Processing_Type == "CTRL1"})
g = r.groupby("supplier")
n_pos = r.reset_index().groupby("supplier").PO_ID.nunique()
always = g.direct.mean()[(g.direct.mean() >= MIN_DIRECT) & (n_pos >= MIN_POS)].index

x = inv[(inv.po_type == "ZNB1") & inv.posted].copy()
x["supplier"] = r.supplier.reindex(x.index)
x["po_value"] = po_value.reindex(x.index.get_level_values("PO_ID")).values
x = x[x.supplier.isin(always) & (x.po_value > MIN_PO)]
x["days"] = x.time_to_post_days
x["week"] = x.clock_start.dt.to_period("W-SUN")
x["month"] = x.clock_start.dt.to_period("M")


def agg(col):
    a = x.groupby(col).agg(invoices=("days", "size"), mean_days=("days", "mean"), median_days=("days", "median"),
                           late_pct=("days", lambda d: 100 * (d > SLA_SOFT_DAYS).mean()))
    return a


wk, mo = agg("week"), agg("month")
wk = wk.reindex(pd.period_range(wk.index.min(), wk.index.max(), freq="W-SUN"))
pd.concat({"week": wk, "month": mo}, names=["view", "period"]).round(2).to_csv(
    NOTES / "znb1_always_manual_large_po_trend.csv")

# --- Figure --------------------------------------------------------------------------------
INK, MUTED, GRID, C, CL = "#1f1f1e", "#6b6a64", "#e4e3dc", "#2a78d6", "#b7d3f6"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 2, figsize=(19, 8.5), gridspec_kw={"width_ratios": [2.2, 1], "height_ratios": [2.4, 1]},
                         sharex="col")
overall = x.days.mean()
fig.suptitle(f"Always-manual ZNB1 suppliers, invoices on POs > {MIN_PO / 1e6:.1f}M NOK: average throughput time "
             f"({len(x):,} invoices, {x.index.get_level_values('PO_ID').nunique()} POs, overall mean "
             f"{overall:.1f} days)", x=0.01, ha="left", fontsize=14, fontweight="bold", color=INK)

ax = axes[0, 0]
xs = np.arange(len(wk))
ok = wk.invoices.fillna(0) >= 5
ax.plot(xs, wk.mean_days, color=C, linewidth=2, marker="o", markersize=5, label="Mean days to posting")
ax.scatter(xs[~ok.values], wk.mean_days[~ok.values], s=40, facecolor="white", edgecolor=C, zorder=3,
           label="Week with < 5 invoices")
ax.axhline(SLA_SOFT_DAYS, color=INK, linewidth=1, linestyle="--")
ax.text(len(wk) - 0.5, SLA_SOFT_DAYS + 0.3, "10-day limit", ha="right", fontsize=9, color=INK)
ax.set_ylabel("Mean days to posting")
ax.set_title("Per week (week the invoice arrived)", loc="left", color=INK, fontweight="bold")
ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)

ax = axes[1, 0]
ax.bar(xs, wk.invoices.fillna(0), color=CL, width=0.8)
ax.set_ylabel("Invoices")
step = 2
ax.set_xticks(xs[::step], [p.start_time.strftime("%d %b") for p in wk.index[::step]], rotation=45, ha="right")
ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)

ax = axes[0, 1]
xm = np.arange(len(mo))
ax.bar(xm, mo.mean_days, color=C, width=0.6)
for i, (p, rr) in enumerate(mo.iterrows()):
    ax.text(i, rr.mean_days + 0.3, f"{rr.mean_days:.1f} d\n{rr.late_pct:.0f}% late", ha="center", fontsize=9,
            color=INK)
ax.axhline(SLA_SOFT_DAYS, color=INK, linewidth=1, linestyle="--")
ax.set_ylim(0, max(mo.mean_days.max(), SLA_SOFT_DAYS) * 1.3)
ax.set_ylabel("Mean days to posting")
ax.set_title("Per month (month the invoice arrived)", loc="left", color=INK, fontweight="bold")
ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)

ax = axes[1, 1]
ax.bar(xm, mo.invoices, color=CL, width=0.6)
for i, n in enumerate(mo.invoices):
    ax.text(i, n, f"{int(n)}", ha="center", va="bottom", fontsize=9, color=INK)
ax.set_xticks(xm, [p.strftime("%b %y") for p in mo.index])
ax.set_ylabel("Invoices")
ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)

fig.text(0.01, 0.005, "From December the log only holds later invoices on POs started before 30 Nov 2025, so the "
         "last months are thin. Clock: pipeline (these suppliers skip the goods-receipt wait).",
         fontsize=9, color=MUTED)
fig.tight_layout(rect=(0, 0.02, 1, 0.94))
fig.savefig(NOTES / "figures" / "znb1_always_manual_large_po_trend.png", dpi=140)

pd.set_option("display.width", 200)
print(f"invoices {len(x)}, POs {x.index.get_level_values('PO_ID').nunique()}, suppliers {x.supplier.nunique()}, "
      f"mean {overall:.2f}, median {x.days.median():.2f}, late {100 * (x.days > 10).mean():.1f}%")
print(mo.round(1).to_string())
print(wk.round(1).to_string())
