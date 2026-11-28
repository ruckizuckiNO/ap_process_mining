"""Potential savings on User1341's invoices posted after more than 10 days: the 2% discount on their value, by supplier,
by how far past the limit they were, and how much faster queue handling would recover.

Savings = 2% of the invoice value (PO value split evenly over its posted invoices), assuming 2/10 terms. Clock as in
the pipeline. Scenario: User1341's time in the manual queue capped at N days, all else unchanged.
Run from the repo root:  PYTHONPATH=. conda run -n datasci python -W ignore notes/user1341_savings.py
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
U = "User1341"

ev = load_events()
inv = load_invoices().set_index(K)
inv = inv[inv.posted].copy()
po_value = load_cases().set_index("PO_ID").invoice_value_nok
inv["value_nok"] = po_value.reindex(inv.index.get_level_values("PO_ID")).values \
    / inv.groupby(level="PO_ID").posted.transform("size")
pm = ev[ev.Activity_ID == "Posted manually"].sort_values("ts").drop_duplicates(K).set_index(K)
u = inv.join(pm[["User_ID", "Supplier_ID", "ts"]], how="inner")
u = u[u.User_ID == U].copy()
u["queue_days"] = (u.ts - ev[ev.SAP_Next_Processing_Type == "CTRL1"].groupby(K).ts.min().reindex(u.index)) \
    .dt.total_seconds().div(86400).fillna(0)
u["days"] = u.time_to_post_days
u["late"] = u.days > SLA_SOFT_DAYS
u["savings_nok"] = 0.02 * u.value_nok * u.late
total = u.savings_nok.sum()

by_sup = u[u.late].groupby("Supplier_ID").agg(late_invoices=("late", "size"), late_value_nok=("value_nok", "sum"),
                                              savings_nok=("savings_nok", "sum")).sort_values("savings_nok",
                                                                                              ascending=False)
by_sup["cum_share_pct"] = 100 * by_sup.savings_nok.cumsum() / total
edges = [0, 1, 2, 3, 5, 10, 20, np.inf]
labels = ["≤1", "1–2", "2–3", "3–5", "5–10", "10–20", ">20"]
over = pd.cut(u.loc[u.late, "days"] - SLA_SOFT_DAYS, edges, labels=labels)
by_over = u[u.late].groupby(over).agg(late_invoices=("late", "size"), savings_nok=("savings_nok", "sum"))
caps = [1, 3, 5, 6, 7, 8]
scen = pd.DataFrame([{
    "queue_cap_days": c,
    "recovered_nok": u.savings_nok[u.late & ((u.days - u.queue_days + np.minimum(u.queue_days, c)) <= SLA_SOFT_DAYS)].sum(),
} for c in caps])
scen["recovered_pct"] = 100 * scen.recovered_nok / total
queue_wd = (u.ts - pd.to_timedelta(u.queue_days, unit="D")).dt.dayofweek.map(dict(enumerate(
    ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])))
by_weekday = u.groupby(queue_wd).agg(invoices=("late", "size"), late_pct=("late", "mean"),
                                     savings_nok=("savings_nok", "sum"))
by_weekday["late_pct"] *= 100
pd.concat({"supplier": by_sup, "days_over_limit": by_over, "queue_weekday": by_weekday,
           "queue_cap": scen.set_index("queue_cap_days")},
          names=["view", "group"]).round(1).to_csv(NOTES / "user1341_savings.csv")

# --- Figure ---------------------------------------------------------------------------------
INK, MUTED, GRID, C, C2 = "#1f1f1e", "#6b6a64", "#e4e3dc", "#2a78d6", "#b7d3f6"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 3, figsize=(20, 6.2), gridspec_kw={"width_ratios": [1.25, 1, 1]})
fig.suptitle(f"User1341: {total / 1e6:.1f}M NOK potential savings (2% discount) on {u.late.sum()} invoices "
             f"posted after more than 10 days", x=0.01, ha="left", fontsize=15, fontweight="bold", color=INK)

ax = axes[0]
t = by_sup.head(12).iloc[::-1]
ax.barh(t.index, t.savings_nok / 1e6, color=C, height=0.65)
for y, (s, r) in enumerate(t.iterrows()):
    ax.text(r.savings_nok / 1e6 + 0.05, y, f"{r.savings_nok / 1e6:.2f}M · {int(r.late_invoices)} inv · "
            f"cum. {r.cum_share_pct:.0f}%", va="center", fontsize=9.5, color=INK)
ax.set_xlim(0, t.savings_nok.max() / 1e6 * 1.6)
ax.set_xlabel("Potential savings (M NOK)")
ax.set_title("By supplier: three suppliers hold 74%", loc="left", color=INK, fontweight="bold")
ax.grid(axis="x", color=GRID); ax.set_axisbelow(True)

ax = axes[1]
x = np.arange(len(by_over))
ax.bar(x, by_over.savings_nok / 1e6, color=[C if i < 4 else C2 for i in range(len(by_over))], width=0.65)
for xi, (lab, r) in zip(x, by_over.iterrows()):
    ax.text(xi, r.savings_nok / 1e6 + 0.05, f"{r.savings_nok / 1e6:.1f}M\n{int(r.late_invoices)} inv", ha="center",
            fontsize=9, color=INK)
near = by_over.savings_nok.iloc[:4].sum()
ax.text(0.98, 0.95, f"{100 * near / total:.0f}% of the savings were\nmissed by 5 days or less",
        transform=ax.transAxes, ha="right", va="top", fontsize=10.5, color=INK)
ax.set_xticks(x, labels)
ax.set_ylim(0, by_over.savings_nok.max() / 1e6 * 1.35)
ax.set_xlabel("Days past the 10-day limit")
ax.set_ylabel("Potential savings (M NOK)")
ax.set_title("Most were missed by a few days", loc="left", color=INK, fontweight="bold")
ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)

ax = axes[2]
wd = by_weekday.reindex(["Mon", "Tue", "Wed", "Thu", "Fri"])
ax.bar(wd.index, wd.savings_nok / 1e6, color=["#eb6834" if d == "Fri" else C for d in wd.index], width=0.6)
for d, r in wd.iterrows():
    ax.text(d, r.savings_nok / 1e6 + 0.05, f"{r.savings_nok / 1e6:.1f}M\n{r.late_pct:.0f}% late", ha="center",
            fontsize=9.5, color=INK)
ax.set_ylim(0, wd.savings_nok.max() / 1e6 * 1.3)
ax.set_xlabel("Weekday the invoice entered the queue")
ax.set_ylabel("Potential savings (M NOK)")
ax.set_title("Friday arrivals carry the most", loc="left", color=INK, fontweight="bold")
ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)

fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(NOTES / "figures" / "user1341_savings.png", dpi=150)

pd.set_option("display.width", 200)
print(f"total {total:,.0f} NOK on {u.late.sum()} late invoices")
print(by_sup.round(0).to_string()); print(by_over.round(0).to_string()); print(scen.round(1).to_string())
