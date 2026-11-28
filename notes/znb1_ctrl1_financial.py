"""Financial overview of the ZNB1 manual queue (CTRL1): missed 2% discount and late-payment interest, User1341 vs the
other clerks.

Scope (same as znb1_ctrl1_user1341_lateness.py): posted ZNB1 invoices routed to CTRL1 at their first Initial
validation and posted manually (all suppliers). Clerk = user of that posting. Days = pipeline clock
(time_to_post_days: arrival in AP to first posting).
Invoice amount = PO value (Case_Invoice_Value_NOK) split evenly over the PO's posted invoices (pipeline rule).
Cost per invoice:
  missed discount = 2% x amount                                   if days > 10
  late interest   = amount x 12.5% p.a. x (days - 30) / 365       if days > 30
An invoice posted after 30 days carries both. Assumption: payment on the posting day (the log ends at posting), so
interest runs from day 30 to the posting; any later payment would add more interest.
-> notes/figures/znb1_ctrl1_financial.png + notes/znb1_ctrl1_financial.csv
Run from the repo root:  PYTHONPATH=. conda run -n datasci python -W ignore notes/znb1_ctrl1_financial.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import POSTING_ACTIVITIES, SLA_HARD_DAYS, SLA_SOFT_DAYS, load_cases, load_events, load_invoices
from common import use_invoice_proxy_labels

use_invoice_proxy_labels()  # chart text: invoice -> invoice proxy

NOTES = Path(__file__).resolve().parent
PNG = NOTES / "figures" / "znb1_ctrl1_financial.png"
CSV = NOTES / "znb1_ctrl1_financial.csv"
K = ["PO_ID", "SAP_PO_No"]
FOCUS = "User1341"
MIN_POSTINGS = 100
DISCOUNT = 0.02
INTEREST = 0.125  # p.a.

BLUE_LIGHT, ORANGE, RED = "#9cc3ee", "#eb6834", "#a3261b"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"

# ---------------------------------------------------------------- data
ev = load_events()
z = ev[ev.SAP_PO_Type_Code == "ZNB1"].sort_values(["PO_ID", "ts", "Event_ID"])
route = z[z.Activity_ID == "Initial validation"].groupby(K).SAP_Next_Processing_Type.first().rename("route")
first_post = z[z.Activity_ID.isin(POSTING_ACTIVITIES)].groupby(K).agg(act=("Activity_ID", "first"), user=("User_ID", "first"))
inv = load_invoices()
inv = inv[inv.po_type == "ZNB1"].set_index(K)
po_value = load_cases().set_index("PO_ID").invoice_value_nok
inv["n_posted"] = inv.groupby(level="PO_ID").posted.transform("sum")
inv["amount"] = np.where(inv.posted, po_value.reindex(inv.index.get_level_values(0)).values
                         / inv.n_posted.where(inv.n_posted > 0), 0)
inv = inv[inv.posted].join(route).join(first_post)
m = inv[(inv.route == "CTRL1") & (inv.act == "Posted manually")].copy()
d = m.time_to_post_days
m["late10"], m["late30"] = d > SLA_SOFT_DAYS, d > SLA_HARD_DAYS
m["discount"] = DISCOUNT * m.amount * m.late10
m["interest"] = m.amount * INTEREST * (d - SLA_HARD_DAYS).clip(lower=0) / 365
m["cost"] = m.discount + m.interest


def summary(sub):
    return pd.Series({
        "invoices": len(sub), "amount": sub.amount.sum(),
        "late10": int(sub.late10.sum()), "late30": int(sub.late30.sum()),
        "amount_late10": sub.amount[sub.late10].sum(), "amount_late30": sub.amount[sub.late30].sum(),
        "days_over30_mean": (sub.time_to_post_days[sub.late30] - SLA_HARD_DAYS).mean() if sub.late30.any() else 0.0,
        "missed_discount": sub.discount.sum(), "late_interest": sub.interest.sum(), "total_cost": sub.cost.sum(),
    })


others = m[m.user != FOCUS]
big = others.user.value_counts()
big = big[big >= MIN_POSTINGS].index
rows = {FOCUS: summary(m[m.user == FOCUS]), "All other clerks": summary(others), "All clerks": summary(m)}
rows.update({u: summary(others[others.user == u]) for u in big})
tab = pd.DataFrame(rows).T
tab["cost_per_mnok"] = tab.total_cost / (tab.amount / 1e6)  # NOK lost per 1M NOK handled
tab.index.name = "clerk"
tab.to_csv(CSV)
tot = tab.loc["All clerks"]

# ---------------------------------------------------------------- chart
# Left: cost in NOK (discount + interest, stacked). Right: same rows, cost per 1M NOK handled.
order = [FOCUS, "All other clerks", None] + list(tab.loc[big].sort_values("total_cost", ascending=False).index)
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"]})
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 6.8), dpi=200, sharey=True, gridspec_kw={"width_ratios": [2.3, 1], "wspace": 0.06})
fig.patch.set_facecolor(SURFACE)
ypos, labels, y = [], [], 0.0
xmax = tab.loc[[FOCUS, "All other clerks"], "total_cost"].max() / 1e6
rmax = tab.loc[[r for r in order if r], "cost_per_mnok"].max()
for name in order:
    if name is None:
        y -= 0.5
        continue
    r = tab.loc[name]
    h = 0.72 if name in (FOCUS, "All other clerks") else 0.55
    dsc, itr = r.missed_discount / 1e6, r.late_interest / 1e6
    ax.barh(y, dsc, height=h, color=ORANGE, edgecolor=SURFACE, linewidth=1.5, zorder=2)
    ax.barh(y, itr, left=dsc, height=h, color=RED, edgecolor=SURFACE, linewidth=1.5, zorder=2)
    lab = f"{(dsc + itr):,.2f}M  ({dsc:,.2f}M + {itr * 1e3:,.0f}k interest)" if itr > 0 else f"{dsc:,.2f}M  (no interest)"
    ax.text(dsc + itr + xmax * 0.01, y, lab, va="center", ha="left", fontsize=9 if h > 0.6 else 8.5, color=INK)
    ax2.barh(y, r.cost_per_mnok, height=h, color=BLUE_LIGHT if name not in (FOCUS, "All other clerks") else "#2a78d6",
             edgecolor=SURFACE, linewidth=1.5, zorder=2)
    ax2.text(r.cost_per_mnok + rmax * 0.02, y, f"{r.cost_per_mnok:,.0f}", va="center", ha="left",
             fontsize=9 if h > 0.6 else 8.5, color=INK)
    ypos.append(y)
    labels.append(name)
    y -= 1.0
for a in (ax, ax2):
    a.set_facecolor(SURFACE)
    a.grid(axis="x", color=GRID, zorder=0)
    a.tick_params(axis="both", length=0, colors=INK2)
    for s in a.spines.values():
        s.set_visible(False)
ax.set_yticks(ypos)
ax.set_yticklabels(labels)
for t in ax.get_yticklabels():
    bold = t.get_text() in (FOCUS, "All other clerks")
    t.set_fontweight("bold" if bold else "normal")
    t.set_fontsize(11 if bold else 9.5)
    t.set_color(INK)
ax.set_xlim(0, xmax * 1.55)
ax.set_xlabel("Cost (M NOK)", color=INK2, fontsize=10)
ax.set_title("Cost: missed 2% discount + late interest", loc="left", fontsize=11, color=INK, fontweight="bold")
ax2.set_xlim(0, rmax * 1.3)
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
ax2.set_xlabel("NOK per 1M NOK of invoices handled", color=INK2, fontsize=10)
ax2.set_title("Cost per 1M NOK handled", loc="left", fontsize=11, color=INK, fontweight="bold")
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (ORANGE, RED)]
ax.legend(handles, [f"Missed 2% discount (posted > {SLA_SOFT_DAYS} days)",
                    f"Late interest {INTEREST:.1%} p.a. (days beyond {SLA_HARD_DAYS})"],
          loc="lower right", frameon=False, fontsize=9, labelcolor=INK2)
f, o = tab.loc[FOCUS], tab.loc["All other clerks"]
fig.suptitle(f"ZNB1 manual queue (CTRL1): {tot.total_cost / 1e6:,.2f}M NOK lost, "
             f"{f.total_cost / tot.total_cost:.0%} of it on invoices posted by {FOCUS}",
             x=0.01, ha="left", y=0.975, fontsize=13.5, fontweight="bold", color=INK)
fig.text(0.01, 0.912,
         f"{int(tot.invoices):,} invoices worth {tot.amount / 1e6:,.0f}M NOK · missed discount {tot.missed_discount / 1e6:,.2f}M "
         f"on {int(tot.late10):,} invoices > {SLA_SOFT_DAYS} days · late interest {tot.late_interest / 1e3:,.0f}k NOK on "
         f"{int(tot.late30)} invoices > {SLA_HARD_DAYS} days ({f.late30:.0f} of them {FOCUS}'s)",
         ha="left", fontsize=9.5, color=INK2)
fig.text(0.01, 0.012, f"Posted ZNB1 invoices routed to CTRL1 at their first initial validation and posted manually (all "
         f"suppliers). Days = arrival in AP to first posting. Amount = PO value split evenly over the PO's posted invoices. "
         f"Interest assumes payment on the posting day.", ha="left", fontsize=7.5, color=INK2)
fig.subplots_adjust(left=0.12, right=0.98, top=0.82, bottom=0.1)
fig.savefig(PNG, facecolor=SURFACE)
pd.set_option("display.width", 220)
print(tab.round(0).to_string())
print(PNG)
