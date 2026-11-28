"""Who posts the invoices of the ZNB1 suppliers that go 100% to CTRL1, and how many postings they make on a workday.

Invoices: as in znb1_ctrl1_100pct_suppliers.py (suppliers with 100% CTRL1 at first validation; invoices posted straight
from the queue). Per user:
  - share of these invoices posted by the user
  - per active workday (Mon-Fri with at least one "Posted manually" by the user, any PO type): postings of these
    invoices and all manual postings (all PO types); mean (stacked bar) and median of the total (dot)
Outputs: notes/figures/znb1_ctrl1_100pct_users.png, notes/znb1_ctrl1_100pct_users.csv
Run from the repo root:  PYTHONPATH=. conda run -n datasci python -W ignore notes/znb1_ctrl1_100pct_users.py
"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import BOT_USERS, POSTING_ACTIVITIES, load_events
from common import use_invoice_proxy_labels

use_invoice_proxy_labels()  # chart text: invoice -> invoice proxy

NOTES = Path(__file__).resolve().parent
K = ["PO_ID", "SAP_PO_No"]
MIN_INV = 20  # users with fewer of these invoices are grouped as "others"

ev = load_events().sort_values(["PO_ID", "ts", "Event_ID"])
z = ev[ev.SAP_PO_Type_Code == "ZNB1"].copy()
z["nxt"] = z.groupby(K).Activity_ID.shift(-1)
fv = z[z.Activity_ID == "Initial validation"].groupby(K).agg(route=("SAP_Next_Processing_Type", "first"),
                                                             nxt=("nxt", "first"), sup=("Supplier_ID", "first"))
share100 = fv.groupby("sup").route.agg(lambda x: (x == "CTRL1").mean())
ids = fv.index[fv.sup.isin(share100.index[share100 == 1]) & fv.nxt.isin(POSTING_ACTIVITIES)]
pz = z[z.Activity_ID.isin(POSTING_ACTIVITIES)].groupby(K).first().loc[ids].reset_index()
pz["day"] = pz.ts.dt.normalize()
n_inv, n_sup = len(pz), fv.loc[ids, "sup"].nunique()

# Every posting event by a person, all PO types (for these clerks that is only "Posted manually"), split by order type
allp = ev[ev.Activity_ID.isin(POSTING_ACTIVITIES) & ~ev.User_ID.isin(BOT_USERS)].copy()
allp["day"] = allp.ts.dt.normalize()
sup35 = set(fv.loc[ids, "sup"])
CATS = {f"ZNB1, the {n_sup} suppliers": "#2a78d6", "ZNB1, other suppliers": "#eb6834", "ZJD1": "#1baf7a",
        "ZJD3": "#eda100", "Other PO types": "#c9d1cd"}
cat_names = list(CATS)
allp["cat"] = np.select(
    [(allp.SAP_PO_Type_Code == "ZNB1") & allp.Supplier_ID.isin(sup35), allp.SAP_PO_Type_Code == "ZNB1",
     allp.SAP_PO_Type_Code == "ZJD1", allp.SAP_PO_Type_Code == "ZJD3"], cat_names[:4], cat_names[4])
weekdays = pd.bdate_range(allp.day.min(), allp.day.max())

rows = []
for u, n in pz.User_ID.value_counts().items():
    a = allp[allp.User_ID == u]
    days = np.sort(a[a.day.dt.dayofweek < 5].day.unique())  # Mon-Fri with at least one posting
    a = a[a.day.isin(days)]                                    # weekend postings are left out
    tot = a.groupby("day").size().reindex(days, fill_value=0)
    by_cat = a.groupby(["day", "cat"]).size().unstack(fill_value=0).reindex(index=days, columns=cat_names, fill_value=0)
    row = dict(user=u, invoices=n, share=n / n_inv, active_days=len(days),
               all_postings=len(a), all_mean=tot.mean(), all_median=tot.median(), all_q1=tot.quantile(0.25),
               all_q3=tot.quantile(0.75), all_max=int(tot.max()), these_median=by_cat[cat_names[0]].median())
    for c in cat_names:
        row[f"mean: {c}"] = by_cat[c].mean()
        row[f"postings: {c}"] = int(by_cat[c].sum())
    rows.append(row)
r = pd.DataFrame(rows).set_index("user")
r["share_35_of_own"] = r[f"postings: {cat_names[0]}"] / r.all_postings
r.round(2).to_csv(NOTES / "znb1_ctrl1_100pct_users.csv")

top = r[r.invoices >= MIN_INV]
rest = r[r.invoices < MIN_INV]

# ---------------------------------------------------------------- chart
INK, MUTED, GRID = "#16201c", "#5a6a63", "#d9e1dd"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "axes.edgecolor": GRID,
                     "xtick.color": MUTED, "ytick.color": INK})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.333, 7.5), sharey=True, gridspec_kw={"width_ratios": [1, 1.6], "wspace": 0.06})
labels = [f"{u}  ({int(x.active_days)} workdays)" for u, x in top.iterrows()]
y = np.arange(len(top))[::-1]

# Left: share of these invoices
a1.barh(y, top.share * 100, color=CATS[cat_names[0]], height=0.6)
for yi, (u, x) in zip(y, top.iterrows()):
    a1.text(x.share * 100 + 1.5, yi, f"{x.share:.0%}  ({int(x.invoices):,})", va="center", fontsize=10, color=INK)
a1.set_xlim(0, 100)
a1.set_title(f"Share of the {n_inv:,} invoices posted", loc="left", fontsize=12.5, fontweight="bold", color=INK, pad=10)
a1.set_xticks([0, 25, 50, 75, 100])
a1.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])

# Right: postings per active workday (mean, stacked) + median total (dot)
left = np.zeros(len(top))
for c, col in CATS.items():
    w = top[f"mean: {c}"].to_numpy()
    a2.barh(y, w, left=left, color=col, height=0.6, label=c, edgecolor="white", linewidth=1)
    left += w
a2.scatter(top.all_median, y, s=60, color=INK, zorder=5, edgecolor="white", linewidth=1.5, label="Median total on a workday")
for yi, (u, x) in zip(y, top.iterrows()):
    a2.text(max(x.all_mean, x.all_median) + 0.8, yi,
            f"{x.all_mean:.0f} / day · {x.share_35_of_own:.0%} for the {n_sup}", va="center", fontsize=9.5, color=INK)
a2.set_xlim(0, max(top.all_mean.max(), top.all_median.max()) * 1.65)
a2.set_title("All postings per workday, all PO types (mean)", loc="left", fontsize=12.5, fontweight="bold", color=INK, pad=10)
fig.legend(*a2.get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.02, 0.845), ncol=6, frameon=False, fontsize=9.5)

for ax in (a1, a2):
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
a1.set_yticks(y)
a1.set_yticklabels(labels)

u0 = top.iloc[0]
fig.text(0.02, 0.94, f"ZNB1: who posts the invoices of the {n_sup} suppliers that always go to the manual queue (CTRL1)",
         fontsize=15, fontweight="bold", color=INK)
fig.text(0.02, 0.92, va="top", s=f"{top.index[0]} posts {u0.share:.0%} of them, and they make up {u0.share_35_of_own:.0%} of all {top.index[0]}'s postings "
         f"across all orders (typical workday: {u0.these_median:.0f} of {u0.all_median:.0f}).\nFor the other clerks they are a small side task.",
         fontsize=12, color=MUTED, linespacing=1.4)
fig.text(0.02, 0.025,
         f"Left: the {n_inv:,} ZNB1 invoices of the {n_sup} suppliers posted straight from CTRL1. Right: every posting event by the clerk on any "
         f"purchase order (all PO types; for these clerks always 'Posted manually').\nWorkday = any day with at least one posting by the "
         f"clerk, Monday–Friday only (weekend postings left out); number of such days in brackets, out of {len(weekdays)} weekdays in the log.\n"
         f"Bars = mean per workday; dot = median total. {len(rest)} clerks with fewer than {MIN_INV} of these invoices ({int(rest.invoices.sum())}) "
         f"are not shown. Other PO types: ZNB3, ZRPC, ZRPM, NB2. Source: notes/znb1_ctrl1_100pct_users.csv",
         fontsize=8.5, color=MUTED, linespacing=1.4)
fig.subplots_adjust(left=0.17, right=0.98, top=0.76, bottom=0.15)
png = NOTES / "figures" / "znb1_ctrl1_100pct_users.png"
fig.savefig(png, dpi=200, facecolor="white")
plt.close(fig)
print("wrote", png)
print(r.round(1).to_string())
