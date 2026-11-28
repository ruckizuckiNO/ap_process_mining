"""What stays manual above a 1.75M NOK PO-value cut? Overview of the remaining invoice proxies of the 35 ZNB1 suppliers
that go 100% to the manual queue (CTRL1), same scope as 13_report_po_cut_gbr.py.

Scenario (report_po_cut_gbr.html): invoice proxies on POs with PO value <= cut are handled like normal orders (ZJD1);
those on POs above the cut stay in the manual queue. This script describes the ones above 1.75M NOK as they are today.
Scope: ZNB1 suppliers with 100% of invoice proxies routed to CTRL1 at their first Initial validation (and at least one
posted straight from the queue), all their posted invoice proxies. Deleted and never-posted ones left out.
Clock: pipeline (goods receipt if WAIT_GR, else arrival in AP) -> first posting. Amount = PO value split evenly over the
PO's posted invoice proxies. Missed discount = 2% x amount if days > 10; late interest = amount x 12.5% p.a. x (days - 30)
/ 365 if days > 30. Poster = user of the first posting.
-> notes/figures/znb1_po_cut_1p75m_remaining.png, notes/znb1_po_cut_1p75m_remaining.xlsx
Run from the repo root:  PYTHONPATH=. conda run -n datasci python -W ignore notes/znb1_po_cut_1p75m_remaining.py
"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (POSTING_ACTIVITIES, SLA_HARD_DAYS, SLA_SOFT_DAYS, load_cases, load_events, load_invoices,
                    use_invoice_proxy_labels)

use_invoice_proxy_labels()  # chart text: invoice -> invoice proxy

NOTES = Path(__file__).resolve().parent
PNG = NOTES / "figures" / "znb1_po_cut_1p75m_remaining.png"
XLSX = NOTES / "znb1_po_cut_1p75m_remaining.xlsx"
K = ["PO_ID", "SAP_PO_No"]
CUT = 1.75e6
DISCOUNT, INTEREST = 0.02, 0.125
FOCUS = "User1341"
TOP = 8
BANDS = [CUT, 2.5e6, 5e6, 10e6, 25e6, 50e6, np.inf]
BAND_LABELS = ["1.75–2.5M", "2.5–5M", "5–10M", "10–25M", "25–50M", "> 50M"]

BLUE, ORANGE, RED, GREY = "#2a78d6", "#eb6834", "#a3261b", "#9aa5a0"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"]})

# ---------------------------------------------------------------- data (selection as in 13_report_po_cut_gbr.py)
ev = load_events()
ev = ev[ev.SAP_PO_Type_Code == "ZNB1"].sort_values(["PO_ID", "ts", "Event_ID"])
ev["nxt"] = ev.groupby(K).Activity_ID.shift(-1)
iv = ev[ev.Activity_ID == "Initial validation"].groupby(K)
route, nxt = iv.SAP_Next_Processing_Type.first(), iv.nxt.first()
poster = ev[ev.Activity_ID.isin(POSTING_ACTIVITIES)].groupby(K).User_ID.first()

Z = load_invoices()
Z = Z[Z.po_type == "ZNB1"].set_index(K)
Z["route"], Z["supplier"], Z["poster"] = route, ev.groupby(K).Supplier_ID.first(), poster
Z["po_value"] = load_cases().set_index("PO_ID").invoice_value_nok.reindex(Z.index.get_level_values(0)).values
share = Z[Z.route.notna()].groupby("supplier").route.agg(lambda x: (x == "CTRL1").mean())
s100 = share.index[share == 1]
direct = Z.posted & nxt.reindex(Z.index).isin(POSTING_ACTIVITIES)
s100 = s100.intersection(Z.loc[direct, "supplier"].unique())
Z["n_posted"] = Z.groupby(level="PO_ID").posted.transform("sum")
Z["amount"] = np.where(Z.posted, Z.po_value / Z.n_posted.where(Z.n_posted > 0), 0)
Z = Z[Z.supplier.isin(s100) & Z.posted].copy()
d = Z.time_to_post_days
Z["late10"], Z["late30"] = d > SLA_SOFT_DAYS, d > SLA_HARD_DAYS
Z["missed"] = DISCOUNT * Z.amount * Z.late10
Z["interest"] = Z.amount * INTEREST * (d - SLA_HARD_DAYS).clip(lower=0) / 365
Z["band"] = pd.cut(Z.po_value, BANDS, labels=BAND_LABELS, right=True)

A = Z[Z.po_value > CUT].copy()          # stays manual
B = Z[Z.po_value <= CUT]                # handled like ZJD1 in the scenario
assert len(A) + len(B) == len(Z)


def summary(x):
    return pd.Series({"invoice_proxies": len(x), "pos": x.index.get_level_values("PO_ID").nunique(),
                      "suppliers": x.supplier.nunique(), "amount_mnok": x.amount.sum() / 1e6,
                      "median_days": x.time_to_post_days.median(), "late10": int(x.late10.sum()),
                      "late10_share": x.late10.mean(), "late30": int(x.late30.sum()),
                      "missed_mnok": x.missed.sum() / 1e6, "interest_mnok": x.interest.sum() / 1e6,
                      f"posted_by_{FOCUS}_share": (x.poster == FOCUS).mean()})


overview = pd.DataFrame({"All 35 suppliers": summary(Z), f"PO value <= {CUT / 1e6:g}M (cut away)": summary(B),
                         f"PO value > {CUT / 1e6:g}M (stays manual)": summary(A)}).T
by_band = A.groupby("band", observed=False).apply(summary)
by_sup = A.groupby(A.supplier.rename("Supplier")).apply(summary).sort_values("amount_mnok", ascending=False)
by_poster = A.groupby(A.poster.rename("Poster")).apply(summary).sort_values("invoice_proxies", ascending=False)
with pd.ExcelWriter(XLSX) as w:
    overview.to_excel(w, sheet_name="Overview")
    by_band.to_excel(w, sheet_name="By PO value band")
    by_sup.to_excel(w, sheet_name="By supplier")
    by_poster.to_excel(w, sheet_name="By poster")
    A.reset_index()[["PO_ID", "SAP_PO_No", "supplier", "po_value", "amount", "start", "first_post", "time_to_post_days",
                     "late10", "late30", "missed", "interest", "poster", "band"]].sort_values(
        ["po_value", "PO_ID"], ascending=[False, True]).to_excel(w, sheet_name="Invoice proxies", index=False)

# ---------------------------------------------------------------- chart: 2 x 2
fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=200, gridspec_kw={"hspace": 0.55, "wspace": 0.42})
fig.patch.set_facecolor(SURFACE)
(a1, a2), (a3, a4) = axes


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.tick_params(axis="both", length=0, colors=INK2, labelsize=9)
    for s in ax.spines.values():
        s.set_visible(False)


def hbar(ax, labels, values, colors, texts, title, xlabel):
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, height=0.64, color=colors, zorder=2)
    for yi, v, t in zip(y, values, texts):
        ax.text(v + max(values) * 0.015, yi, t, va="center", ha="left", fontsize=8.5, color=INK)
    ax.set_yticks(y, labels, fontsize=9.5, color=INK)
    ax.set_xlim(0, max(values) * 1.75)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9.5)
    ax.set_title(title, loc="left", fontsize=11.5, color=INK, pad=8)
    ax.grid(axis="x", color=GRID, zorder=0)
    style(ax)


# A: by PO value band
bb = by_band
hbar(a1, [f"{b}" for b in bb.index], bb.invoice_proxies.values, [BLUE] * len(bb),
     [f"{int(r.invoice_proxies):,} on {int(r.pos)} POs · {r.amount_mnok:,.0f}M NOK · {r.missed_mnok:.2f}M missed"
      for _, r in bb.iterrows()],
     "PO value of the remaining POs (NOK)", "Invoice proxies")

# B: days to first posting
cap = 45
days = A.time_to_post_days.clip(upper=cap)
counts = np.bincount(np.minimum(np.floor(days).astype(int), cap), minlength=cap + 1)
x = np.arange(cap + 1)
cols = [BLUE if k < SLA_SOFT_DAYS else ORANGE if k < SLA_HARD_DAYS else RED for k in x]
a2.bar(x + 0.5, counts, width=0.8, color=cols, zorder=2)
for xv in (SLA_SOFT_DAYS, SLA_HARD_DAYS):
    a2.axvline(xv, color=INK2, lw=1, ls="--", zorder=3)
a2.set_xlim(0, cap + 1.2)
a2.set_xticks([0, 5, 10, 15, 20, 25, 30, 35, 40, cap + 0.5], ["0", "5", "10", "15", "20", "25", "30", "35", "40", f"{cap}+"])
a2.set_ylim(0, counts.max() * 1.45)
a2.set_yticks(np.arange(0, counts.max() + 1, 100))
a2.grid(axis="y", color=GRID, zorder=0)
a2.set_xlabel("Days to first posting", color=INK2, fontsize=9.5)
a2.set_ylabel("Invoice proxies", color=INK2, fontsize=9.5)
a2.set_title("How late they are posted today", loc="left", fontsize=11.5, color=INK, pad=8)
ov = overview.iloc[2]
for xpos, lab, col in ((5, f"≤ 10 days\n{int(ov.invoice_proxies - ov.late10):,} ({1 - ov.late10_share:.0%})", BLUE),
                       (20, f"10–30 days\n{int(ov.late10 - ov.late30):,} ({(ov.late10 - ov.late30) / ov.invoice_proxies:.0%})", ORANGE),
                       (38, f"> 30 days\n{int(ov.late30):,} ({ov.late30 / ov.invoice_proxies:.1%})", RED)):
    a2.text(xpos, counts.max() * 1.4, lab, ha="center", va="top", fontsize=9, color=INK, linespacing=1.3,
            bbox=dict(boxstyle="round,pad=0.35", fc=SURFACE, ec=col, lw=1.1))
style(a2)

# C: by supplier
top = by_sup.head(TOP)
rest = by_sup.iloc[TOP:]
sup_labels = list(top.index) + ([f"Other {len(rest)} suppliers"] if len(rest) else [])
sup_vals = list(top.amount_mnok) + ([rest.amount_mnok.sum()] if len(rest) else [])
sup_txt = [f"{v:,.0f}M · {int(r.invoice_proxies):,} invoices · {r.missed_mnok:.2f}M missed" for v, (_, r) in zip(top.amount_mnok, top.iterrows())]
if len(rest):
    sup_txt.append(f"{rest.amount_mnok.sum():,.0f}M · {int(rest.invoice_proxies.sum()):,} invoices · {rest.missed_mnok.sum():.2f}M missed")
hbar(a3, sup_labels, sup_vals, [BLUE] * len(top) + [GREY] * (len(sup_vals) - len(top)), sup_txt,
     "Suppliers (by amount)", "Amount (M NOK)")

# D: by poster
pp = by_poster
top_p = pp.head(5)
rest_p = pp.iloc[5:]
p_labels = list(top_p.index) + ([f"Other {len(rest_p)} users"] if len(rest_p) else [])
p_vals = list(top_p.missed_mnok) + ([rest_p.missed_mnok.sum()] if len(rest_p) else [])
p_n = list(top_p.invoice_proxies) + ([rest_p.invoice_proxies.sum()] if len(rest_p) else [])
p_late = list(top_p.late10_share) + ([(rest_p.late10.sum() / rest_p.invoice_proxies.sum()) if len(rest_p) else 0])
hbar(a4, p_labels, p_vals, [ORANGE if l == FOCUS else BLUE for l in p_labels[:len(top_p)]] + [GREY] * (len(p_vals) - len(top_p)),
     [f"{v:.2f}M · {int(n):,} invoices · {s:.0%} late" for v, n, s in zip(p_vals, p_n, p_late)],
     "Who posts them (missed discount)", "Missed 2% discount (M NOK)")

fig.suptitle(f"ZNB1 always-manual suppliers: what stays manual above a {CUT / 1e6:g}M NOK PO-value cut",
             x=0.02, ha="left", y=0.985, fontsize=15, color=INK)
fig.text(0.02, 0.935,
         f"{int(ov.invoice_proxies):,} of {len(Z):,} invoices ({ov.invoice_proxies / len(Z):.0%}) on {int(ov.pos)} of "
         f"{Z.index.get_level_values('PO_ID').nunique()} POs stay manual · {ov.amount_mnok:,.0f}M of "
         f"{Z.amount.sum() / 1e6:,.0f}M NOK ({ov.amount_mnok / (Z.amount.sum() / 1e6):.0%}) · "
         f"{int(ov.late10):,} posted > 10 days ({ov.late10_share:.0%}) · {ov.missed_mnok:.2f}M of "
         f"{Z.missed.sum() / 1e6:.2f}M NOK missed discount ({ov.missed_mnok / (Z.missed.sum() / 1e6):.0%})",
         fontsize=10, color=INK2, ha="left")
fig.text(0.02, 0.012,
         f"Scope: the {len(s100)} ZNB1 suppliers with 100% of invoices routed to CTRL1 at initial validation, posted invoices "
         "(deleted and never-posted left out), as in report_po_cut_gbr.html. Today's handling, no model.\n"
         "Clock: goods receipt (WAIT_GR) if present, else arrival in AP, to first posting. Amount = PO value split evenly "
         "over the PO's posted invoices. Missed discount = 2% of invoices posted after > 10 days.",
         fontsize=7.5, color=INK2, ha="left")
fig.subplots_adjust(left=0.1, right=0.98, top=0.84, bottom=0.1)
fig.savefig(PNG, facecolor=SURFACE)
plt.close(fig)

pd.set_option("display.width", 250)
print(overview.round(3).to_string())
print(by_band.round(2).to_string())
print(by_sup.head(10).round(2).to_string())
print(by_poster.round(2).to_string())
