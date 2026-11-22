"""Does a high invoice amount take longer to post? ZNB1 manual postings, User1341 vs. the other clerks.

Handling time per posting = gap since the same user's previous event on the same day (timestamps mark completion),
kept only when <= BREAK_MIN minutes (longer gaps are breaks / other work, so they say nothing about the invoice).
Queue wait = from the invoice's first CTRL1 routing to its manual posting (days), for contrast.
Invoice value = PO value split evenly over the PO's posted invoices (as in the other notes).
-> notes/figures/znb1_handling_time_vs_value.png (handling time + explanation),
   notes/figures/znb1_handling_and_wait_vs_value.png (handling and queue wait side by side), notes/znb1_handling_time_vs_value.csv
Run from the repo root:  PYTHONPATH=. conda run -n datasci python -W ignore notes/znb1_handling_time_vs_value.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import BOT_USERS, POSTING_ACTIVITIES, load_cases, load_events

NOTES = Path(__file__).resolve().parent
K = ["PO_ID", "SAP_PO_No"]
U = "User1341"
BREAK_MIN = 15
EDGES = [0, 1e5, 5e5, 1.5e6, 5e6, np.inf]
LABELS = ["< 100k", "100k–500k", "500k–1.5M", "1.5M–5M", "> 5M"]
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"

ev = load_events().sort_values(["ts", "Event_ID"])
h = ev[ev.User_ID.notna() & ~ev.User_ID.isin(BOT_USERS)].copy()
h["day"] = h.ts.dt.normalize()
h["gap_min"] = h.groupby(["User_ID", "day"]).ts.diff().dt.total_seconds() / 60

z = ev[ev.SAP_PO_Type_Code == "ZNB1"]
n_posted = z[z.Activity_ID.isin(POSTING_ACTIVITIES)].groupby("PO_ID").SAP_PO_No.nunique()
po_value = load_cases().set_index("PO_ID").invoice_value_nok
qin = z[z.SAP_Next_Processing_Type == "CTRL1"].groupby(K).ts.min()

p = h[(h.Activity_ID == "Posted manually") & (h.SAP_PO_Type_Code == "ZNB1")].drop_duplicates(K).set_index(K)
p["inv_value"] = p.index.get_level_values("PO_ID").map(po_value) / p.index.get_level_values("PO_ID").map(n_posted)
p["who"] = np.where(p.User_ID == U, U, "Other clerks")
p["bin"] = pd.cut(p.inv_value, EDGES, labels=LABELS)
p["wait_days"] = (p.ts - qin.reindex(p.index)).dt.total_seconds() / 86400
hand = p[p.gap_min <= BREAK_MIN]
wait = p[p.wait_days.notna()]


def stats(df, col):
    return (df.groupby(["who", "bin"], observed=False)[col]
            .agg(n="size", q25=lambda s: s.quantile(0.25), median="median", q75=lambda s: s.quantile(0.75))
            .reset_index())


hs, ws = stats(hand, "gap_min"), stats(wait, "wait_days")
out = hs.merge(ws, on=["who", "bin"], suffixes=("_handling_min", "_wait_days"))
out.to_csv(NOTES / "znb1_handling_time_vs_value.csv", index=False)
rho = {w: spearmanr(d.inv_value, d.gap_min) for w, d in hand.groupby("who")}
rho_w = {w: spearmanr(d.inv_value, d.wait_days) for w, d in wait.groupby("who")}

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"]})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 8.8), dpi=200)
fig.patch.set_facecolor(SURFACE)
x = np.arange(len(LABELS))
for ax, s, unit in [(a1, hs, "min"), (a2, ws, "days")]:
    for who, c, dx in [(U, BLUE, -0.12), ("Other clerks", ORANGE, 0.12)]:
        d = s[s.who == who].set_index("bin").reindex(LABELS)
        ax.vlines(x + dx, d.q25, d.q75, color=c, linewidth=2, alpha=0.45, zorder=2)
        ax.plot(x + dx, d["median"], color=c, linewidth=2, marker="o", markersize=8, markeredgecolor=SURFACE,
                markeredgewidth=1.5, zorder=3, label=who)
        for xi, (_, r) in zip(x + dx, d.iterrows()):
            ax.text(xi + (0.06 if dx > 0 else -0.06), r["median"], f"{r['median']:.1f}",
                    ha="left" if dx > 0 else "right", va="center", fontsize=8.5, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=9.5, color=INK)
    ax.set_xlabel("Invoice value (NOK)", color=INK2, fontsize=10)
    ax.set_ylim(0, None)
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(axis="both", length=0, colors=INK2, labelsize=9)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color(GRID)
nU = hs[hs.who == U].set_index("bin")["n"]
a1.set_title("Handling time per invoice (minutes)", loc="left", fontsize=11, fontweight="bold", color=INK, pad=24)
a1.text(0, 1.015, f"Spearman ρ with value: {U} {rho[U].statistic:.2f}, others {rho['Other clerks'].statistic:.2f}",
        transform=a1.transAxes, fontsize=9, color=INK2)
a2.set_title("Waiting time in the queue before posting (days)", loc="left", fontsize=11, fontweight="bold", color=INK,
              pad=24)
a2.text(0, 1.015, f"Spearman ρ with value: {U} {rho_w[U].statistic:.2f}, others {rho_w['Other clerks'].statistic:.2f}",
        transform=a2.transAxes, fontsize=9, color=INK2)
nW = ws[(ws.who == U) & (ws.bin == LABELS[-1])]
if len(nW) and nW["n"].iat[0] < 50:
    a2.annotate(f"only {nW['n'].iat[0]} invoices", xy=(x[-1] - 0.12, nW["median"].iat[0]),
                xytext=(x[-1] - 0.9, ws.q75.max() * 0.25), fontsize=8.5, color=INK2,
                arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.8))
a1.legend(loc="upper left", frameon=False, fontsize=9.5, labelcolor=INK2)
a1.set_ylim(0, hs.q75.max() * 1.15)
a2.set_ylim(0, ws.q75.max() * 1.15)

fig.suptitle("The invoice amount barely changes posting or waiting time – the clerk does", x=0.05, ha="left", y=0.982,
             fontsize=15, fontweight="bold", color=INK)
fig.text(0.05, 0.948, "ZNB1 manual postings · dot = median, bar = middle 50% of invoices · "
         f"{len(hand):,} postings with a measurable handling time, {len(wait):,} with a queue time",
         fontsize=9.5, color=INK2, ha="left", va="top")

# --- how to read the chart (numbers taken from the data) ---
hu = hs[hs.who == U].set_index("bin")
ho = hs[hs.who == "Other clerks"].set_index("bin")
wu = ws[ws.who == U].set_index("bin")
wo = ws[ws.who == "Other clerks"].set_index("bin")
main = LABELS[:-1]  # bins with enough invoices for both groups
left_txt = (
    "• Each dot is the median handling time for invoices in that value range; the vertical bar spans\n"
    "  the middle 50% of invoices (25th to 75th percentile). Blue = User1341, orange = other clerks.\n"
    f"• A flat line means value does not matter. User1341 goes from {hu.loc[LABELS[0], 'median']:.1f} to "
    f"{hu.loc[LABELS[-1], 'median']:.1f} min, others stay around\n"
    f"  {ho.loc[main, 'median'].min():.1f}–{ho.loc[main, 'median'].max():.1f} min: at most about half a minute "
    "more for a large invoice.\n"
    "• The bars overlap almost completely, so a large invoice is not reliably slower than a small one.\n"
    f"• Spearman ρ measures how strongly time rises with value (0 = not at all, 1 = perfectly):\n"
    f"  {rho[U].statistic:.2f} for User1341 and {rho['Other clerks'].statistic:.2f} for the others – practically none."
)
right_txt = (
    "• Same layout, but days an invoice waits in the manual queue before it is posted.\n"
    f"• The two lines are far apart: User1341's invoices wait about {wu.loc[main, 'median'].median():.0f} days, "
    f"other clerks' about {wo.loc[main, 'median'].median():.1f}\n"
    "  days, and both lines are flat across value ranges. Waiting depends on who posts, not on the amount.\n"
    f"• The drop for User1341 above 5M rests on only {int(wu.loc[LABELS[-1], 'n'])} invoices: the very largest "
    "seem to be picked\n  out and posted at once, but the group is too small to generalise.\n"
    "• So high-value invoices wait longer overall only because most of them go to User1341."
)
fig.text(0.05, 0.215, "How to read the left panel", fontsize=9.5, fontweight="bold", color=INK, va="top")
fig.text(0.05, 0.19, left_txt, fontsize=8.6, color=INK, ha="left", va="top", linespacing=1.45)
fig.text(0.555, 0.215, "How to read the right panel", fontsize=9.5, fontweight="bold", color=INK, va="top")
fig.text(0.555, 0.19, right_txt, fontsize=8.6, color=INK, ha="left", va="top", linespacing=1.45)
fig.add_artist(plt.Line2D([0.05, 0.98], [0.232, 0.232], color=GRID, linewidth=0.8))
fig.text(0.05, 0.008, f"Handling time = gap since the clerk's previous event that day, only gaps ≤ {BREAK_MIN} min "
         "(timestamps mark completion; a lower bound). Queue wait = first CTRL1 routing to manual posting. "
         "Invoice value = PO value / posted invoices.", fontsize=7.5, color=INK2)
fig.subplots_adjust(left=0.05, right=0.98, top=0.84, bottom=0.33, wspace=0.18)
fig.savefig(NOTES / "figures" / "znb1_handling_and_wait_vs_value.png", facecolor=SURFACE)
plt.close(fig)

# --- main figure: handling time only, explanation on the right ---
fig = plt.figure(figsize=(13.5, 6.6), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax = fig.add_axes([0.055, 0.14, 0.53, 0.68])
for who, c, dx in [(U, BLUE, -0.12), ("Other clerks", ORANGE, 0.12)]:
    d = hs[hs.who == who].set_index("bin").reindex(LABELS)
    ax.vlines(x + dx, d.q25, d.q75, color=c, linewidth=2, alpha=0.45, zorder=2)
    ax.plot(x + dx, d["median"], color=c, linewidth=2, marker="o", markersize=8, markeredgecolor=SURFACE,
            markeredgewidth=1.5, zorder=3, label=who)
    for xi, (_, r) in zip(x + dx, d.iterrows()):
        ax.text(xi + (0.06 if dx > 0 else -0.06), r["median"], f"{r['median']:.1f}",
                ha="left" if dx > 0 else "right", va="center", fontsize=8.5, color=INK)
ax.set_xticks(x)
ax.set_xticklabels([f"{l}\nn = {int(hu.loc[l, 'n'])} / {int(ho.loc[l, 'n'])}" for l in LABELS], fontsize=9,
                   color=INK)
ax.set_xlabel("Invoice value (NOK) · n = invoices User1341 / other clerks", color=INK2, fontsize=9.5)
ax.set_ylabel("Minutes per invoice", color=INK2, fontsize=9.5)
ax.set_ylim(0, hs.q75.max() * 1.15)
ax.set_facecolor(SURFACE)
ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
ax.tick_params(axis="both", length=0, colors=INK2, labelsize=9)
for sp in ax.spines.values():
    sp.set_visible(False)
ax.spines["bottom"].set_visible(True)
ax.spines["bottom"].set_color(GRID)
ax.legend(loc="upper left", frameon=False, fontsize=9.5, labelcolor=INK2)
ax.set_title("Handling time per invoice (minutes)", loc="left", fontsize=11, fontweight="bold", color=INK, pad=10)

fig.suptitle("A high invoice amount barely changes how long posting takes", x=0.05, ha="left", y=0.965, fontsize=15,
             fontweight="bold", color=INK)
fig.text(0.05, 0.915, f"ZNB1 manual postings · dot = median, bar = middle 50% of invoices · {len(hand):,} postings "
         "with a measurable handling time", fontsize=9.5, color=INK2, ha="left", va="top")
side = (
    "• Each dot is the median handling time for invoices in that\n"
    "  value range; the vertical bar spans the middle 50% of\n"
    "  invoices (25th to 75th percentile).\n"
    "  Blue = User1341, orange = other clerks.\n\n"
    "• A flat line means value does not matter.\n"
    f"  User1341 goes from {hu.loc[LABELS[0], 'median']:.1f} to {hu.loc[LABELS[-1], 'median']:.1f} min, "
    f"others stay around\n  {ho.loc[main, 'median'].min():.1f}–{ho.loc[main, 'median'].max():.1f} min: at most "
    "about half a minute more\n  for a large invoice.\n\n"
    "• The bars overlap almost completely, so a large invoice\n"
    "  is not reliably slower than a small one.\n\n"
    "• Spearman ρ measures how strongly time rises with\n"
    f"  value (0 = not at all, 1 = perfectly): {rho[U].statistic:.2f} for User1341\n"
    f"  and {rho['Other clerks'].statistic:.2f} for the others – practically none.\n\n"
    "• The > 5M group is small, so its higher median for\n"
    "  other clerks is not a reliable trend.\n\n"
    "• Conclusion: the amount is not what makes invoices\n"
    "  slow. Posting takes 1–2 minutes either way; the delay\n"
    "  comes from waiting in the queue."
)
fig.add_artist(plt.Line2D([0.625, 0.625], [0.14, 0.86], color=GRID, linewidth=0.8))
fig.text(0.645, 0.86, "How to read the chart", fontsize=12, fontweight="bold", color=INK, va="top")
fig.text(0.645, 0.805, side, fontsize=10.2, color=INK, ha="left", va="top", linespacing=1.4)
fig.text(0.05, 0.012, f"Handling time = gap since the clerk's previous event that day, only gaps ≤ {BREAK_MIN} min "
         "(timestamps mark completion; a lower bound). Invoice value = PO value / posted invoices.",
         fontsize=7.5, color=INK2)
fig.savefig(NOTES / "figures" / "znb1_handling_time_vs_value.png", facecolor=SURFACE)

print(out.round(2).to_string(index=False))
for w in rho:
    print(w, "handling rho", round(rho[w].statistic, 3), "p", rho[w].pvalue, "| wait rho", round(rho_w[w].statistic, 3),
          "p", rho_w[w].pvalue)
