"""ZNB1: which clerk posts which supplier's invoices (manual postings), and how the clerks' portfolios differ.

A supplier counts as "dedicated" to a clerk when that clerk does >= 80% of its manual postings (suppliers with >= 20).
Queue age = time from entering the manual queue (CTRL1) to the manual posting. Late = pipeline clock > 10 days.
Run from the repo root:  PYTHONPATH=. conda run -n datasci python -W ignore notes/znb1_supplier_clerk_assignment.py
"""
from pathlib import Path

import pandas as pd

from common import SLA_SOFT_DAYS, load_cases, load_events, load_invoices

NOTES = Path(__file__).resolve().parent
K = ["PO_ID", "SAP_PO_No"]
MIN_POSTINGS, DEDICATED = 20, 0.8

ev = load_events()
z = ev[ev.SAP_PO_Type_Code == "ZNB1"]
inv = load_invoices().set_index(K)
po_value = load_cases().set_index("PO_ID").invoice_value_nok

t = z[z.Activity_ID == "Posted manually"].sort_values("ts").drop_duplicates(K).set_index(K)[["User_ID", "Supplier_ID", "ts"]]
t["queue_age_days"] = (t.ts - z[z.SAP_Next_Processing_Type == "CTRL1"].groupby(K).ts.min().reindex(t.index)) \
    .dt.total_seconds() / 86400
t["direct_to_queue"] = z[(z.Activity_ID == "Initial validation") & (z.SAP_Next_Processing_Type == "CTRL1")] \
    .groupby(K).size().reindex(t.index).notna()
t["late"] = inv.time_to_post_days.reindex(t.index) > SLA_SOFT_DAYS
t["po_value_nok"] = po_value.reindex(t.index.get_level_values("PO_ID")).values

g = t.groupby("Supplier_ID")
sup = pd.DataFrame({
    "manual_postings": g.size(),
    "main_clerk": g.User_ID.agg(lambda u: u.value_counts().index[0]),
    "main_clerk_pct": g.User_ID.agg(lambda u: 100 * u.value_counts(normalize=True).iloc[0]),
    "second_clerk": g.User_ID.agg(lambda u: u.value_counts().index[1] if u.nunique() > 1 else None),
    "clerks": g.User_ID.nunique(),
    "median_queue_age_days": g.queue_age_days.median(),
    "late_pct": 100 * g.late.mean(),
    "direct_to_queue_pct": 100 * g.direct_to_queue.mean(),
    "median_po_value_nok": t.reset_index().drop_duplicates("PO_ID").groupby("Supplier_ID").po_value_nok.median(),
})
sup["dedicated"] = (sup.manual_postings >= MIN_POSTINGS) & (sup.main_clerk_pct >= 100 * DEDICATED)
sup = sup.sort_values(["main_clerk", "manual_postings"], ascending=[True, False]).round(1)
sup.to_csv(NOTES / "znb1_supplier_main_clerk.csv")

c = t.groupby("User_ID")
ded = sup[sup.dedicated]
clerks = pd.DataFrame({
    "manual_postings": c.size(),
    "suppliers": c.Supplier_ID.nunique(),
    "dedicated_suppliers": ded.groupby("main_clerk").size(),
    # share of the clerk's postings that are at suppliers dedicated to that same clerk
    "postings_at_own_dedicated_pct": 100 * t[t.Supplier_ID.map(ded.main_clerk) == t.User_ID].groupby("User_ID").size()
                                     / c.size(),
    "median_queue_age_days": c.queue_age_days.median(),
    "late_pct": 100 * c.late.mean(),
    "direct_to_queue_pct": 100 * c.direct_to_queue.mean(),
    "median_po_value_nok": t.reset_index().drop_duplicates(["User_ID", "PO_ID"]).groupby("User_ID").po_value_nok.median(),
    "dedicated_list": ded.groupby("main_clerk").apply(
        lambda x: ", ".join(f"{s}({int(r.manual_postings)})" for s, r in x.sort_values("manual_postings",
                                                                                      ascending=False).iterrows())),
}).sort_values("manual_postings", ascending=False)
clerks = clerks[clerks.manual_postings >= 50].round(1)
clerks.to_csv(NOTES / "znb1_clerk_portfolios.csv")

big = sup[sup.manual_postings >= MIN_POSTINGS]
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 70)
print(f"suppliers with >= {MIN_POSTINGS} manual postings: {len(big)}; dedicated (>= {DEDICATED:.0%} one clerk): "
      f"{big.dedicated.sum()}; median main-clerk share {big.main_clerk_pct.median():.0f}%; postings at dedicated "
      f"suppliers {100 * t.Supplier_ID.isin(ded.index).mean():.0f}%")
print(clerks.drop(columns="dedicated_list").to_string())
print("backup (second clerk) at shared suppliers:", big[~big.dedicated].second_clerk.value_counts().head(5).to_dict())
print("backup at dedicated suppliers:", big[big.dedicated].second_clerk.value_counts().head(5).to_dict())

# --- Overview figure: supplier x clerk heatmap + clerk summary ---------------------------
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from common import use_invoice_proxy_labels

use_invoice_proxy_labels()  # chart text: invoice -> invoice proxy

INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dc"
CLERKS = list(clerks.index)                       # clerks with >= 50 postings, by volume
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED})
big = big.copy()
big["order"] = big.main_clerk.map({u: i for i, u in enumerate(CLERKS)}).fillna(len(CLERKS))
big = big.sort_values(["order", "manual_postings"], ascending=[True, False])
share = pd.crosstab(t.Supplier_ID, t.User_ID, normalize="index").reindex(index=big.index, columns=CLERKS, fill_value=0)

fig = plt.figure(figsize=(16, 21))
gs = GridSpec(2, 2, figure=fig, width_ratios=[1.35, 1], height_ratios=[1, 0.28], hspace=0.12, wspace=0.35)

ax = fig.add_subplot(gs[:, 0])
ax.imshow(share.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
for r in range(share.shape[0]):
    for c in range(share.shape[1]):
        v = share.iat[r, c]
        if v >= 0.05:
            ax.text(c, r, f"{100 * v:.0f}", ha="center", va="center", fontsize=7,
                    color="white" if v > 0.6 else INK)
labels = [f"{s.replace('Supplier', 'S')}  ({int(n)})" for s, n in zip(big.index, big.manual_postings)]
ax.set_yticks(range(len(big)), labels, fontsize=7.5)
ax.set_xticks(range(len(CLERKS)), CLERKS, rotation=45, ha="left")
ax.xaxis.tick_top()
# separators between portfolio blocks
bounds = np.flatnonzero(np.diff(big.order.values)) + 0.5
for b in bounds:
    ax.axhline(b, color=INK, linewidth=0.8)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title("Share of each supplier's manual ZNB1 postings by clerk (%)\nsuppliers with >= 20 postings, grouped by main "
             "clerk; (n) = postings", loc="left", color=INK, fontweight="bold", fontsize=11, pad=60)

# right top: postings per clerk, own dedicated suppliers vs other work
ax = fig.add_subplot(gs[0, 1])
own = t[t.Supplier_ID.map(ded.main_clerk) == t.User_ID].groupby("User_ID").size().reindex(CLERKS, fill_value=0)
other = clerks.manual_postings.reindex(CLERKS) - own
y = np.arange(len(CLERKS))[::-1]
ax.barh(y, own, color="#2a78d6", height=0.6, label="At own dedicated suppliers")
ax.barh(y, other, left=own, color="#b7d3f6", height=0.6, label="Other suppliers")
for yi, u in zip(y, CLERKS):
    r = clerks.loc[u]
    ax.text(own[u] + other[u] + 25, yi,
            f"{r.median_queue_age_days:.1f} d in queue · {r.late_pct:.0f}% late · {int(r.suppliers)} suppliers",
            va="center", fontsize=8.5, color=INK)
ax.set_yticks(y, CLERKS)
ax.set_xlim(0, clerks.manual_postings.max() * 1.9)
ax.set_xlabel("Manual ZNB1 postings")
ax.set_title("Workload and speed per clerk", loc="left", color=INK, fontweight="bold", fontsize=11)
ax.legend(frameon=False, loc="lower right", fontsize=8.5)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", color=GRID); ax.set_axisbelow(True)

# right bottom: key facts as text
ax = fig.add_subplot(gs[1, 1]); ax.axis("off")
facts = [
    f"{big.dedicated.sum()} of {len(big)} suppliers have one clerk doing >= 80% of postings",
    f"Median share of the main clerk: {big.main_clerk_pct.median():.0f}%",
    f"{100 * t.Supplier_ID.isin(ded.index).mean():.0f}% of manual ZNB1 postings are at dedicated suppliers",
    "Portfolio owners: User1341, User394, User1564",
    "Floaters / backups: User400 (82 suppliers), User1331, User712",
    "Speed is a clerk trait: 0.2 d (User1177) to 8.2 d (User1341)",
]
ax.text(0, 1, "Key facts", fontsize=11, fontweight="bold", color=INK, va="top")
for i, f in enumerate(facts):
    ax.text(0, 0.86 - i * 0.14, "–  " + f, fontsize=9.5, color=INK, va="top")

fig.savefig(NOTES / "figures" / "znb1_supplier_clerk_assignment.png", dpi=130, bbox_inches="tight")
