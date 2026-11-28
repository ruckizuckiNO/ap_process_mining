"""Profile of User1341, the clerk who posts most ZNB1 invoices: suppliers, PO values, speed, and working pattern.

Uses the pipeline outputs (run 01_prepare.py first); clock as in the pipeline (ZNB1 from goods receipt where a WAIT_GR
step exists). Queue age = time from entering the manual queue (CTRL1) to the manual posting.
Run from the repo root:  PYTHONPATH=. conda run -n datasci python -W ignore notes/user1341_profile.py
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
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]

ev = load_events()
inv = load_invoices().set_index(K)
cases = load_cases().set_index("PO_ID")
posted = ev[ev.Activity_ID == "Posted manually"].sort_values("ts").drop_duplicates(K).set_index(K)
inv["poster"] = posted.User_ID.reindex(inv.index)
inv["supplier"] = ev.groupby(K).Supplier_ID.first().reindex(inv.index)
inv["po_value_nok"] = cases.invoice_value_nok.reindex(inv.index.get_level_values("PO_ID")).values
inv["value_nok"] = inv.po_value_nok / inv[inv.posted].groupby(level="PO_ID").posted.transform("size")
inv["late"] = inv.posted & (inv.time_to_post_days > SLA_SOFT_DAYS)
inv["missed_discount_nok"] = inv.value_nok * 0.02 * inv.late
queue_in = ev[ev.SAP_Next_Processing_Type == "CTRL1"].groupby(K).ts.min()
inv["queue_in"] = queue_in.reindex(inv.index)
inv["posted_at"] = posted.ts.reindex(inv.index)
inv["queue_age_days"] = (inv.posted_at - inv.queue_in).dt.total_seconds() / 86400
man = inv[inv.poster.notna()].copy()
mine = man[man.poster == U]

# --- Suppliers -------------------------------------------------------------------
g = mine.groupby("supplier")
sup = pd.DataFrame({
    "po_types": g.po_type.agg(lambda t: "/".join(sorted(set(t)))),
    "invoices": g.size(),
    "pos": g.apply(lambda x: x.index.get_level_values("PO_ID").nunique()),
    "share_of_supplier_manual_pct": 100 * man.groupby("supplier").poster.apply(lambda p: (p == U).mean()),
    "median_days_to_post": g.time_to_post_days.median(),
    "median_queue_age_days": g.queue_age_days.median(),
    "late_pct": 100 * g.late.mean(),
    "missed_discount_nok": g.missed_discount_nok.sum(),
}).dropna(subset=["invoices"])
pos = mine.reset_index().drop_duplicates("PO_ID").groupby("supplier").po_value_nok
sup["median_po_value_nok"] = pos.median()
sup["max_po_value_nok"] = pos.max()
sup["po_value_nok"] = pos.sum()
sup = sup.sort_values("invoices", ascending=False).round(1)
sup.to_csv(NOTES / "user1341_suppliers.csv")

# --- PO values: User1341 vs other clerks (manually posted POs) ----------------------
rows = []
for t in ["ZNB1", "ZJD1"]:
    x = man[man.po_type == t].reset_index().drop_duplicates("PO_ID")
    for who, y in [(U, x[x.poster == U]), ("Other clerks", x[x.poster != U])]:
        q = y.po_value_nok.quantile([0.25, 0.5, 0.75, 0.9])
        rows.append({"po_type": t, "who": who, "pos": len(y), "p25": q[0.25], "median": q[0.5], "p75": q[0.75],
                     "p90": q[0.9], "mean": y.po_value_nok.mean()})
values = pd.DataFrame(rows).round(0)
values.to_csv(NOTES / "user1341_po_values.csv", index=False)

# --- Working pattern ----------------------------------------------------------------
p = ev[(ev.User_ID == U) & (ev.Activity_ID == "Posted manually")]
days = p.ts.dt.date
per_day = p.groupby(days).size()
others = man[(man.poster != U) & man.queue_age_days.notna()]
mq = mine[mine.queue_age_days.notna()]
pattern = pd.Series({
    "first_active_day": str(days.min()), "last_active_day": str(days.max()), "active_days": days.nunique(),
    "postings": len(p), "postings_per_active_day_mean": per_day.mean(), "postings_per_active_day_max": per_day.max(),
    "median_minutes_between_postings": p.sort_values("ts").groupby(days).ts.diff().dt.total_seconds().median() / 60,
    "share_posted_13_to_19h_pct": 100 * p.ts.dt.hour.between(13, 18).mean(),
    "queue_age_median_days": mq.queue_age_days.median(),
    "queue_age_under_5d_pct": 100 * (mq.queue_age_days < 5).mean(),
    "others_queue_age_median_days": others.queue_age_days.median(),
    "others_queue_age_under_5d_pct": 100 * (others.queue_age_days < 5).mean(),
    "znb1_manual_postings_share_pct": 100 * (man[man.po_type == "ZNB1"].poster == U).mean(),
    "znb1_late_share_pct": 100 * mine[mine.po_type == "ZNB1"].late.sum() / inv[inv.po_type == "ZNB1"].late.sum(),
    "znb1_missed_discount_share_pct": 100 * mine[mine.po_type == "ZNB1"].missed_discount_nok.sum()
                                      / inv[inv.po_type == "ZNB1"].missed_discount_nok.sum(),
})
pattern.to_csv(NOTES / "user1341_pattern.csv", header=["value"])
week = pd.crosstab(mq.queue_in.dt.dayofweek, mq.posted_at.dt.dayofweek).reindex(index=range(5), columns=range(5),
                                                                              fill_value=0)

# --- Figure ---------------------------------------------------------------------------
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dc"
CU, CO = "#2a78d6", "#8a8980"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 4, figsize=(23, 5.8), gridspec_kw={"width_ratios": [1.3, 0.85, 1, 0.8]})

ax = axes[0]
t = sup[sup.invoices >= 20].iloc[::-1]
ax.barh(t.index, t.invoices, color=CU, height=0.65)
for y, (s, r) in enumerate(t.iterrows()):
    ax.text(r.invoices + 12, y, f"{r.share_of_supplier_manual_pct:.0f}% of its manual work · median PO "
            f"{r.median_po_value_nok / 1e6:.2f}M · {r.late_pct:.0f}% late", va="center", fontsize=8.5, color=INK)
ax.set_xlim(0, t.invoices.max() * 2.1)
ax.set_xlabel("Invoices posted by User1341")
ax.set_title("A fixed portfolio of suppliers", loc="left", color=INK, fontweight="bold")
ax.grid(axis="x", color=GRID); ax.set_axisbelow(True)

ax = axes[1]
v = values[values.po_type == "ZNB1"].set_index("who")[["p25", "median", "p75", "p90"]] / 1e6
x = np.arange(4)
for off, who, c in [(-0.2, U, CU), (0.2, "Other clerks", CO)]:
    ax.bar(x + off, v.loc[who], width=0.38, color=c, label=who)
    for xi, val in zip(x + off, v.loc[who]):
        ax.text(xi, val + 0.08, f"{val:.2f}", ha="center", fontsize=8.5, color=INK)
ax.set_xticks(x, ["25th pct", "Median", "75th pct", "90th pct"])
ax.set_ylabel("PO value (M NOK)")
ax.set_title("ZNB1 PO values: smaller median, bigger tail", loc="left", color=INK, fontweight="bold")
ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)

ax = axes[2]
bins = np.arange(0, 22, 1)
for d, lab, c in [(others.queue_age_days, f"Other clerks (median {others.queue_age_days.median():.1f} d)", CO),
                  (mq.queue_age_days, f"User1341 (median {mq.queue_age_days.median():.1f} d)", CU)]:
    ax.hist(d.clip(upper=21), bins=bins, density=True, histtype="step", linewidth=2, color=c, label=lab)
ax.axvline(7, color=INK, linewidth=1, linestyle=":")
ax.text(7.2, ax.get_ylim()[1] * 0.6, "one week", fontsize=9, color=INK)
ax.set_xlabel("Days in the manual queue before posting")
ax.set_ylabel("Share of invoices")
ax.set_title("User1341 posts invoices about a week old", loc="left", color=INK, fontweight="bold")
ax.legend(frameon=False, fontsize=9, loc="upper right")
ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)

ax = axes[3]
share = week.div(week.sum(axis=1), axis=0)
ax.imshow(share.values, cmap="Blues", vmin=0, vmax=share.values.max())
for r in range(5):
    for c in range(5):
        ax.text(c, r, f"{100 * share.iat[r, c]:.0f}%", ha="center", va="center", fontsize=9,
                color="white" if share.iat[r, c] > share.values.max() * 0.6 else INK)
ax.set_xticks(range(5), DAYS); ax.set_yticks(range(5), DAYS)
ax.set_xlabel("Weekday posted")
ax.set_ylabel("Weekday the invoice entered the queue")
ax.set_title("Same weekday, one week later", loc="left", color=INK, fontweight="bold")
ax.spines[:].set_visible(False)

fig.tight_layout()
fig.savefig(NOTES / "figures" / "user1341_profile.png", dpi=140)

pd.set_option("display.width", 220)
print(pattern.to_string())
print(values.to_string())
print(sup.to_string())
print(week)
