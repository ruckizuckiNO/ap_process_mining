"""Profile of one clerk's manual postings for one PO type: suppliers, PO values, time bands (<=10 / 10-30 / >30 days),
where the late invoices lose their time, working pattern and exposure.

"Own part" = time from the end of the goods-receipt wait (or from arrival, if there was no wait) to the manual posting,
i.e. the part the clerk controls. "Before" = the goods-receipt wait. Clock as in the pipeline.
Run from the repo root:
  PYTHONPATH=. conda run -n datasci python -W ignore notes/clerk_profile.py User1177 ZNB1
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import SLA_SOFT_DAYS, SLA_HARD_DAYS, load_cases, load_events, load_invoices
from common import use_invoice_proxy_labels

use_invoice_proxy_labels()  # chart text: invoice -> invoice proxy

USER = sys.argv[1] if len(sys.argv) > 1 else "User1177"
PO_TYPE = sys.argv[2] if len(sys.argv) > 2 else "ZNB1"
NOTES = Path(__file__).resolve().parent
PREFIX = f"{USER.lower()}_{PO_TYPE.lower()}"
K = ["PO_ID", "SAP_PO_No"]
BANDS = ["<=10 d", "10-30 d", ">30 d"]
DAYNAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

ev = load_events()
z = ev[ev.SAP_PO_Type_Code == PO_TYPE]
inv = load_invoices().set_index(K)
inv = inv[(inv.po_type == PO_TYPE) & inv.posted].copy()
po_value = load_cases().set_index("PO_ID").invoice_value_nok
inv["value_nok"] = po_value.reindex(inv.index.get_level_values("PO_ID")).values \
    / inv.groupby(level="PO_ID").posted.transform("size")

pm = z[z.Activity_ID == "Posted manually"].sort_values("ts").drop_duplicates(K).set_index(K)
wait = z[z.Activity_ID == "Wait for goods receipt"].sort_values("ts").drop_duplicates(K).set_index(K)
man = inv.join(pm[["User_ID", "Supplier_ID", "ts"]].rename(columns={"ts": "posted_at"}), how="inner")
man["wait_status"] = wait.SAP_Object_Code.reindex(man.index).fillna("no wait (direct to queue)")
man["wait_end"] = wait.ts.reindex(man.index)
man["before_days"] = ((man.wait_end - man.start).dt.total_seconds() / 86400).fillna(0)
man["own_days"] = (man.posted_at - man.wait_end.fillna(man.start)).dt.total_seconds() / 86400
man["days"] = man.time_to_post_days
man["band"] = pd.cut(man.days, [-np.inf, SLA_SOFT_DAYS, SLA_HARD_DAYS, np.inf], labels=BANDS)
man["late"] = man.days > SLA_SOFT_DAYS
man["late_value_nok"] = man.value_nok * man.late
u = man[man.User_ID == USER].copy()
if u.empty:
    sys.exit(f"{USER} has no manual {PO_TYPE} postings")


def bands(x):
    b = x.band.value_counts(normalize=True).reindex(BANDS, fill_value=0)
    return pd.Series({
        "invoices": len(x), "pos": x.index.get_level_values("PO_ID").nunique(),
        "median_po_value_nok": po_value.reindex(x.index.get_level_values("PO_ID").unique()).median(),
        "within_10d_pct": 100 * b[BANDS[0]], "10_to_30d_pct": 100 * b[BANDS[1]], "over_30d_pct": 100 * b[BANDS[2]],
        "median_days": x.days.median(), "median_gr_wait_days": x.before_days.median(),
        "median_own_days": x.own_days.median(),
        "late_value_nok": x.late_value_nok.sum(), "discount_at_stake_nok": 0.02 * x.late_value_nok.sum()})


share = man.groupby("Supplier_ID").User_ID.apply(lambda s: 100 * (s == USER).mean())
sup = u.groupby("Supplier_ID").apply(bands)
sup.insert(2, "share_of_supplier_manual_pct", share.reindex(sup.index))
sup = sup.sort_values("invoices", ascending=False)

views = {
    "all": bands(u).to_frame(USER).T,
    "wait_status": u.groupby("wait_status").apply(bands),
    "month": u.groupby(u.posted_at.dt.to_period("M")).apply(bands),
    "arrival_weekday": u.groupby(u.wait_end.fillna(u.start).dt.dayofweek).apply(bands).rename(
        index=dict(enumerate(DAYNAMES))),
    "supplier": sup,
}
pd.concat(views, names=["view", "group"]).round(1).to_csv(NOTES / f"{PREFIX}_profile.csv")

# PO values vs other clerks (same PO type, manually posted POs)
pos = man.reset_index().drop_duplicates(["PO_ID", "User_ID"])
cmp_ = pd.DataFrame({who: po_value.reindex(x.PO_ID).quantile([0.25, 0.5, 0.75, 0.9]).values
                     for who, x in [(USER, pos[pos.User_ID == USER]), ("Other clerks", pos[pos.User_ID != USER])]},
                    index=["p25", "median", "p75", "p90"])

p = ev[(ev.User_ID == USER) & (ev.Activity_ID == "Posted manually")]
pt = p[p.SAP_PO_Type_Code == PO_TYPE]
pattern = pd.Series({
    "all_types_postings": len(p), "po_types": p.SAP_PO_Type_Code.value_counts().to_dict(),
    "first_day": str(p.ts.dt.date.min()), "last_day": str(p.ts.dt.date.max()), "active_days": p.ts.dt.date.nunique(),
    "postings_per_active_day": round(p.groupby(p.ts.dt.date).size().mean(), 1),
    f"{PO_TYPE}_postings_per_active_day": round(pt.groupby(pt.ts.dt.date).size().mean(), 1),
    "share_posted_07_to_16h_pct": round(100 * p.ts.dt.hour.between(7, 15).mean(), 1),
    "own_part_under_1d_pct": round(100 * (u.own_days < 1).mean(), 1),
    "others_own_part_under_1d_pct": round(100 * (man[man.User_ID != USER].own_days < 1).mean(), 1),
})

# --- Figure ------------------------------------------------------------------------------
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dc"
CB = {BANDS[0]: "#2a78d6", BANDS[1]: "#eb6834", BANDS[2]: "#8a8980"}
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 4, figsize=(24, 6), gridspec_kw={"width_ratios": [1.3, 1, 1, 0.9]})

ax = axes[0]
t = sup[sup.invoices >= 10].head(12).iloc[::-1]
left = np.zeros(len(t))
for b, col in zip(BANDS, ["within_10d_pct", "10_to_30d_pct", "over_30d_pct"]):
    n = t.invoices * t[col] / 100
    ax.barh(t.index, n, left=left, color=CB[b], height=0.65, label=b)
    left += n.values
for y, (s, r) in enumerate(t.iterrows()):
    ax.text(r.invoices + 3, y, f"{r.share_of_supplier_manual_pct:.0f}% of its manual work · median PO "
            f"{r.median_po_value_nok / 1e6:.2f}M · {100 - r.within_10d_pct:.0f}% late", va="center", fontsize=8.5,
            color=INK)
ax.set_xlim(0, t.invoices.max() * 2.2)
ax.set_xlabel(f"{PO_TYPE} invoices posted by {USER}")
ax.set_title(f"{USER}'s {PO_TYPE} suppliers", loc="left", color=INK, fontweight="bold")
ax.legend(frameon=False, fontsize=9, loc="lower right")
ax.grid(axis="x", color=GRID); ax.set_axisbelow(True)

ax = axes[1]
ws = views["wait_status"]
order = [w for w in ["no wait (direct to queue)", "WAIT_GR", "END_WAIT_GR"] if w in ws.index]
ws = ws.reindex(order)
bottom = np.zeros(len(ws))
for b, col in zip(BANDS, ["within_10d_pct", "10_to_30d_pct", "over_30d_pct"]):
    n = ws.invoices * ws[col] / 100
    ax.bar(range(len(ws)), n, bottom=bottom, color=CB[b], width=0.6)
    bottom += n.values
for i, (w, r) in enumerate(ws.iterrows()):
    ax.text(i, r.invoices + 4, f"GR wait {r.median_gr_wait_days:.0f} d · own {r.median_own_days:.1f} d",
            ha="center", fontsize=8.5, color=INK)
ax.set_xticks(range(len(ws)), [w.replace(" (direct to queue)", "\n(direct to queue)").replace("_GR", "_GR\n(GR booked)")
                               if w == "WAIT_GR" else w.replace(" (direct to queue)", "\n(direct to queue)")
                               .replace("END_WAIT_GR", "END_WAIT_GR\n(no GR)") for w in ws.index])
ax.set_ylabel("Invoices")
ax.set_title("Where the time goes: by goods-receipt wait", loc="left", color=INK, fontweight="bold")
ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)

ax = axes[2]
bins = np.arange(0, 31, 1)
for d, lab, c in [(man[man.User_ID != USER].own_days, "Other clerks", "#8a8980"), (u.own_days, USER, "#2a78d6")]:
    ax.hist(d.clip(upper=30), bins=bins, density=True, histtype="step", linewidth=2, color=c,
            label=f"{lab} ({100 * (d < 1).mean():.0f}% within 1 day)")
ax.set_xlabel("Own part: days from end of wait / arrival to posting")
ax.set_ylabel("Share of invoices")
ax.set_title(f"{USER}'s own part: {100 * (u.own_days < 1).mean():.0f}% within a day, "
             f"{100 * (u.own_days > 10).mean():.0f}% over 10 days", loc="left", color=INK, fontweight="bold")
ax.legend(frameon=False, fontsize=9, loc="upper right")
ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)

ax = axes[3]
m = views["month"]
labels = [p_.strftime("%b") for p_ in m.index]
bottom = np.zeros(len(m))
for b, col in zip(BANDS, ["within_10d_pct", "10_to_30d_pct", "over_30d_pct"]):
    ax.bar(labels, m[col], bottom=bottom, color=CB[b], width=0.65)
    bottom += m[col].values
for x, v, n in zip(labels, m.within_10d_pct, m.invoices):
    ax.text(x, 101, f"n={int(n)}", ha="center", fontsize=8, color=MUTED)
    ax.text(x, v / 2, f"{v:.0f}%", ha="center", va="center", fontsize=9, color="white")
ax.set_ylim(0, 108)
ax.set_ylabel("Share of invoices (%)")
ax.set_title("By month posted", loc="left", color=INK, fontweight="bold")
fig.tight_layout()
fig.savefig(NOTES / "figures" / f"{PREFIX}_profile.png", dpi=140)

pd.set_option("display.width", 230)
for k, v in views.items():
    print(f"== {k}"); print(v.round(1).head(15).to_string())
print("== PO values (NOK)"); print(cmp_.round(0).to_string())
print("== pattern"); print(pattern.to_string())
