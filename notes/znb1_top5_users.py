"""ZNB1: the top five suppliers that miss the 10-day limit on the manual queue route, and who handles them.

Top five = highest-ranked suppliers in znb1_supplier_priority.csv whose invoices all go to the manual control
queue (CTRL1) - the "one clerk" pattern. Run znb1_supplier_priority.py first.
Run from the repo root:  conda run -n datasci python -W ignore notes/znb1_top5_users.py
"""
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = Path(os.environ.get("AP_EVENT_LOG", ROOT.parent / "log.csv"))
NOTES = ROOT / "notes"
FIG = NOTES / "figures"
BOTS = ["User1587", "User280"]
TOP5 = ["Supplier381", "Supplier361", "Supplier380", "Supplier362", "Supplier363"]
K = ["PO_ID", "SAP_PO_No"]

df = pd.read_csv(RAW, low_memory=False)
z = df[df.SAP_PO_Type_Code == "ZNB1"].copy()
z["ts"] = pd.to_datetime(z.Timestamp_DDMMYYYY, format="%d/%m/%Y %H:%M:%S")
z = z.sort_values(K + ["ts", "Event_ID"])


def queue_table(ev):
    """One row per invoice that went to CTRL1 and was posted manually: start, queue entry, posting, poster."""
    posted = ev[ev.Activity_ID == "Posted manually"].drop_duplicates(K).set_index(K)
    t = pd.concat({
        "start": ev.groupby(K).ts.min(),
        "queue_in": ev[ev.SAP_Next_Processing_Type == "CTRL1"].groupby(K).ts.min(),
        "posted": posted.ts,
        "user": posted.User_ID,
        "supplier": ev.groupby(K).Supplier_ID.first(),
    }, axis=1).dropna()
    t["days_to_post"] = (t.posted - t.start).dt.total_seconds() / 86400
    t["queue_days"] = (t.posted - t.queue_in).dt.total_seconds() / 86400
    t["late"] = t.days_to_post > 10
    return t


top = queue_table(z[z.Supplier_ID.isin(TOP5)])
rest = queue_table(z[~z.Supplier_ID.isin(TOP5)])
prio = pd.read_csv(NOTES / "znb1_supplier_priority_invoices.csv").set_index(K)
top["missed_discount_nok"] = prio.missed_discount_nok.reindex(top.index).fillna(0)
top["user_group"] = top.user.where(top.user.isin(["User1341", "User1482", "User1082"]), "Other users")
human = z[z.Supplier_ID.isin(TOP5) & ~z.User_ID.isin(BOTS)]

# --- Tables ------------------------------------------------------------------
by_sup_user = top.groupby(["supplier", "user"]).agg(
    invoices=("late", "size"), late_10d=("late", "sum"), median_days_to_post=("days_to_post", "median"),
    median_queue_days=("queue_days", "median"), missed_discount_nok=("missed_discount_nok", "sum"))
by_sup_user["late_pct"] = 100 * by_sup_user.late_10d / by_sup_user.invoices
by_sup_user["share_of_supplier_pct"] = 100 * by_sup_user.invoices / by_sup_user.groupby(level=0).invoices.transform("sum")
by_sup_user = by_sup_user.round(1).sort_values(["supplier", "invoices"], ascending=[True, False])
by_sup_user.to_csv(NOTES / "znb1_top5_supplier_users.csv")

by_sup = top.groupby("supplier").agg(
    invoices=("late", "size"), late_10d=("late", "sum"), median_days_to_post=("days_to_post", "median"),
    median_queue_days=("queue_days", "median"), missed_discount_nok=("missed_discount_nok", "sum"),
    user1341_pct=("user", lambda u: 100 * (u == "User1341").mean())).reindex(TOP5)
by_sup["within_10d_pct"] = 100 * (1 - by_sup.late_10d / by_sup.invoices)
by_sup["pos"] = z[z.Supplier_ID.isin(TOP5)].groupby("Supplier_ID").PO_ID.nunique()

# Daily workload: plain average of distinct invoices (PO_ID x SAP_PO_No) a user touched per day they were active
USERS = ["User1341", "User1482", "User1082", "User489", "User1331"]
work = df[df.User_ID.isin(USERS)].copy()
work["day"] = pd.to_datetime(work.Timestamp_DDMMYYYY, format="%d/%m/%Y %H:%M:%S").dt.date
work = work.drop_duplicates(["User_ID", "day"] + K)


def per_day(x, name):
    g = x.groupby(["User_ID", "day"]).size().groupby("User_ID")
    return pd.DataFrame({f"{name}_per_day": g.mean(), f"{name}_active_days": g.size(), f"{name}_max_day": g.max()})


daily = pd.concat([per_day(work, "all_types"), per_day(work[work.SAP_PO_Type_Code == "ZNB1"], "znb1"),
                   per_day(work[work.Supplier_ID.isin(TOP5)], "top5")], axis=1).reindex(USERS)
daily.index.name = "user"
daily.round(1).to_csv(NOTES / "znb1_top5_user_daily_load.csv")

top["month"] = top.start.dt.to_period("M")
monthly = top.groupby("month").agg(invoices=("late", "size"), late_pct=("late", "mean"),
                                   median_queue_days=("queue_days", "median"),
                                   user1341_pct=("user", lambda u: (u == "User1341").mean()))
monthly[["late_pct", "user1341_pct"]] *= 100
top["queue_weekday"] = top.queue_in.dt.day_name()
weekday = top.groupby("queue_weekday").agg(invoices=("late", "size"), late_pct=("late", "mean"))
weekday["late_pct"] *= 100

# --- Figure ------------------------------------------------------------------
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dc"
COL = {"User1341": "#2a78d6", "User1482": "#eb6834", "User1082": "#1baf7a", "Other users": "#eda100"}
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 4, figsize=(22, 5.2), gridspec_kw={"width_ratios": [1.25, 1, 1, 0.8]})

# 1: late invoices per supplier, split by who posted them
ax = axes[0]
late = top[top.late].groupby(["supplier", "user_group"]).size().unstack(fill_value=0).reindex(TOP5[::-1])
left = np.zeros(len(late))
for g in COL:
    if g in late:
        ax.barh(late.index, late[g], left=left, color=COL[g], height=0.6, edgecolor="white", linewidth=2, label=g)
        left += late[g].values
for y, s in enumerate(late.index):
    r = by_sup.loc[s]
    ax.text(left[y] + 4, y, f"{r.late_10d:.0f} of {r.invoices:.0f} late  ·  {r.missed_discount_nok / 1e6:.1f}M NOK",
            va="center", color=INK, fontsize=9)
ax.set_xlim(0, left.max() * 1.75)
ax.set_xlabel("Invoices posted after more than 10 days")
ax.set_title("Late invoices by supplier and who posted them", loc="left", color=INK, fontweight="bold")
ax.legend(frameon=False, loc="lower right", fontsize=9)
ax.grid(axis="x", color=GRID); ax.set_axisbelow(True)

# 2: queue wait, top five vs other ZNB1 suppliers
ax = axes[1]
bins = np.arange(0, 29, 1)
for data, lab, c in [(rest.queue_days, f"Other ZNB1 suppliers (median {rest.queue_days.median():.1f} d)", "#8a8980"),
                     (top.queue_days, f"Top five (median {top.queue_days.median():.1f} d)", COL["User1341"])]:
    ax.hist(data.clip(upper=28), bins=bins, density=True, histtype="step", linewidth=2, color=c, label=lab)
ax.axvline(10, color=INK, linewidth=1, linestyle="--")
ax.text(10.3, ax.get_ylim()[1] * 0.55, "10-day limit", color=INK, fontsize=9)
ax.set_xlabel("Days in manual queue (CTRL1) until posted")
ax.set_ylabel("Share of invoices")
ax.set_title("Top-five invoices wait about a week in the queue", loc="left", color=INK, fontweight="bold")
ax.legend(frameon=False, fontsize=9, loc="upper right", bbox_to_anchor=(1, 0.95))
ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)

# 3: share late per month
ax = axes[2]
m = monthly.copy(); m.index = m.index.strftime("%b %y")
ax.plot(m.index, m.late_pct, color=COL["User1341"], linewidth=2, marker="o", markersize=8,
        markeredgecolor="white", markeredgewidth=2)
for x, r in m.iterrows():
    ax.annotate(f"{r.late_pct:.0f}%", (x, r.late_pct), textcoords="offset points", xytext=(0, 9), ha="center",
                fontsize=9, color=INK)
ax.annotate("summer holidays", ("Aug 25", m.loc["Aug 25", "late_pct"]), xytext=(18, -32),
            textcoords="offset points", fontsize=9, color=MUTED)
ax.annotate("User1341 absent,\nUser1482 / User1082 cover", ("Jan 26", m.loc["Jan 26", "late_pct"]),
            xytext=(-150, -30), textcoords="offset points", fontsize=9, color=MUTED)
ax.set_ylim(0, 112)
ax.set_ylabel("Invoices later than 10 days (%)")
ax.set_xlabel("Month the invoice arrived")
ax.set_title("Lateness spikes when the usual clerk is away", loc="left", color=INK, fontweight="bold")
ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)

# 4: daily workload per user (plain average per active day)
ax = axes[3]
y = np.arange(len(USERS))[::-1]
for off, col, lab, c in [(0.18, "all_types_per_day", "All PO types", "#8a8980"),
                         (-0.18, "top5_per_day", "Top-five suppliers", COL["User1341"])]:
    v = daily[col].fillna(0).values
    ax.barh(y + off, v, height=0.34, color=c, edgecolor="white", linewidth=2, label=lab)
    for yi, vi in zip(y + off, v):
        ax.text(vi + 0.6, yi, f"{vi:.1f}", va="center", fontsize=9, color=INK)
ax.set_yticks(y, USERS)
ax.set_xlim(0, daily.all_types_per_day.max() * 1.25)
ax.set_xlabel("Invoices handled per active day (average)")
ax.set_title("Daily workload per user", loc="left", color=INK, fontweight="bold")
ax.legend(frameon=False, fontsize=9, loc="lower right")
ax.grid(axis="x", color=GRID); ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(FIG / "znb1_top5_supplier_users.png", dpi=140)

pd.set_option("display.width", 220)
print(by_sup.round(1).to_string())
print(by_sup_user.to_string())
print(daily.round(1).to_string())
print(monthly.round(1).to_string())
print(weekday.round(1).to_string())
print("human activities on top five:", human.Activity_ID.value_counts().to_dict())
print("total missed discount top five:", round(top.missed_discount_nok.sum()))
