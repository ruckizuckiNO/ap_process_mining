"""Step 0 + Step 2: clean the event log, profile users to find bots, build case-level KPIs."""
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import RAW_CSV, OUT, FIG, EVENTS_PKL, CASES_CSV, INVOICES_CSV, BOT_USERS, POSTING_ACTIVITIES

df = pd.read_csv(RAW_CSV, low_memory=False)
df["ts"] = pd.to_datetime(df.Timestamp_DDMMYYYY, format="%d/%m/%Y %H:%M:%S")

n_cases_raw = df.PO_ID.nunique()
n_events_raw = len(df)
untyped = df.loc[df.SAP_PO_Type_Code.isna(), "PO_ID"].unique()
df = df[~df.PO_ID.isin(untyped)].copy()

# Item_No is not chronological for ~45k steps, so order by timestamp (Event_ID breaks ties)
df = df.sort_values(["PO_ID", "ts", "Event_ID"]).reset_index(drop=True)
df["gap_s"] = df.groupby("PO_ID").ts.diff().dt.total_seconds()
df["hour"] = df.ts.dt.hour

# --- User profiling (bot detection) ---------------------------------------
users = df.groupby("User_ID").agg(
    events=("Event_ID", "size"),
    cases=("PO_ID", "nunique"),
    activities=("Activity_ID", "nunique"),
    active_days=("ts", lambda t: t.dt.normalize().nunique()),
    median_gap_s=("gap_s", "median"),
    night_share=("hour", lambda h: ((h < 6) | (h >= 22)).mean()),
    batch_0306_share=("hour", lambda h: h.between(3, 5).mean()),
)
users["events_per_active_day"] = users.events / users.active_days
users["share_of_all_events"] = users.events / len(df)
users = users.sort_values("events", ascending=False)
users["is_bot"] = users.index.isin(BOT_USERS)
users.to_csv(OUT / "bot_profile.csv")

fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
for ax, u in zip(axes, ["User1587", "User280", "__humans__"]):
    sub = df[~df.User_ID.isin(BOT_USERS)] if u == "__humans__" else df[df.User_ID == u]
    sub.hour.value_counts().reindex(range(24), fill_value=0).plot.bar(ax=ax, color="#2a7f62" if u != "__humans__" else "#888")
    ax.set_title("All other users" if u == "__humans__" else u)
    ax.set_xlabel("Hour of day")
axes[0].set_ylabel("Events")
fig.suptitle("Event timing by hour: system users vs. humans")
fig.tight_layout()
fig.savefig(FIG / "bot_hours.png", dpi=130)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 5))
u = users[users.events >= 20]
ax.scatter(u.events_per_active_day, u.median_gap_s / 3600 + 1e-3, s=np.sqrt(u.cases) * 2,
           c=np.where(u.is_bot, "#d1495b", "#4a6fa5"), alpha=0.7)
for name in BOT_USERS + ["User1550"]:
    ax.annotate(name, (u.loc[name, "events_per_active_day"], u.loc[name, "median_gap_s"] / 3600 + 1e-3))
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Events per active day (log)")
ax.set_ylabel("Median gap to previous event in case, hours (log)")
ax.set_title("User workload vs. waiting time (bubble size ~ cases touched)")
fig.tight_layout()
fig.savefig(FIG / "bot_scatter.png", dpi=130)
plt.close(fig)

# --- Case-level KPIs -------------------------------------------------------
df["is_bot"] = df.User_ID.isin(BOT_USERS)
g = df.groupby("PO_ID")
cases = pd.DataFrame({
    "po_type": g.SAP_PO_Type_Code.first(),
    "po_type_name": g.PO_Type.first(),
    "supplier": g.Supplier_ID.first(),
    "start": g.ts.min(),
    "end": g.ts.max(),
    "trace_length": g.size(),
    "bot_events": g.is_bot.sum(),
    "last_activity": g.Activity_ID.last(),
    "invoice_value_nok": g.Case_Invoice_Value_NOK.first(),
})
cases["throughput_days"] = (cases.end - cases.start).dt.total_seconds() / 86400
# First posting = earliest moment the invoice became payable (the log has no payment dates)
first_post = df[df.Activity_ID.isin(POSTING_ACTIVITIES)].groupby("PO_ID").ts.min()
cases["first_post"] = first_post
cases["posted"] = cases.first_post.notna()
cases["n_postings"] = df[df.Activity_ID.isin(POSTING_ACTIVITIES)].groupby("PO_ID").size().reindex(cases.index, fill_value=0)
cases["time_to_post_days"] = (cases.first_post - cases.start).dt.total_seconds() / 86400
cases["manual_count"] = cases.trace_length - cases.bot_events
cases["automation_rate"] = cases.bot_events / cases.trace_length
cases["completed"] = cases.last_activity.isin(
    ["MM Posted Automatically", "Posted manually", "MM Autopost after manual process", "Invoice document deleted"])
for k in ["trace_length", "throughput_days", "manual_count", "time_to_post_days"]:
    cases[f"log_{k}"] = np.log(cases[k] + 1)
cases["start_week"] = cases.start.dt.to_period("W-SUN").dt.start_time
cases.index.name = "PO_ID"
cases.to_csv(CASES_CSV)

# Invoice cycles: a PO_ID can hold several SAP_PO_No cycles (Item_No restarts at 1 for each), most
# likely separate invoices against the same PO. Each cycle gets its own SLA clock: first event -> first posting.
df["is_post"] = df.Activity_ID.isin(POSTING_ACTIVITIES)
gi = df.groupby(["PO_ID", "SAP_PO_No"])
inv = pd.DataFrame({
    "po_type": gi.SAP_PO_Type_Code.first(),
    "start": gi.ts.min(),
    "first_post": df[df.is_post].groupby(["PO_ID", "SAP_PO_No"]).ts.min(),
    "events": gi.size(),
})
inv["posted"] = inv.first_post.notna()
inv["time_to_post_days"] = (inv.first_post - inv.start).dt.total_seconds() / 86400
inv["log_time_to_post_days"] = np.log(inv.time_to_post_days + 1)
inv["start_week"] = inv.start.dt.to_period("W-SUN").dt.start_time
inv.to_csv(INVOICES_CSV)
df = df.drop(columns="is_post")

df.drop(columns=["Timestamp_MMDDYYYY", "Case_Start_Date_MMDDYYYY", "Case_End_Date_MMDDYYYY"]).to_pickle(EVENTS_PKL)

summary = {
    "events_raw": int(n_events_raw),
    "cases_raw": int(n_cases_raw),
    "cases_untyped_dropped": int(len(untyped)),
    "cases": int(len(cases)),
    "events": int(len(df)),
    "users": int(df.User_ID.nunique()),
    "activities": int(df.Activity_ID.nunique()),
    "first_event": str(df.ts.min()),
    "last_event": str(df.ts.max()),
    "bot_event_share": float(df.is_bot.mean()),
    "incomplete_cases": int((~cases.completed).sum()),
    "invoice_cycles": int(len(inv)),
    "cases_multi_cycle": int((inv.groupby(level="PO_ID").size() > 1).sum()),
}
pd.Series(summary).to_json(OUT / "summary.json", indent=1)
print(pd.Series(summary))
print(users.head(8).round(3).to_string())
