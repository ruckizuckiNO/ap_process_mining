"""Daily numbers for User1341 from their first day to their last (12 Jan 2026), with an estimate of handling time.

The log only has completion timestamps, so handling time is estimated from the gap between two consecutive User1341
events on the same day: the gap before an event is the time spent on that invoice. Gaps longer than BREAK_MIN minutes
are treated as breaks / other work and split the day into work sessions; the first event of a session gets the
median in-session gap. Active time = sum of the in-session gaps + one median gap per session.

Queue = ZNB1 invoices of the suppliers whose main clerk is User1341 (notes/znb1_supplier_main_clerk.csv), from the
first CTRL1 routing to the first posting. "Past 10 days" = still open and more than 10 days after the clock start
(GR clock, so the 2% discount is already lost).
-> notes/user1341_daily.xlsx (+ .csv), notes/figures/user1341_daily.png
Run from the repo root:  PYTHONPATH=. conda run -n datasci python -W ignore notes/user1341_daily.py
"""
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from common import POSTING_ACTIVITIES, SLA_SOFT_DAYS, load_events, load_invoices
from common import use_invoice_proxy_labels

use_invoice_proxy_labels()  # chart text: invoice -> invoice proxy

NOTES = Path(__file__).resolve().parent
K = ["PO_ID", "SAP_PO_No"]
U = "User1341"
BREAK_MIN = 15
SENSITIVITY = [5, 10, 15, 30, 60]
BLUE, ORANGE, RED_INK = "#2a78d6", "#eb6834", "#b3261e"
INK, INK2, GRID, SURFACE, SHADE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb", "#efeee9"

ev = load_events()
u = ev[ev.User_ID == U].sort_values(["ts", "Event_ID"]).copy()
u["day"] = u.ts.dt.normalize()
u["gap_min"] = u.groupby("day").ts.diff().dt.total_seconds() / 60
first_day, last_day = u.day.min(), u.day.max()


def sessions(df, brk):
    """Return per-event session id and estimated minutes for a break threshold."""
    new = df.gap_min.isna() | (df.gap_min > brk)
    sid = new.cumsum()
    med = df.gap_min[df.gap_min <= brk].median()
    est = df.gap_min.where(~new, med)
    return sid, est, med


u["session"], u["est_min"], MED = sessions(u, BREAK_MIN)
u["is_post"] = u.Activity_ID == "Posted manually"
u["znb1"] = u.SAP_PO_Type_Code == "ZNB1"

sens = []
for b in SENSITIVITY:
    sid, est, med = sessions(u, b)
    days = u.day.nunique()
    sens.append({"break_threshold_min": b, "median_min_per_invoice": med,
                 "mean_min_per_invoice": est.mean(), "share_of_gaps_counted_as_work_pct": 100 * (u.gap_min <= b).mean(),
                 "active_hours_per_working_day": est.sum() / 60 / days, "sessions_per_day": sid.nunique() / days})
sens = pd.DataFrame(sens)

# --- queue of User1341's suppliers, reconstructed per day ---
z = ev[ev.SAP_PO_Type_Code == "ZNB1"]
mc = pd.read_csv(NOTES / "znb1_supplier_main_clerk.csv")
mine = set(mc.loc[mc.main_clerk == U, "Supplier_ID"])
qin = z[z.SAP_Next_Processing_Type == "CTRL1"].groupby(K).ts.min()
post = z[z.Activity_ID.isin(POSTING_ACTIVITIES)].sort_values("ts").groupby(K).first()
q = pd.DataFrame({"qin": qin, "sup": z.groupby(K).Supplier_ID.first().reindex(qin.index)})
q = q[q.sup.isin(mine)]
q["out"] = post.ts.reindex(q.index)
q["poster"] = post.User_ID.reindex(q.index)
q["clock"] = load_invoices().set_index(K).clock_start.reindex(q.index)

# --- daily table: every calendar day, so absences and weekends show ---
rows = []
for d in pd.date_range(first_day, last_day):
    e = d + pd.Timedelta(days=1)
    x = u[u.day == d]
    open_end = (q.qin < e) & ((q.out >= e) | q.out.isna())
    rows.append({
        "date": d.date(), "weekday": d.strftime("%a"),
        "active": len(x) > 0,
        "postings": int(x.is_post.sum()),
        "znb1_postings": int((x.is_post & x.znb1).sum()),
        "other_type_postings": int((x.is_post & ~x.znb1).sum()),
        "first_event": x.ts.min().strftime("%H:%M") if len(x) else "",
        "last_event": x.ts.max().strftime("%H:%M") if len(x) else "",
        "span_hours": (x.ts.max() - x.ts.min()).total_seconds() / 3600 if len(x) else 0,
        "sessions": x.session.nunique(),
        "est_active_hours": x.est_min.sum() / 60,
        "est_min_per_invoice": x.est_min.sum() / len(x) if len(x) else None,
        "median_gap_min": x.gap_min.median(),
        "queue_start": int(((q.qin < d) & ((q.out >= d) | q.out.isna())).sum()),
        "entered": int(((q.qin >= d) & (q.qin < e)).sum()),
        "posted_from_queue": int(((q.out >= d) & (q.out < e)).sum()),
        "queue_end": int(open_end.sum()),
        "queue_end_past_10d": int((open_end & ((e - q.clock).dt.total_seconds() / 86400 > SLA_SOFT_DAYS)).sum()),
        "oldest_in_queue_days": ((e - q.qin[open_end]).dt.total_seconds() / 86400).max() if open_end.any() else 0,
    })
daily = pd.DataFrame(rows)
daily.to_csv(NOTES / "user1341_daily.csv", index=False)

act = daily[daily.active]
summary = {
    "Period": f"{first_day:%d %b %Y} – {last_day:%d %b %Y}",
    "Active days": len(act),
    "Postings": int(act.postings.sum()),
    "Postings per active day (median / max)": f"{act.postings.median():.0f} / {act.postings.max()}",
    "Est. minutes per invoice (median gap in a session)": round(MED, 2),
    "Est. minutes per invoice (incl. longer pauses ≤ break)": round(u.est_min.mean(), 2),
    "Est. active hours per active day (median / max)": f"{act.est_active_hours.median():.1f} / {act.est_active_hours.max():.1f}",
    "Span first→last event per active day (median hours)": round(act.span_hours.median(), 1),
    "Days with queue_end_past_10d > 0": int((daily.queue_end_past_10d > 0).sum()),
    "Break threshold (min)": BREAK_MIN,
}

# --- Excel ---
with pd.ExcelWriter(NOTES / "user1341_daily.xlsx", engine="openpyxl") as xw:
    pd.DataFrame(list(summary.items()), columns=["Measure", "Value"]).to_excel(xw, sheet_name="Summary", index=False)
    out = daily.rename(columns={
        "date": "Date", "weekday": "Day", "active": "Active", "postings": "Postings", "znb1_postings": "ZNB1 postings",
        "other_type_postings": "Other PO types", "first_event": "First event", "last_event": "Last event",
        "span_hours": "Span (h)", "sessions": "Work sessions", "est_active_hours": "Est. active time (h)",
        "est_min_per_invoice": "Est. min per invoice", "median_gap_min": "Median gap (min)",
        "queue_start": "Queue at start", "entered": "Entered queue", "posted_from_queue": "Posted from queue",
        "queue_end": "Queue at end", "queue_end_past_10d": "Of which past 10 days",
        "oldest_in_queue_days": "Oldest in queue (days)"})
    out.to_excel(xw, sheet_name="Daily", index=False)
    sens.rename(columns={
        "break_threshold_min": "Gap counted as break above (min)", "median_min_per_invoice": "Median min per invoice",
        "mean_min_per_invoice": "Mean min per invoice", "share_of_gaps_counted_as_work_pct": "Gaps counted as work (%)",
        "active_hours_per_working_day": "Active hours per active day", "sessions_per_day": "Sessions per day",
    }).to_excel(xw, sheet_name="Handling time", index=False)
    pd.DataFrame([
        ("Source", "INN354 AP event log; notes/user1341_daily.py"),
        ("Handling time", "Timestamps mark the completion of a step. The gap between two consecutive User1341 events "
                          "on the same day is taken as the time spent on the second one."),
        ("Break", f"A gap over {BREAK_MIN} min is a break or other work (meetings, mail, other systems) and starts a "
                  "new work session. The first event of a session is given the median in-session gap."),
        ("Active time", "Sum of the estimated minutes per event. A lower bound on working time: it misses work that "
                        "leaves no trace in the log (lookups, mails, phone calls)."),
        ("Queue", "ZNB1 invoices of the 13 suppliers whose main clerk is User1341, from first CTRL1 routing to first "
                  "posting, whoever posts them. Counted at midnight."),
        ("Past 10 days", "Open at the end of the day and more than 10 days after the clock start (goods receipt for "
                         "ZNB1): the 2% discount is already lost."),
        ("Time zone", "Timestamps as in the log. Most activity is 13:00–19:00, which may be a time-zone offset."),
    ], columns=["Term", "Definition"]).to_excel(xw, sheet_name="Notes", index=False)
    head_fill = PatternFill("solid", fgColor="2A78D6")
    for ws in xw.book.worksheets:
        for c in ws[1]:
            c.font = Font(bold=True, color="FFFFFF")
            c.fill = head_fill
            c.alignment = Alignment(wrap_text=True, vertical="center")
        for i, col in enumerate(ws.columns, 1):
            width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
            ws.column_dimensions[get_column_letter(i)].width = min(max(10, width + 2), 70)
        ws.freeze_panes = "A2"
    ws = xw.book["Daily"]
    for row in ws.iter_rows(min_row=2):
        for c in row:
            if isinstance(c.value, float):
                c.number_format = "0.0"
        if row[2].value is False:
            for c in row:
                c.font = Font(color="9A9993")

# --- chart: three panels ---
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"]})
fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(13, 9.5), dpi=200, sharex=True)
fig.patch.set_facecolor(SURFACE)
dd = pd.to_datetime(daily.date)
a1.bar(dd, daily.znb1_postings, width=0.8, color=BLUE, zorder=2, label="ZNB1")
a1.bar(dd, daily.other_type_postings, bottom=daily.znb1_postings, width=0.8, color="#8fb6e8", zorder=2,
       label="Other PO types (mostly ZJD1)")
a1.set_title("Invoices posted per day", loc="left", fontsize=10.5, fontweight="bold", color=INK)
a1.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=INK2)
a2.bar(dd, daily.est_active_hours, width=0.8, color=BLUE, zorder=2)
a2.set_title(f"Estimated active posting time per day (hours; gaps over {BREAK_MIN} min count as breaks)", loc="left",
             fontsize=10.5, fontweight="bold", color=INK)
a3.fill_between(dd, daily.queue_end, step="mid", color="#f6c4ad", zorder=1, label="Open in queue at end of day")
a3.step(dd, daily.queue_end_past_10d, where="mid", color=RED_INK, linewidth=1.6, zorder=2,
        label="Of which already past 10 days (discount lost)")
a3.set_title("Queue of User1341's 13 suppliers", loc="left", fontsize=10.5, fontweight="bold", color=INK)
a3.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=INK2)
off = daily[(~daily.active) & daily.weekday.isin(["Mon", "Tue", "Wed", "Thu", "Fri"])]
for ax in (a1, a2, a3):
    ax.set_facecolor(SURFACE)
    for d in pd.to_datetime(off.date):
        ax.axvspan(d - pd.Timedelta(hours=12), d + pd.Timedelta(hours=12), color=SHADE, zorder=0, linewidth=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(axis="both", length=0, colors=INK2, labelsize=9)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color(GRID)
a3.xaxis.set_major_locator(mdates.MonthLocator())
a3.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
a3.set_xlim(dd.min() - pd.Timedelta(days=1), dd.max() + pd.Timedelta(days=1))
fig.suptitle(f"User1341 day by day: about {MED:.1f} min per invoice, "
             f"{act.est_active_hours.median():.1f} h of posting on a typical day", x=0.05, ha="left", y=0.975,
             fontsize=15, fontweight="bold", color=INK)
fig.text(0.05, 0.94, f"{first_day:%d %b %Y} – {last_day:%d %b %Y} · {len(act)} active days · "
         f"{int(act.postings.sum()):,} postings · grey bands = working days without any User1341 activity",
         fontsize=9.5, color=INK2, ha="left", va="top")
fig.text(0.05, 0.008, "Handling time estimated from gaps between consecutive User1341 events on the same day (timestamps "
         "mark completion). Queue counted at midnight; ZNB1 clock from goods receipt.", fontsize=7.5, color=INK2)
fig.subplots_adjust(left=0.05, right=0.985, top=0.89, bottom=0.05, hspace=0.3)
fig.savefig(NOTES / "figures" / "user1341_daily.png", facecolor=SURFACE)

for k, v in summary.items():
    print(f"{k}: {v}")
print(sens.round(2).to_string(index=False))
print(act.postings.describe().round(1).to_dict())
print("corr postings vs active hours", act.postings.corr(act.est_active_hours).round(2))
print(daily.sort_values("queue_end_past_10d", ascending=False).head(5)[
    ["date", "postings", "queue_end", "queue_end_past_10d", "oldest_in_queue_days"]].to_string(index=False))
