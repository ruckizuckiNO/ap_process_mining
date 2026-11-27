"""Builds output/report_znb1.html: a short PO-value overview per PO type, then the whole analysis for ZNB1 only.

Reads the same outputs as 05_report.py (steps 01-04) and reuses its stylesheet. output/report.html is not touched.
"""
import html
import json
import re

import numpy as np
import pandas as pd

from common import (OUT, ROOT, PO_TYPES, BOT_USERS, POSTING_ACTIVITIES, SLA_SOFT_DAYS, GR_ACTIVITY, GR_OBJECT_CODE,
                    load_events, load_cases, load_invoices)

T = "ZNB1"
CSS = re.search(r"<style>(.*?)</style>", (ROOT / "05_report.py").read_text(), re.S).group(1)
CSS = CSS.replace("{{", "{").replace("}}", "}")

S = json.loads((OUT / "summary.json").read_text())
types = pd.read_csv(OUT / "po_type_overview.csv", index_col="po_type")
variants = pd.read_csv(OUT / "top_variants.csv")
bots = pd.read_csv(OUT / "bot_profile.csv", index_col="User_ID")
ctrl = pd.read_csv(OUT / "control_chart_results.csv")
cap = pd.read_csv(OUT / "capability.csv", index_col="po_type")
cap_post = pd.read_csv(OUT / "capability_first_post.csv", index_col="po_type")
cap_inv = pd.read_csv(OUT / "capability_invoice.csv", index_col="po_type")
pr = pd.read_csv(OUT / "priority.csv", index_col="po_type")
routing = pd.read_csv(OUT / "routing.csv", index_col="po_type")
direct = pd.read_csv(OUT / "direct_posting.csv", index_col="po_type")
lt = pd.read_csv(OUT / "lead_time_variants.csv", index_col="variant_rank")
presence = pd.read_csv(OUT / "activity_presence.csv", index_col="Activity_ID")[T]


def pct(x, d=1):
    return "–" if pd.isna(x) else f"{x * 100:.{d}f}%"


def num(x, d=0):
    return "–" if pd.isna(x) else f"{x:,.{d}f}"


def mnok(x):
    return f"{x / 1e6:,.1f}M"


def bnok(x):
    return f"{x / 1e9:,.2f}B"


def cpu_cell(v):
    cls = "bad" if v < 0.5 else "warn" if v < 1 else "ok"
    return f'<td class="n {cls}">{v:.2f}</td>'


def code(c):
    return f'<span class="code">{c}</span>'


def note(body, label="Annotation"):
    return f'<div class="note"><div class="lbl">{label}</div>{body}</div>'


def table(head, rows, cls=""):
    th = "".join(f"<th>{h}</th>" for h in head)
    body = "".join(f"<tr>{r}</tr>" for r in rows)
    return f'<div class="tbl {cls}"><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def fig(src, cap_text, cls=""):
    return (f'<figure class="{cls}"><div class="plate"><img src="{src}" alt="{html.escape(cap_text)}" loading="lazy"></div>'
            f"<figcaption>{cap_text}</figcaption></figure>")


def variant_html(v):
    return '<ol class="trace">' + "".join(f"<li>{html.escape(s)}</li>" for s in v.split(" → ")) + "</ol>"


def fmt_gap(s):
    if s < 120:
        return f"{s:.0f} s"
    if s < 2 * 3600:
        return f"{s / 60:.0f} min"
    if s < 2 * 86400:
        return f"{s / 3600:.1f} h"
    return f"{s / 86400:.1f} days"


def example_case(pid):
    """Event table for one real case, marking system users and the gap since the previous event."""
    e = EV[EV.PO_ID == pid]
    rows, prev = [], None
    for i, (_, r) in enumerate(e.iterrows(), 1):
        gap = "" if prev is None else fmt_gap((r.ts - prev).total_seconds())
        prev = r.ts
        who = '<span class="pill sys">system</span>' if r.User_ID in BOT_USERS else '<span class="pill">human</span>'
        rows.append(f"<td class='n'>{i}</td><td class='n'>{r.ts:%d %b %H:%M:%S}</td><td class='n'>{gap}</td>"
                    f"<td>{html.escape(r.Activity_ID)}</td><td class='n'>{r.SAP_PO_No}</td><td>{r.User_ID} {who}</td>")
    return table(["#", "Timestamp", "Wait since previous", "Activity", "Invoice (SAP_PO_No)", "User"], rows)


# ---------------------------------------------------------------- overview: PO value per PO type
CASES_ALL = load_cases()
val = CASES_ALL.groupby("po_type").invoice_value_nok.agg(pos="size", total="sum", mean="mean", median="median")
val = val.sort_values("total", ascending=False)
val_total = val.total.sum()
val_max = val.total.max()
val_rows = []
for c, r in val.iterrows():
    bar = f'<div class="bar"><span style="width:{r.total / val_max * 100:.1f}%"></span></div>'
    hl = " class='hl'" if c == T else ""
    val_rows.append(f"<td{hl}>{code(c)}</td><td{hl}>{PO_TYPES[c]}</td><td class='n'>{num(r.pos)}</td>"
                    f"<td class='n'>{mnok(r.total)}</td><td class='n'>{pct(r.total / val_total)}</td><td>{bar}</td>"
                    f"<td class='n'>{num(r['mean'])}</td><td class='n'>{num(r['median'])}</td>")
val_rows.append(f"<td></td><td><b>All PO types</b></td><td class='n'><b>{num(val.pos.sum())}</b></td><td class='n'><b>{mnok(val_total)}</b></td>"
                f"<td class='n'>100%</td><td></td><td class='n'>{num(val_total / val.pos.sum())}</td><td></td>")
v1, vj = val.loc[T], val.loc["ZJD1"]

# ---------------------------------------------------------------- ZNB1 data
EV_ALL = load_events()
EV = EV_ALL[EV_ALL.SAP_PO_Type_Code == T].copy()
CASES = CASES_ALL[CASES_ALL.po_type == T].set_index("PO_ID")
INV = load_invoices()
INV = INV[INV.po_type == T].set_index(["PO_ID", "SAP_PO_No"])
t, p, r_ = types.loc[T], pr.loc[T], routing.loc[T]
n_inv = len(INV)

# How each invoice was first posted, and its routing code at the first validation
INV["route"] = EV[EV.Activity_ID.isin(POSTING_ACTIVITIES)].groupby(["PO_ID", "SAP_PO_No"]).Activity_ID.first()
INV["rcode"] = EV[EV.Activity_ID == "Initial validation"].groupby(["PO_ID", "SAP_PO_No"]).SAP_Next_Processing_Type.first()
# Missed 2% discount: PO value split evenly over its posted invoices (same rule as 04_prioritise.py)
INV["value"] = CASES.invoice_value_nok.reindex(INV.index.get_level_values(0)).values
INV["n_posted"] = INV.groupby(level=0).posted.transform("sum")
INV["share_value"] = np.where(INV.posted, INV.value / INV.n_posted.where(INV.n_posted > 0), 0)
INV["late"] = INV.posted & (INV.time_to_post_days > SLA_SOFT_DAYS)
INV["missed"] = np.where(INV.late, 0.02 * INV.share_value, 0)
# Same with the clock from the invoice's first event (arrival in AP) instead of goods receipt: upper bound
INV["arr_days"] = (INV.first_post - INV.start).dt.total_seconds() / 86400
INV["late_arr"] = INV.posted & (INV.arr_days > SLA_SOFT_DAYS)
INV["missed_arr"] = np.where(INV.late_arr, 0.02 * INV.share_value, 0)
miss, miss_arr = INV.missed.sum(), INV.missed_arr.sum()
late_n, late_arr_n = int(INV.late.sum()), int(INV.late_arr.sum())
gr_clock = INV.goods_receipt.notna()

posted = INV[INV.posted]
ROUTE = posted.groupby("route").agg(n=("time_to_post_days", "size"), median=("time_to_post_days", "median"),
                                    late=("late", "sum"), missed=("missed", "sum"))
ROUTE["share"] = ROUTE.n / ROUTE.n.sum()
ROUTE = ROUTE.sort_values("n", ascending=False)
m1 = ROUTE.loc["Posted manually"]
route_rows = [f"<td>{html.escape(k)}</td><td class='n'>{num(r.n)}</td><td class='n'>{pct(r.share, 0)}</td><td class='n'>{num(r['median'], 2)}</td>"
              f"<td class='n'>{num(r.late)}</td><td class='n'>{pct(r.late / r.n, 0)}</td><td class='n'>{mnok(r.missed)}</td>"
              for k, r in ROUTE.iterrows()]

RC = posted.groupby("rcode").agg(n=("late", "size"), late=("late", "sum"), missed=("missed", "sum"),
                                 manual=("route", lambda x: (x == "Posted manually").mean()))
RC = RC.sort_values("n", ascending=False)
rc_label = {"CTRL1": "manual control queue", "WAITGR": "goods-receipt check", "ZWMGRW": "goods receipt via WM", "DELE": "deletion"}
rc_rows = [f"<td>{code(k)} <small>{rc_label.get(k, '')}</small></td><td class='n'>{num(r.n)}</td><td class='n'>{pct(r.manual, 0)}</td>"
           f"<td class='n'>{num(r.late)}</td><td class='n'>{pct(r.late / r.n, 0)}</td><td class='n'>{mnok(r.missed)}</td><td class='n'>{pct(r.missed / miss, 0)}</td>"
           for k, r in RC.iterrows()]

# Who works on ZNB1
EV["grp"] = np.where(EV.User_ID.isin(BOT_USERS), EV.User_ID, "Humans")
by_grp = EV.groupby("grp").agg(events=("Event_ID", "size"), cases=("PO_ID", "nunique"), users=("User_ID", "nunique"))
hum_ev = EV[~EV.User_ID.isin(BOT_USERS)]
top_h = hum_ev.groupby("User_ID").agg(events=("Event_ID", "size"), cases=("PO_ID", "nunique"),
                                      posts=("Activity_ID", lambda x: (x == "Posted manually").sum()),
                                      top=("Activity_ID", lambda x: x.value_counts().index[0]))
top_h = top_h.sort_values("events", ascending=False)
n_hum = len(top_h)
top5_share = top_h.events.head(5).sum() / top_h.events.sum()
grp_rows = []
for g in [*BOT_USERS, "Humans"]:
    if g not in by_grp.index:
        continue
    r = by_grp.loc[g]
    tag = '<span class="pill sys">system</span>' if g in BOT_USERS else f'<span class="pill">{num(r.users)} people</span>'
    grp_rows.append(f"<td>{g} {tag}</td><td class='n'>{num(r.events)}</td><td class='n'>{pct(r.events / len(EV))}</td><td class='n'>{num(r.cases)}</td>")
hum_rows = [f"<td>{u}</td><td class='n'>{num(r.events)}</td><td class='n'>{pct(r.events / len(hum_ev))}</td><td class='n'>{num(r.cases)}</td>"
            f"<td class='n'>{num(r.posts)}</td><td>{html.escape(r.top)}</td>" for u, r in top_h.head(5).iterrows()]

# Activities: share of ZNB1 cases that contain each one
pres = presence[presence > 0].sort_values(ascending=False)
occ = EV.groupby("Activity_ID").size()
pres_rows = [f"<td>{html.escape(a)}</td><td class='n'>{pct(s, 0)}</td><td class='n'>{num(occ.get(a, 0))}</td>" for a, s in pres.items()]

# Top 5 variants and lead time by variant
var = variants[variants.po_type == T]
lt_var = var.set_index("rank").variant
vrows = [f"<td class='n'>{int(r['rank'])}</td><td>{variant_html(r.variant)}</td><td class='n'>{num(r.cases)}</td>"
         f"<td class='n'>{pct(r.share)}</td><td class='n'>{num(r.median_tp_days, 1)}</td><td class='n'>{num(r.mean_manual, 2)}</td>"
         for _, r in var.head(5).iterrows()]
lt_rows = []
for k, label in [*((str(i), None) for i in range(1, 6)), ("0", "Other variants"), ("ALL", f"All {T}")]:
    r = lt.loc[k]
    first = (f"<td class='n'>{k}</td><td>{variant_html(lt_var[int(k)])}</td>" if label is None
             else f"<td></td><td><b>{label}</b></td>")
    lt_rows.append(f"{first}<td class='n'>{num(r.cases)}</td><td class='n'>{num(r.invoices)}</td>"
                   f"<td class='n'>{num(r.median_days, 1)}</td><td class='n'>{num(r.mean_days, 1)}</td>"
                   f"<td class='n'>{num(r.late_invoices)}</td><td class='n'>{num(r.missed_discount_nok)}</td>")
lt_all, lt_other = lt.loc["ALL"], lt.loc["0"]

# Worked examples: a single-invoice happy-path case (variant 1) and a case holding three invoices
EX_REWORK = "PurDoc174751"
v1_steps = lt_var[1]
_cand = CASES[(CASES.manual_count == 0) & (CASES.trace_length == len(v1_steps.split(" → ")))]
_cand = _cand[_cand.index.isin(INV.groupby(level=0).size().loc[lambda s: s == 1].index)]
EX_HAPPY = _cand.sort_values("start").index[len(_cand) // 2]
xh, xr = CASES.loc[EX_HAPPY], CASES.loc[EX_REWORK]
EX_INV = INV.loc[EX_REWORK].time_to_post_days
ex_late = int((EX_INV > SLA_SOFT_DAYS).sum())
ex_poster = EV.loc[(EV.PO_ID == EX_REWORK) & (EV.Activity_ID == "Posted manually"), "User_ID"].iloc[0]

# Control charts
sub = ctrl[ctrl.po_type == T].set_index("kpi")
kpi_name = {"events": "Events per invoice", "time_to_post_days": "Days to first posting", "manual_count": "Manual events per invoice"}
clockv = pd.read_csv(OUT / "control_chart_clock_views.csv").set_index("view")
cv = {k: clockv.iloc[i] for i, k in enumerate(["head", "gr", "arr", "upper"])}
z_ttp = sub.loc["time_to_post_days"]
cc_first, cc_last, cc_fu = (pd.Timestamp(z_ttp[k]) for k in ("window_first", "window_last", "followup_start"))
wk_pass = [kpi_name[k].lower() for k in kpi_name if sub.loc[k].weekly_I_in_control in (True, "True")]
wk_fail = [kpi_name[k].lower() for k in kpi_name if sub.loc[k].weekly_I_in_control not in (True, "True")]
wk_text = ((" and ".join(wk_pass).capitalize() + (" passes" if len(wk_pass) == 1 else " pass") + " that test; ") if wk_pass else "No KPI passes that test: ") + \
    (" and ".join(wk_fail) + (" doesn't." if len(wk_fail) == 1 else " don't.") if wk_pass else ", ".join(wk_fail) + " all fail.")
ctrl_rows = []
for k in ["events", "time_to_post_days", "manual_count"]:
    r = sub.loc[k]
    rules = ", ".join(rr for rr in ["R1", "R2", "R3", "R4", "R5"] if r[rr] > 0) or "none"
    wk = "passes" if r.weekly_I_in_control in (True, "True") else "fails"
    ctrl_rows.append(f"<td>{kpi_name[k]}</td><td class='n'>{int(r.points_flagged)}/{int(r.subgroups)}</td><td>{rules}</td>"
                     f"<td>{'yes' if r.s_chart_R1 > 0 else 'no'}</td><td class='{'ok' if wk == 'passes' else 'bad'}'>{wk}</td>")

c1, ci, cp = cap.loc[T], cap_inv.loc[T], cap_post.loc[T]
usl10 = np.log(11)
dz = direct.loc[T]

# ---------------------------------------------------------------- page
EXTRA_CSS = """
.bar { background: var(--accent-soft); border-radius: 3px; height: 10px; min-width: 120px; }
.bar span { display: block; height: 100%; background: var(--accent); border-radius: 3px; }
td.hl { font-weight: 600; }
"""

page = f"""<meta charset="utf-8">
<title>ZNB1 Invoice Flow</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,500;6..72,600&family=Instrument+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap">
<style>{CSS}{EXTRA_CSS}</style>

<div class="wrap">
<header>
  <div class="eyebrow">Accounts payable · Process mining case · ZNB1 focus</div>
  <h1>ZNB1 Invoice Flow</h1>
  <p class="lede">A short look at where the purchase-order value sits across the eleven SAP PO types, then the full analysis (workflow, KPIs, control and capability, focus) for manual orders, {code(T)}, only.</p>
  <div class="facts">
    <span><b>{num(t.cases)}</b> ZNB1 POs</span>
    <span><b>{num(n_inv)}</b> invoices</span>
    <span><b>{num(t.events)}</b> events</span>
    <span><b>{num(p.suppliers)}</b> suppliers</span>
    <span><b>{bnok(v1.total)}</b> NOK PO value</span>
    <span>cases started <b>Jul–Nov 2025</b></span>
  </div>
</header>

<ul class="toc">
  <li><a href="#value">PO value per SAP category</a></li>
  <li><a href="#findings">ZNB1 key findings</a></li>
  <li><a href="#data">Data</a></li>
  <li><a href="#step1">1 · Workflow</a></li>
  <li><a href="#step2">2 · KPIs</a></li>
  <li><a href="#step3">3 · Control &amp; capability</a></li>
  <li><a href="#step4">4 · Focus</a></li>
</ul>

<h2 id="value"><span class="step">Overview</span>Total PO value per SAP category</h2>
<div class="prose"><p>The value is counted once per purchase order (<code>PO_ID</code>), using <code>Case_Invoice_Value_NOK</code>. That field holds the same amount on every row of a PO, so it's the PO total, not the value of a single invoice.</p></div>
{table(["Code", "PO type", "POs", "Total PO value (NOK)", "Share of value", "", "Mean per PO (NOK)", "Median per PO (NOK)"], val_rows)}
<div class="callout">
<p><b>Two PO types hold almost all the money.</b> {code('ZJD1')} ({mnok(vj.total)} NOK) and {code(T)} ({mnok(v1.total)} NOK) together make up {pct((vj.total + v1.total) / val_total, 0)} of the {bnok(val_total)} NOK total, from only {pct((vj.pos + v1.pos) / val.pos.sum(), 1)} of the POs. A ZNB1 PO is worth {num(v1['mean'] / 1e6, 2)}M NOK on average (median {num(v1['median'] / 1e3, 0)}k). The third-party types (ZNB2–4) have most of the volume but small values.</p>
<p>The rest of this report covers {code(T)} (Manual order) only.</p>
</div>

<h2 id="findings">ZNB1 key findings</h2>
<div class="findings">
  <div class="finding"><div class="k">{mnok(miss)} NOK</div><p>of 2% cash discount is missed on {num(late_n)} of {num(ci.posted)} posted invoices ({pct(late_n / ci.posted, 0)}) that took more than 10 days from goods receipt to posting.</p></div>
  <div class="finding"><div class="k">{pct(m1.share, 0)}</div><p>of ZNB1 invoices are posted manually, after a median of {m1['median']:.1f} days. They cause {pct(m1.missed / miss, 0)} of the missed discount. Automatically posted invoices are done the same day.</p></div>
  <div class="finding"><div class="k">{pct(r_.CTRL1, 0)}</div><p>of ZNB1 invoices are sent to the manual control queue (<code>CTRL1</code>) already at validation, so the manual posting is decided up front, not after a failed auto-post.</p></div>
  <div class="finding"><div class="k">{num(t.variants)}</div><p>variants for {num(t.cases)} POs. Mostly because a PO holds {p.mean_invoice_cycles:.1f} invoices on average and their steps interleave; real rework within an invoice is only {pct(p.rework_rate, 0)}.</p></div>
  <div class="finding"><div class="k">Cpu {ci.Cpu_soft10:.2f}</div><p>against the 10-day SLA, per invoice. ZNB1 isn't capable on any measure, and none of its three per-invoice KPIs is in statistical control.</p></div>
  <div class="finding"><div class="k">{pct(p.manual_share, 0)}</div><p>of all manual work in accounts payable comes from ZNB1, although it has only {pct(p.case_share, 1)} of the POs.</p></div>
</div>

<h2 id="data"><span class="step">Data preparation</span>What the log holds for ZNB1</h2>
<div class="prose">
<p>A <b>case</b> is one purchase order (<code>PO_ID</code>). A ZNB1 PO often receives several invoices: each new <code>SAP_PO_No</code> starts again with <i>Data management</i>. {pct(p.multi_cycle_rate, 0)} of ZNB1 POs hold more than one invoice, {num(n_inv)} invoices in total ({p.mean_invoice_cycles:.1f} per PO). {num(ci.not_posted)} invoices were never posted and are left out of the lead-time and discount figures.</p>
<ul class="tight">
  <li>Events are ordered by timestamp, then <code>Event_ID</code>. Throughput times are in fractional days from the timestamps.</li>
  <li>The log ends at posting, not payment. "SLA met" means posted within 10 (or 30) days.</li>
  <li>Case volume falls sharply in October and November 2025 across all PO types. That looks like an extraction effect and shows up in the control charts.</li>
</ul>
{note(f'''<p><b>Clock start at goods receipt.</b> For ZNB1 the payment term is assumed to start at goods receipt. The log has no goods-receipt date, so the first <i>{GR_ACTIVITY}</i> event with object code <code>{GR_OBJECT_CODE}</code> is used as a proxy. {num(gr_clock.sum())} invoices ({pct(gr_clock.mean(), 0)}) have that step. The others, mostly the manual-queue route, keep the clock from their first event. With the clock from arrival instead, {num(late_arr_n)} invoices would be late and {mnok(miss_arr)} NOK of discount missed (upper bound).</p>''', "Annotation · clock")}
</div>

<h2 id="step1"><span class="step">Step 1</span>Understand the ZNB1 workflow</h2>
<div class="prose"><p>Every invoice starts with <i>Data management → Initial validation</i>. At validation SAP sets a routing code that picks the road. On the <b>goods-receipt road</b> (<code>WAITGR</code>, {pct(r_.WAITGR, 0)} of invoices) the invoice waits for the goods receipt and usually posts automatically; {pct(RC.loc['WAITGR', 'manual'], 0)} of these are still posted manually. On the <b>manual road</b> (<code>CTRL1</code>, {pct(r_.CTRL1, 0)}) it goes to the manual control queue and waits for a clerk to post it. A few go via Warehouse Management first (<code>ZWMGRW</code>, {pct(r_.ZWMGRW, 1)}) or are deleted (<code>DELE</code>, {pct(r_.DELE, 1)}).</p>
<p>ZNB1 uses {num(t.activities)} different activities. The top variant covers {pct(t.top1_share, 0)} of the POs, the top five {pct(t.top5_share, 0)}, and it takes {num(t.variants_for_80pct)} variants to cover 80%.</p></div>
{fig(f"figures/process_maps/dfg_freq_{T}.png", f"ZNB1 frequency process map (filter: {t.map_filter}). Numbers are event and transition counts.", "wide")}
{fig(f"figures/process_maps/dfg_perf_{T}.png", "ZNB1 performance map: median time per transition.", "wide")}
<div class="example">
  <div class="lbl">Example · reading the process map</div>
  <p>Each <b>box</b> is an activity with how often it occurred, and each <b>arrow</b> counts how often one activity directly followed another. The arrow from <i>Initial validation</i> to <i>Posted manually</i> is the manual road. Arrows from <i>MM Posted Automatically</i> back to <i>Data management</i> are the next invoice on the same PO, not rework. The performance map uses the same layout but shows the median time on each arrow, so it shows where invoices wait.</p>
</div>
<h4>How often each activity occurs</h4>
{table(["Activity", "POs containing it", "Events"], pres_rows)}

<h3>Top 5 variants</h3>
{table(["#", "Variant", "POs", "Share", "Median days", "Manual acts"], vrows, "variants")}
<div class="example">
  <div class="lbl">Example · reading a variant</div>
  <p>Variant 1 (<i>Data management → Initial validation → Wait for goods receipt → MM Posted Automatically</i>) is the happy path: one invoice, goods-receipt check, automatic posting. Variant 2 is the manual road. Variants 3 and 4 are POs with two invoices, which is why the same steps appear twice.</p>
</div>

<h3>Lead time and missed discount by variant</h3>
<p class="prose">Lead time runs per invoice, from goods receipt (or the first event, see Data) to posting. An invoice posted after more than 10 days misses 2% of its value. The value is only known per PO, so it's split evenly over the PO's posted invoices.</p>
{table(["#", "Variant", "POs", "Invoices", "Median days", "Mean days", "Invoices > 10 d", "Missed 2% discount (NOK)"], lt_rows, "variants")}
<p class="prose">{pct(lt_other.missed_discount_nok / lt_all.missed_discount_nok, 0)} of the {mnok(lt_all.missed_discount_nok)} NOK sits outside the top five, in POs with {lt_other.invoices / lt_other.cases:.1f} invoices on average. Variants 3 and 4 show why the count runs per invoice: their POs take 20–25 days because a second invoice arrives weeks after the first, but each invoice is mostly posted within days.</p>

<h3 id="routing">Where the road is chosen: routing at validation</h3>
{note(f'''<p>Counted per invoice (first validation of each <code>SAP_PO_No</code>), with the lateness and missed discount of the posted ones:</p>
{table(["Routing code at validation", "Posted invoices", "Posted manually", "Invoices &gt; 10 d", "Late share", "Missed discount (NOK)", "Share of missed"], rc_rows)}
<p><b>"Posted manually" really is manual in ZNB1.</b> On the direct path <i>Data management → Initial validation → Posted manually</i> ({num(dz.cases)} POs), people make {pct(1 - dz.bot_share, 1)} of the postings, with a median of {dz.median_days_human:.1f} days and a 90th percentile of {dz.p90_days_human:.0f} days. The nightly batch job {BOT_USERS[1]}, which handles most of these postings in NB2, hardly touches ZNB1.</p>
<p><b>What the data can't say:</b> which validation rule sends an invoice to <code>CTRL1</code>. That needs the SAP configuration or the validation messages.</p>''')}

<h2 id="step2"><span class="step">Step 2</span>KPIs and who does the work</h2>
<div class="prose">
<p>The three KPIs per PO are <b>trace length</b> (number of events), <b>throughput time</b> (days from first to last event) and <b>manual activities</b> (events not made by a system user). The system users {BOT_USERS[0]} (system service) and {BOT_USERS[1]} (nightly batch) were identified by profiling all users across the whole log on workload, speed and timing; see the full report for that step.</p>
</div>
{table(["KPI", "ZNB1", "All PO types"], [
  f"<td>Mean trace length (events)</td><td class='n'>{num(p.mean_trace_length, 1)}</td><td class='n'>{num(S['events'] / S['cases'], 1)}</td>",
  f"<td>Median throughput (days)</td><td class='n'>{num(p.median_tp_days, 2)}</td><td class='n'>{num(CASES_ALL.throughput_days.median(), 2)}</td>",
  f"<td>Mean manual activities</td><td class='n'>{num(p.mean_manual, 2)}</td><td class='n'>{num(CASES_ALL.manual_count.mean(), 2)}</td>",
  f"<td>Touchless POs</td><td class='n'>{pct(p.touchless_rate, 0)}</td><td class='n'>{pct((CASES_ALL.manual_count == 0).mean(), 0)}</td>",
  f"<td>Invoices per PO</td><td class='n'>{p.mean_invoice_cycles:.2f}</td><td class='n'>{S['invoice_cycles'] / S['cases']:.2f}</td>",
  f"<td>Rework within an invoice</td><td class='n'>{pct(p.rework_rate, 0)}</td><td class='n'>{pct((pr.rework_rate * pr.cases).sum() / pr.cases.sum(), 0)}</td>",
  f"<td>Total manual events</td><td class='n'>{num(p.total_manual)}</td><td class='n'>{num(pr.total_manual.sum())}</td>",
])}
<h3>Who works on ZNB1</h3>
{table(["User", "ZNB1 events", "Share", "POs touched"], grp_rows)}
<p class="prose">{num(n_hum)} people work on ZNB1 invoices. The five busiest do {pct(top5_share, 0)} of the human events:</p>
{table(["User", "Events", "Share of human events", "POs", "Manual postings", "Most frequent activity"], hum_rows)}

<h3>Worked example: KPIs for two real ZNB1 POs</h3>
<div class="example">
  <div class="lbl">Example 1 · {EX_HAPPY} · happy path, one invoice</div>
  {example_case(EX_HAPPY)}
  <p><b>Trace length</b> = {int(xh.trace_length)} events. <b>Throughput time</b> = {num(xh.throughput_days, 2)} days. All steps were done by the system user, so <b>manual activities</b> = {int(xh.trace_length)} − {int(xh.bot_events)} = {int(xh.manual_count)}. The PO is touchless.</p>
</div>
<div class="example">
  <div class="lbl">Example 2 · {EX_REWORK} · three invoices</div>
  {example_case(EX_REWORK)}
  <p>At first glance this looks like rework: validation runs three times and the invoice is posted three times. The <code>SAP_PO_No</code> column shows otherwise. These are <b>three different invoices</b> for the same PO. The system validated each one within seconds, then each waited for a person ({ex_poster}) to post it manually.</p>
  <p><b>Trace length</b> = {int(xr.trace_length)} events. <b>Throughput time</b> = {num(xr.throughput_days, 1)} days. <b>Manual activities</b> = {int(xr.trace_length)} − {int(xr.bot_events)} system events = {int(xr.manual_count)}. The PO value is {num(xr.invoice_value_nok)} NOK.</p>
  <p>This PO is typical of ZNB1: no real rework, but invoices that sit for days before someone posts them.</p>
</div>

<h2 id="step3"><span class="step">Step 3</span>Control and capability</h2>
<div class="prose">
<p>The charts follow the invoice clock: one point per <b>posted invoice</b>, time from goods receipt (first <code>WAIT_GR</code> event) or, without one, from the first event to the <b>first posting</b>. The KPIs are events, days to first posting and manual events per invoice, all transformed with log(x + 1), which pulls in the long tail of slow invoices. Invoices are grouped by the week in which their clock started ({num(z_ttp.invoices)} invoices with clock start {cc_first:%d %b %Y} – {cc_last:%d %b %Y}; from {cc_fu:%d %b} only follow-up invoices of older POs, shaded), and each week is one subgroup on an <b>X̄–S chart</b>. The Western Electric rules are applied: <b>R1</b> one point beyond 3σ, <b>R2</b> 2 of 3 beyond 2σ, <b>R3</b> 4 of 5 beyond 1σ, <b>R4</b> 8 in a row on one side, <b>R5</b> 6 in a row rising or falling. A KPI is in control only if no point breaks a rule.</p>
</div>
{table(["KPI", "Weeks flagged", "Rules broken (X̄)", "S chart R1", "Weekly means as I-chart"], ctrl_rows)}
{fig(f"figures/control_charts/cc_{T}.png", "ZNB1: control charts per posted invoice, time on the invoice clock. Red points violate at least one rule.", "wide")}
<div class="example">
  <div class="lbl">Example · reading the days-to-posting chart</div>
  <p>Each dot on the middle X̄ chart is the <b>average log days to first posting of all invoices whose clock started in that week</b>; the centre line is e<sup>{z_ttp.center:.2f}</sup> − 1 = {z_ttp.center_days:.1f} days as a geometric mean. From late July to late September the weeks lie above the average, many beyond the upper limit (R1) and in long runs (R4). In October and November they sit below it. In the shaded follow-up period they climb back above it until early February; the last two weeks drop far below because those invoices had only days to be posted before the log ends. Part of that is mix: invoices on the goods-receipt clock post within a geometric mean of {cv['gr'].center_days:.1f} days, invoices on the arrival clock (mostly the <code>CTRL1</code> road) in {cv['arr'].center_days:.1f} days, and the goods-receipt share grows over the period.</p>
</div>
{fig(f"figures/control_charts/cc_{T}_by_clock.png", f"ZNB1 days to first posting by clock: headline invoice clock ({num(cv['head'].invoices)} invoices), only GR-clock invoices ({num(cv['gr'].invoices)}), only arrival-clock invoices ({num(cv['arr'].invoices)}), and the arrival clock for every invoice (upper bound).", "wide")}
<div class="callout"><p><b>How to read this:</b> with several hundred invoices a week the limits are narrow, so small shifts count as special causes. Charting the weekly means as individual values treats normal week-to-week variation as common cause. {wk_text} Even split by clock, both groups break the rules ({int(cv['gr'].points_flagged)} and {int(cv['arr'].points_flagged)} of {int(cv['gr'].subgroups)} weeks flagged), so the drift is not only a mix effect. Compare GR-clock and arrival-clock invoices with care: they differ in route as well as in clock (see the assumptions document §3).</p></div>

<h3 id="sla">Capability against the payment SLAs</h3>
<div class="prose"><p>Cpu = (log(USL + 1) − μ) / 3σ, with μ and σ from the log-transformed times and σ estimated within weeks. USL is 10 days (cash discount) and 30 days (legal payment deadline). Cpu ≥ 1 means the limit is met with a 3σ margin. Where the clock stops matters, because a PO can hold several invoices:</p></div>
<dl class="gloss">
  <dt>PO, last event</dt><dd>First to last event of the whole PO. Mixes several invoices.</dd>
  <dt>PO, first posting</dt><dd>Clock start to the first posting in the PO. Later invoices are ignored.</dd>
  <dt>Per invoice</dt><dd>For each invoice separately, from goods receipt (or first event) to its first posting. The closest match to a payment SLA, and the recommended measure.</dd>
</dl>
{table(["End point", "Median days", "Cpu 10 d", "Breach &gt; 10 d", "Cpu 30 d", "Breach &gt; 30 d"], [
  f"<td>PO, last event</td><td class='n'>{num(c1.median_tp_days, 2)}</td>{cpu_cell(c1.Cpu_soft10)}<td class='n'>{pct(c1.observed_breach_soft10)}</td>{cpu_cell(c1.Cpu_hard30)}<td class='n'>{pct(c1.observed_breach_hard30)}</td>",
  f"<td>PO, first posting</td><td class='n'>{num(cp.median_ttp_days, 2)}</td>{cpu_cell(cp.Cpu_soft10)}<td class='n'>{pct(cp.observed_breach_soft10)}</td>{cpu_cell(cp.Cpu_hard30)}<td class='n'>{pct(cp.observed_breach_hard30)}</td>",
  f"<td><b>Per invoice</b></td><td class='n'>{num(ci.median_ttp_days, 2)}</td>{cpu_cell(ci.Cpu_soft10)}<td class='n'><b>{pct(ci.observed_breach_soft10)}</b></td>{cpu_cell(ci.Cpu_hard30)}<td class='n'>{pct(ci.observed_breach_hard30)}</td>",
])}
<div class="example">
  <div class="lbl">Worked example · Cpu for the 10-day SLA (PO throughput time)</div>
  <p>The mean of log(throughput + 1) is μ = {c1.mean_log_tp:.3f}, a typical (geometric-mean) throughput of e<sup>μ</sup> − 1 = {np.expm1(c1.mean_log_tp):.1f} days. The limit on the same scale is log(10 + 1) = {usl10:.3f}.</p>
  <div class="math">Cpu = (USL − μ) / (3σ) = ({usl10:.3f} − {c1.mean_log_tp:.3f}) / (3 × {c1.sigma_within:.3f}) = {c1.Cpu_soft10:.3f}</div>
  <p>The average sits almost exactly on the deadline, so about half the POs miss it (observed {pct(c1.observed_breach_soft10, 0)}). Per invoice the picture is better (median {ci.median_ttp_days:.2f} days), but the spread is so wide that Cpu is still only {ci.Cpu_soft10:.2f}: the log distribution has a spike at same-day automatic posting and a long tail of manual postings, so prefer the observed breach rate.</p>
</div>
<div class="example">
  <div class="lbl">Example · the three end points for {EX_REWORK}</div>
  <p>PO, last event: {num(xr.throughput_days, 1)} days, a breach. PO, first posting: {num(xr.time_to_post_days, 1)} days. Per invoice: {", ".join(f"{d:.2f}" for d in EX_INV.values)} days, so {ex_late} of the {len(EX_INV)} invoices break the 10-day SLA.</p>
</div>
<p class="prose">The full report also shows the throughput distribution of ZNB1 next to the other PO types (<code>capability.png</code>, <code>throughput_distributions.png</code>).</p>

<h2 id="step4"><span class="step">Step 4</span>Where to focus within ZNB1</h2>
<div class="prose">
<p>The money is in the late invoices. Each late invoice misses 2% of its share of the PO value. For example, {EX_REWORK} was worth {num(xr.invoice_value_nok)} NOK; {ex_late} of its {len(EX_INV)} invoices were late, so it misses {num(xr.invoice_value_nok * 0.02 * ex_late / len(EX_INV))} NOK. Over all ZNB1 invoices that is <b>{mnok(miss)} NOK</b>. Counted per PO instead (the whole PO value lost once its last invoice is late) it would be {mnok(p.missed_discount_po_nok)} NOK, which overstates the loss.</p>
<p>By posting route:</p>
</div>
{table(["First posting", "Invoices", "Share", "Median days", "Invoices &gt; 10 d", "Late share", "Missed discount (NOK)"], route_rows)}
<div class="callout">
<p><b>Manual posting is the bottleneck.</b> {pct(m1.share, 0)} of ZNB1 invoices are posted by hand after a median of {m1['median']:.1f} days, and {pct(m1.late / m1.n, 0)} of those take longer than 10 days. They carry {pct(m1.missed / miss, 0)} of the missed discount. Most of this manual work is decided at validation: {pct(r_.CTRL1, 0)} of invoices go straight to <code>CTRL1</code>, and those invoices carry {pct(RC.loc['CTRL1', 'missed'] / miss, 0)} of the missed discount.</p>
</div>
<h3>Recommended scope</h3>
<ol class="rec">
  <li><b>Why invoices go to <code>CTRL1</code>.</b> Find which validation rule routes {pct(r_.CTRL1, 0)} of ZNB1 invoices to the manual queue, and whether some of them could take the goods-receipt road and post automatically. This needs the SAP validation configuration or messages, which aren't in the event log.</li>
  <li><b>The waiting time in the manual queue.</b> Once in the queue, invoices wait a median of {m1['median']:.1f} days for a person. {num(n_hum)} people work on ZNB1 and the busiest five do {pct(top5_share, 0)} of the human work. Queue order, workload and supplier assignment are the next things to look at, so that high-value invoices are posted before the 10-day discount deadline.</li>
  <li><b>Measure from goods receipt.</b> The clock assumption changes the result a lot ({mnok(miss)} vs. {mnok(miss_arr)} NOK). The real goods-receipt and payment dates from SAP would turn these estimates into exact figures.</li>
</ol>

<footer>
<p>Built with Python (conda env <code>datasci</code>) by <code>06_report_znb1.py</code> from the same outputs as the full report (<code>05_report.py</code> → <code>report.html</code>). Run steps 01–04 first. All tables come from the CSV files in <code>output/</code>; the routing-code and posting-route breakdowns are computed from <code>events.pkl</code> and <code>invoices.csv</code> with the same discount rule as <code>04_prioritise.py</code>.</p>
</footer>
</div>
"""
(OUT / "report_znb1.html").write_text(page, encoding="utf-8")
print("wrote", OUT / "report_znb1.html", len(page) // 1024, "KB")
print(f"check: missed discount {miss:,.0f} (priority.csv {p.missed_discount_nok:,.0f}), late {late_n} ({int(p.late_invoices)})")
print(f"arrival clock: late {late_arr_n}, missed {miss_arr:,.0f}")
