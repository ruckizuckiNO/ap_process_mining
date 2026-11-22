"""Builds output/report.html from the CSVs and figures produced by steps 01-04."""
import html
import json

import numpy as np
import pandas as pd

from common import OUT, PO_TYPES, BOT_USERS, POSTING_ACTIVITIES, load_events, load_cases, load_invoices

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
rcodes = pd.read_csv(OUT / "routing_codes.csv", index_col="code")
direct = pd.read_csv(OUT / "direct_posting.csv", index_col="po_type")
order = list(pr.index)  # by volume


def pct(x, d=1):
    return "–" if pd.isna(x) else f"{x * 100:.{d}f}%"


def num(x, d=0):
    return "–" if pd.isna(x) else f"{x:,.{d}f}"


def mnok(x):
    return f"{x / 1e6:,.1f}M"


def cpu_cell(v):
    cls = "bad" if v < 0.5 else "warn" if v < 1 else "ok"
    return f'<td class="n {cls}">{v:.2f}</td>'


def code(c):
    return f'<span class="code">{c}</span>'


def note(body, label="Annotation"):
    return f'<div class="note"><div class="lbl">{label}</div>{body}</div>'


def ctrl1(c):
    """Share of a PO type's invoices sent to the manual control queue at validation."""
    return pct(routing.loc[c, "CTRL1"], 0)


def table(head, rows, cls=""):
    th = "".join(f"<th>{h}</th>" for h in head)
    body = "".join(f"<tr>{r}</tr>" for r in rows)
    return f'<div class="tbl {cls}"><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def fig(src, cap_text, cls=""):
    return (f'<figure class="{cls}"><div class="plate"><img src="{src}" alt="{html.escape(cap_text)}" loading="lazy"></div>'
            f"<figcaption>{cap_text}</figcaption></figure>")


def variant_html(v):
    steps = v.split(" → ")
    return '<ol class="trace">' + "".join(f"<li>{html.escape(s)}</li>" for s in steps) + "</ol>"


def example_case(pid):
    """Event table for one real case, marking system users and the gap since the previous event."""
    e = EV[EV.PO_ID == pid]
    rows = []
    prev = None
    for i, (_, r) in enumerate(e.iterrows(), 1):
        gap = "" if prev is None else fmt_gap((r.ts - prev).total_seconds())
        prev = r.ts
        who = '<span class="pill sys">system</span>' if r.User_ID in BOT_USERS else '<span class="pill">human</span>'
        rows.append(f"<td class='n'>{i}</td><td class='n'>{r.ts:%d %b %H:%M:%S}</td><td class='n'>{gap}</td>"
                    f"<td>{html.escape(r.Activity_ID)}</td><td class='n'>{r.SAP_PO_No}</td><td>{r.User_ID} {who}</td>")
    return table(["#", "Timestamp", "Wait since previous", "Activity", "Invoice (SAP_PO_No)", "User"], rows)


def fmt_gap(s):
    if s < 120:
        return f"{s:.0f} s"
    if s < 2 * 3600:
        return f"{s / 60:.0f} min"
    if s < 2 * 86400:
        return f"{s / 3600:.1f} h"
    return f"{s / 86400:.1f} days"


# Two real cases used as worked examples: a ZNB2 happy path and a ZNB1 case holding three invoices
EX_HAPPY, EX_REWORK = "PurDoc230035", "PurDoc174751"
EV = load_events()
# Numbers for the "how to read a process map" example (NB2)
_nb2 = EV[EV.SAP_PO_Type_Code == "NB2"]
_nxt = _nb2.groupby("PO_ID").Activity_ID.shift(-1)
NB2_DM_EVENTS = int((_nb2.Activity_ID == "Data management").sum())
NB2_CASES = _nb2.PO_ID.nunique()
NB2_GR_SELF = int(((_nb2.Activity_ID == "Wait for goods receipt") & (_nxt == "Wait for goods receipt")).sum())
NB2_IV_PM = int(((_nb2.Activity_ID == "Initial validation") & (_nxt == "Posted manually")).sum())
NEG_VALUES = int((EV.Case_Invoice_Value_NOK < 0).sum())
# Sensitivity of the bot classification: events of the two judgment-call users per PO type
U280_BY_TYPE = EV[EV.User_ID == "User280"].groupby("SAP_PO_Type_Code").size()
U1550_EVENTS = int((EV.User_ID == "User1550").sum())
U1550_CASES = EV.loc[EV.User_ID == "User1550", "PO_ID"].nunique()
# How each invoice cycle was first posted, and how long that took
INV = load_invoices().set_index(["PO_ID", "SAP_PO_No"])
INV["route"] = EV[EV.Activity_ID.isin(POSTING_ACTIVITIES)].groupby(["PO_ID", "SAP_PO_No"]).Activity_ID.first()
_posted = INV[INV.posted]
ROUTE = _posted.groupby(["po_type", "route"]).time_to_post_days.agg(
    n="size", median="median", late=lambda x: (x > 10).mean())
ROUTE["share"] = ROUTE.n / ROUTE.groupby(level="po_type").n.transform("sum")
ROUTE = ROUTE.reset_index().set_index(["po_type", "route"])
EX_INV = INV.loc[EX_REWORK].time_to_post_days
EV = EV[EV.PO_ID.isin([EX_HAPPY, EX_REWORK])]
CASES = load_cases().set_index("PO_ID").loc[[EX_HAPPY, EX_REWORK]]
xh, xr = CASES.loc[EX_HAPPY], CASES.loc[EX_REWORK]

# ---------------------------------------------------------------- numbers used in the text
tot = pr.cases.sum()
disc_total = pr.missed_discount_nok.sum()
disc_focus = pr.loc[["ZJD1", "ZNB1"], "missed_discount_nok"].sum()
disc_po_total = pr.missed_discount_po_nok.sum()
ex_late = int((EX_INV > 10).sum())
manual_total = pr.total_manual.sum()
b1, b2 = bots.loc["User1587"], bots.loc["User280"]
# Typical busy human (median over human users with >= 500 events) for the comparison table
humans = bots[~bots.index.isin(BOT_USERS) & (bots.events >= 500)]
hum = humans.median(numeric_only=True)
# Simple rule-based cross-check of the manual classification
RULE = (bots.events >= 100) & ((bots.events_per_active_day > 150) | (bots.batch_0306_share > 0.5))
rule_users = list(bots.index[RULE])
nb2_share_280_human = (pr.loc["NB2", "total_manual"] + U280_BY_TYPE.get("NB2", 0)) / (manual_total + U280_BY_TYPE.sum())
def route(t, r="Posted manually"):
    return ROUTE.loc[(t, r)]


m1, mj = route("ZNB1"), route("ZJD1")
c1, c2 = cap.loc["ZNB1"], cap.loc["ZNB2"]
usl10 = np.log(11)
nb2_ev = types.loc["NB2"]
dn, da = direct.loc["NB2"], direct.loc["ALL"]


def human_share(c):
    b = direct.loc[c, "bot_share"]
    return "All" if b == 0 else "Almost all" if b < 0.005 else pct(1 - b, 0)

# ---------------------------------------------------------------- step 1
# Procurement situation per PO type, from the case description (see docs/domain.md)
SITUATION = {"ZJD1": "Standard purchasing", "ZNB1": "Standard purchasing", "ZJD3": "Standard purchasing",
             "ZNB2": "Third-party", "ZNB3": "Third-party", "ZNB4": "Third-party", "ZNB6": "Third-party",
             "ZNBF": "Transportation", "NB2": "Returns", "ZRPC": "Returns", "ZRPM": "Returns"}
# Suppliers: distinct per PO type, and in total (a supplier can deliver under several PO types)
SUP = pd.read_csv(OUT / "cases.csv", usecols=["po_type", "supplier"])
sup_total = SUP.supplier.nunique()
sup_multi = int((SUP.groupby("supplier").po_type.nunique() > 1).sum())
overview_rows = []
for c in order:
    t = types.loc[c]
    overview_rows.append(
        f"<td>{code(c)}</td><td>{t['name']}</td><td>{SITUATION[c]}</td><td class='n'>{num(t.cases)}</td><td class='n'>{pct(t.cases / tot)}</td>"
        f"<td class='n'>{num(pr.loc[c, 'suppliers'])}</td><td class='n'>{num(t.activities)}</td><td class='n'>{num(t.variants)}</td><td class='n'>{pct(t.top1_share, 0)}</td>"
        f"<td class='n'>{num(t.variants_for_80pct)}</td>")

type_notes = {
    "ZNB2": "Near-perfect straight-through flow: 95% of cases follow the single happy path (data management → validation → wait for GR → automatic posting), usually within minutes.",
    "NB2": f"Two main roads. About a third of the cases post automatically, but the most frequent variant (35%) ends in a manual posting. A goods-receipt loop through WM and 'Transfer to MM' restarts the invoice for about one in six cases. {pct(pr.loc['NB2', 'rework_rate'], 0)} of cases contain real rework, the highest share among the large PO types.",
    "ZNB4": f"Happy path for 74% of cases. {pct(pr.loc['ZNB4', 'multi_cycle_rate'], 0)} of cases hold more than one invoice, which stretches the case's throughput time. Taken invoice by invoice, {pct(route('ZNB4', 'MM Posted Automatically').share, 0)} post automatically and only {pct(cap_inv.loc['ZNB4', 'observed_breach_soft10'])} take longer than 10 days.",
    "ZNB3": "Very stable and fast: two variants cover 80% of cases and the throughput is almost always same-day.",
    "ZJD1": f"{pct(pr.loc['ZJD1', 'touchless_rate'], 0)} of cases are touchless. A case often holds several invoices ({pr.loc['ZJD1', 'mean_invoice_cycles']:.1f} on average), which makes traces look like repeated passes. Real rework within one invoice is rare ({pct(pr.loc['ZJD1', 'rework_rate'], 0)}). The slow part is manual posting: {pct(mj.share, 0)} of invoices are posted by hand, after a median of {mj['median']:.1f} days.",
    "ZNB1": f"The least standardised high-value workflow: 1,161 variants for 4,753 cases. That is mostly because a case holds {pr.loc['ZNB1', 'mean_invoice_cycles']:.1f} invoices on average and their steps interleave. Real rework within one invoice is only {pct(pr.loc['ZNB1', 'rework_rate'], 0)}. {pct(m1.share, 0)} of invoices are posted manually, after a median of {m1['median']:.1f} days, while the automatically posted ones are done the same day.",
    "ZJD3": f"Despite its name, only {pct(pr.loc['ZJD3', 'touchless_rate'], 0)} of automatic orders are touchless. Cases hold {pr.loc['ZJD3', 'mean_invoice_cycles']:.1f} invoices on average, and half of the invoices are posted manually (median {route('ZJD3')['median']:.1f} days).",
    "ZNB6": f"Small but slow: median 29 days, with half the invoices beyond the 30-day payment deadline. {pct(pr.loc['ZNB6', 'rework_rate'], 0)} of cases contain real rework.",
    "ZRPC": "Mostly a short manual-posting flow; a quarter of the cases breach the 10-day SLA.",
    "ZNBF": "Only 34 cases, nearly all with rework and manual handling; median 41 days.",
    "ZRPM": "Only 5 cases, too few to characterise a process.",
}

# Routing findings per type (see "Routing at validation" in Step 1)
type_annot = {
    "NB2": f"Variant 1 (direct manual posting) is mostly automated after all: {pct(dn.bot_share, 0)} of its {num(dn.cases)} postings come from the nightly batch {BOT_USERS[1]} (median {dn.median_days_bot:.1f} days after validation), {pct(1 - dn.bot_share, 0)} from people (median {dn.median_days_human:.1f} days). {ctrl1('NB2')} of NB2 invoices are routed to the manual queue <code>CTRL1</code> at validation, and {pct(routing.loc['NB2', 'ZWMGRW'], 0)} to a goods receipt from WM (<code>ZWMGRW</code>), the start of the WM loop in variant 3.",
    "ZJD1": f"{ctrl1('ZJD1')} of ZJD1 invoices are sent to the manual queue (<code>CTRL1</code>) during validation, before any goods-receipt check. {human_share('ZJD1')} of the direct manual postings are made by people, not by the batch job.",
    "ZNB1": f"{ctrl1('ZNB1')} of ZNB1 invoices are sent to the manual queue (<code>CTRL1</code>) during validation. Most of the manual posting is decided there, not after a failed auto-post. {human_share('ZNB1')} of the direct manual postings are made by people (median {direct.loc['ZNB1', 'median_days_human']:.1f} days).",
    "ZJD3": f"{ctrl1('ZJD3')} of ZJD3 invoices are routed to the manual queue (<code>CTRL1</code>) at validation, even though the PO type is meant to be automatic.",
    "ZNB6": f"{ctrl1('ZNB6')} of ZNB6 invoices go to the manual queue (<code>CTRL1</code>) at validation. Variant 2 (direct manual posting) then waits a median of {direct.loc['ZNB6', 'median_days_human']:.0f} days for a person, the longest human queue in the data. Whether the clerks are holding invoices until the goods arrive can't be checked, because the log has no goods-receipt date.",
    "ZRPC": f"{ctrl1('ZRPC')} of ZRPC invoices are routed to the manual queue (<code>CTRL1</code>) at validation, and {human_share('ZRPC')} of the direct manual postings are made by people. Unlike NB2, the other return type, ZRPC returns aren't handled by the nightly batch.",
}

routing_rows = []
for c in order:
    r = routing.loc[c]
    cls = "bad" if r.CTRL1 >= 0.5 else "warn" if r.CTRL1 >= 0.25 else ""
    routing_rows.append(f"<td>{code(c)}</td><td class='n {cls}'>{pct(r.CTRL1, 0)}</td><td class='n'>{pct(r.WAITGR, 0)}</td>"
                        f"<td class='n'>{pct(r.ZWMGRW, 0)}</td><td class='n'>{pct(r.DELE, 0)}</td>")


def rc(c):
    return pct(rcodes.loc[c, "share"], 0)


# ZNB1: invoice lead time and the discount missed on invoices posted after 10 days, by top-5 variant
lt = pd.read_csv(OUT / "lead_time_variants.csv", index_col="variant_rank")
LT_TYPE = "ZNB1"
lt_var = variants[variants.po_type == LT_TYPE].set_index("rank").variant
lt_rows = []
for k, label in [*((str(i), None) for i in range(1, 6)), ("0", "Other variants"), ("ALL", f"All {LT_TYPE}")]:
    r = lt.loc[k]
    first = (f"<td class='n'>{k}</td><td>{variant_html(lt_var[int(k)])}</td>" if label is None
             else f"<td></td><td><b>{label}</b></td>")
    lt_rows.append(f"{first}<td class='n'>{num(r.cases)}</td><td class='n'>{num(r.invoices)}</td>"
                   f"<td class='n'>{num(r.median_days, 1)}</td><td class='n'>{num(r.mean_days, 1)}</td>"
                   f"<td class='n'>{num(r.late_invoices)}</td><td class='n'>{num(r.missed_discount_nok)}</td>")
lt_all, lt_other = lt.loc["ALL"], lt.loc["0"]
lead_time_html = f"""
    <h4>Lead time and missed discount by variant</h4>
    <p>Lead time runs per invoice, from its first event to its posting; for ZNB1 the clock starts at goods receipt when the invoice waits for one (see Step 3). An invoice posted after more than 10 days misses 2% of its value. The value is only known per PO, so it is split evenly over the PO's posted invoices (see Step 4). Invoices that were never posted are left out.</p>
    {table(["#", "Variant", "POs", "Invoices", "Median days", "Mean days", "Invoices > 10 d", "Missed 2% discount (NOK)"], lt_rows, "variants")}
    <p>{pct(lt_other.missed_discount_nok / lt_all.missed_discount_nok, 0)} of the {mnok(lt_all.missed_discount_nok)} NOK sits outside the top five, in POs with {lt_other.invoices / lt_other.cases:.1f} invoices on average. The top five miss only {mnok(lt_all.missed_discount_nok - lt_other.missed_discount_nok)} NOK. Variants 3 and 4 show why the count runs per invoice: their cases take 20–25 days because a second invoice arrives weeks after the first, but each invoice is mostly posted within days. Counted per PO they would miss {mnok(lt.loc[['3', '4'], 'missed_discount_po_nok'].sum())} NOK, per invoice {mnok(lt.loc[['3', '4'], 'missed_discount_nok'].sum())} NOK.</p>"""

per_type = []
for c in order:
    t = types.loc[c]
    v = variants[variants.po_type == c].head(5)
    vrows = [f"<td class='n'>{int(r['rank'])}</td><td>{variant_html(r.variant)}</td><td class='n'>{num(r.cases)}</td>"
             f"<td class='n'>{pct(r.share)}</td><td class='n'>{num(r.median_tp_days, 1)}</td><td class='n'>{num(r.mean_manual, 2)}</td>"
             for _, r in v.iterrows()]
    per_type.append(f"""
<details class="type" id="map-{c}">
  <summary>{code(c)} <b>{t['name']}</b><span class="meta">{num(t.cases)} cases · {num(t.variants)} variants</span></summary>
  <div class="type-body">
    <p>{type_notes.get(c, '')}</p>
    {note(f"<p>{type_annot[c]}</p>") if c in type_annot else ""}
    {fig(f"figures/process_maps/dfg_freq_{c}.png", f"{c}: frequency process map (filter: {t.map_filter}). Numbers are event and transition counts.", "wide")}
    {fig(f"figures/process_maps/dfg_perf_{c}.png", f"{c}: performance map, median time per transition.", "wide")}
    <h4>Top 5 variants</h4>
    {table(["#", "Variant", "Cases", "Share", "Median days", "Manual acts"], vrows, "variants")}{lead_time_html if c == LT_TYPE else ""}
  </div>
</details>""")

# ---------------------------------------------------------------- step 2
bot_cols = ["events", "cases", "activities", "events_per_active_day", "median_gap_s", "night_share", "batch_0306_share"]
brows = []
for u in list(bots.index[:6]) + ["User1550"]:
    r = bots.loc[u]
    tag = '<span class="pill sys">system</span>' if u in BOT_USERS else '<span class="pill">human</span>'
    brows.append(f"<td>{u} {tag}</td><td class='n'>{num(r.events)}</td><td class='n'>{num(r.cases)}</td><td class='n'>{num(r.activities)}</td>"
                 f"<td class='n'>{num(r.events_per_active_day)}</td><td class='n'>{num(r.median_gap_s / 3600, 2)}</td>"
                 f"<td class='n'>{pct(r.night_share, 0)}</td><td class='n'>{pct(r.batch_0306_share, 0)}</td>")

kpi_rows = []
for c in order:
    r = pr.loc[c]
    kpi_rows.append(f"<td>{code(c)}</td><td class='n'>{num(r.mean_trace_length, 1)}</td><td class='n'>{num(r.median_tp_days, 2)}</td>"
                    f"<td class='n'>{num(r.mean_manual, 2)}</td><td class='n'>{pct(r.touchless_rate, 0)}</td><td class='n'>{r.mean_invoice_cycles:.2f}</td><td class='n'>{pct(r.rework_rate, 0)}</td>")

# ---------------------------------------------------------------- step 3
ctrl_rows = []
for c in order:
    sub = ctrl[ctrl.po_type == c].set_index("kpi")
    cells = []
    for k in ["trace_length", "throughput_days", "manual_count"]:
        r = sub.loc[k]
        flagged, n = int(r.points_flagged), int(r.subgroups)
        rules = ", ".join(rr for rr in ["R1", "R2", "R3", "R4", "R5"] if r[rr] > 0) or "none"
        status = "ok" if r.in_control else "bad"
        wk = "" if pd.isna(r.weekly_I_in_control) else (
            ' <span class="pill ok">weekly-means OK</span>' if r.weekly_I_in_control in (True, "True") else "")
        cells.append(f"<td class='{status}'>{flagged}/{n} flagged<br><small>{rules}</small>{wk}</td>")
    ctrl_rows.append(f"<td>{code(c)}</td><td><small>{sub.chart.iloc[0]}</small></td>{''.join(cells)}")

cap_rows = []
for c in order:
    r = cap.loc[c]
    cap_rows.append(f"<td>{code(c)}</td><td class='n'>{num(r.cases)}</td><td class='n'>{num(r.median_tp_days, 2)}</td><td class='n'>{num(r.p90_tp_days, 1)}</td>"
                    f"{cpu_cell(r.Cpu_soft10)}<td class='n'>{pct(r.observed_breach_soft10)}</td>"
                    f"{cpu_cell(r.Cpu_hard30)}<td class='n'>{pct(r.observed_breach_hard30)}</td>")

cc_details = "".join(
    f'<details class="type"><summary>{code(c)} <b>{PO_TYPES[c]}</b><span class="meta">{ctrl[ctrl.po_type == c].chart.iloc[0]}</span></summary>'
    f'<div class="type-body">{fig(f"figures/control_charts/cc_{c}.png", f"{c}: control charts for the three log-transformed KPIs. Red points violate at least one rule.", "wide")}</div></details>'
    for c in order)

# ---------------------------------------------------------------- step 4
prio_rows = []
for c in order:
    r = pr.loc[c]
    prio_rows.append(
        f"<td>{code(c)}</td><td class='n'>{pct(r.case_share)}</td><td class='n'>{pct(r.manual_share)}</td>"
        f"<td class='n'>{num(cap_inv.loc[c, 'cases_breach_soft10'])}</td><td class='n'>{pct(cap_inv.loc[c, 'cases_breach_soft10'] / cap_inv.cases_breach_soft10.sum())}</td>"
        f"<td class='n'>{mnok(r.missed_discount_nok)}</td>{cpu_cell(cap_inv.loc[c, 'Cpu_soft10'])}"
        f"<td class='n'>{int(r.xbar_in_control_kpis)}/3</td>")

# ---------------------------------------------------------------- page
page = f"""<title>Invoice Flow Screening</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,500;6..72,600&family=Instrument+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
/* Layout: single reading column (~72ch) with wide figure/table breakouts; steps follow the assignment order */
:root {{
  --bg: #f6f8f7; --surface: #ffffff; --ink: #16201c; --muted: #5a6a63; --line: #d9e1dd;
  --accent: #1f6f5c; --accent-soft: #e3efea; --ok: #2c7a4b; --warn: #a8670f; --bad: #b0333f;
  --ok-bg: #e5f2ea; --warn-bg: #fbf0dc; --bad-bg: #f8e4e6; --plate: #ffffff;
  --f-display: "Newsreader", Georgia, "Times New Roman", serif;
  --f-body: "Instrument Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --f-mono: "JetBrains Mono", ui-monospace, Menlo, Consolas, monospace;
}}
/* Light theme only, regardless of the viewer's system setting */
:root, :root[data-theme="dark"] {{ color-scheme: light; }}
body {{ background: var(--bg); color: var(--ink); font: 15.5px/1.6 var(--f-body); }}
.wrap {{ max-width: 1080px; margin: 0 auto; padding-block: 40px 80px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }}
.prose {{ max-width: 72ch; }}
header {{ border-bottom: 1px solid var(--line); padding-bottom: 24px; margin-bottom: 8px; }}
.eyebrow {{ font: 500 12px/1 var(--f-mono); letter-spacing: .08em; text-transform: uppercase; color: var(--accent); }}
h1 {{ font: 600 clamp(32px, 5vw, 46px)/1.1 var(--f-display); margin: 12px 0 10px; text-wrap: balance; }}
h2 {{ font: 600 28px/1.2 var(--f-display); margin: 56px 0 6px; text-wrap: balance; }}
h2 .step {{ font: 500 13px/1 var(--f-mono); color: var(--accent); display: block; margin-bottom: 8px; letter-spacing: .06em; text-transform: uppercase; }}
h3 {{ font: 600 18px/1.3 var(--f-body); margin: 32px 0 6px; }}
h4 {{ font: 600 14px/1.3 var(--f-body); margin: 20px 0 6px; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; }}
p, li {{ color: var(--ink); }}
.lede {{ font-size: 17px; color: var(--muted); max-width: 70ch; margin: 0; }}
.facts {{ display: flex; flex-wrap: wrap; gap: 8px 28px; margin-top: 18px; font: 13px/1.4 var(--f-mono); color: var(--muted); }}
.facts b {{ color: var(--ink); font-weight: 500; }}
.findings {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 300px), 1fr)); gap: 14px; margin-top: 18px; }}
.finding {{ background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 16px 18px; min-width: 0; }}
.finding .k {{ font: 600 26px/1.1 var(--f-display); color: var(--accent); font-variant-numeric: tabular-nums; }}
.finding p {{ margin: 6px 0 0; font-size: 14.5px; }}
.callout {{ background: var(--accent-soft); border-radius: 8px; padding: 14px 18px; margin: 18px 0; max-width: 80ch; }}
.callout p {{ margin: 4px 0; }}
.code {{ font: 500 12.5px/1 var(--f-mono); background: var(--accent-soft); color: var(--accent); padding: 3px 6px; border-radius: 4px; white-space: nowrap; }}
.tbl {{ overflow-x: auto; margin: 14px 0; border: 1px solid var(--line); border-radius: 8px; background: var(--surface); }}
table {{ border-collapse: collapse; width: 100%; font-size: 13.5px; font-variant-numeric: tabular-nums; }}
th {{ text-align: left; font: 600 11.5px/1.3 var(--f-body); text-transform: uppercase; letter-spacing: .04em; color: var(--muted);
  padding: 10px 12px; border-bottom: 1px solid var(--line); vertical-align: bottom; }}
td {{ padding: 8px 12px; border-bottom: 1px solid var(--line); vertical-align: top; }}
tr:last-child td {{ border-bottom: 0; }}
td.n {{ text-align: right; font-family: var(--f-mono); font-size: 12.5px; white-space: nowrap; }}
td.ok {{ background: var(--ok-bg); }} td.warn {{ background: var(--warn-bg); }} td.bad {{ background: var(--bad-bg); }}
td.n.ok {{ color: var(--ok); }} td.n.warn {{ color: var(--warn); }} td.n.bad {{ color: var(--bad); }}
td small {{ color: var(--muted); font-family: var(--f-mono); font-size: 11px; }}
.pill {{ display: inline-block; font: 500 10.5px/1 var(--f-mono); padding: 3px 6px; border-radius: 99px; border: 1px solid var(--line); color: var(--muted); vertical-align: 1px; }}
.pill.sys {{ border-color: var(--accent); color: var(--accent); }}
.pill.ok {{ border-color: var(--ok); color: var(--ok); margin-top: 4px; }}
figure {{ margin: 18px 0; }}
.plate {{ background: var(--plate); border: 1px solid var(--line); border-radius: 8px; padding: 10px; overflow-x: auto; }}
.plate img {{ display: block; max-width: 100%; height: auto; margin: 0 auto; }}
figcaption {{ font-size: 13px; color: var(--muted); margin-top: 6px; }}
.two {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 420px), 1fr)); gap: 16px; }}
.two > * {{ min-width: 0; }}
details.type {{ border: 1px solid var(--line); border-radius: 8px; background: var(--surface); margin: 8px 0; }}
details.type > summary {{ cursor: pointer; padding: 12px 16px; display: flex; flex-wrap: wrap; align-items: center; gap: 8px 12px; list-style: none; }}
details.type > summary::-webkit-details-marker {{ display: none; }}
details.type > summary::before {{ content: "▸"; color: var(--accent); transition: transform .15s; }}
details.type[open] > summary::before {{ transform: rotate(90deg); }}
details.type > summary:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; border-radius: 8px; }}
details.type .meta {{ margin-left: auto; font: 12px var(--f-mono); color: var(--muted); }}
.type-body {{ padding: 0 16px 16px; border-top: 1px solid var(--line); }}
ol.trace {{ list-style: none; margin: 0; padding: 0; display: flex; flex-wrap: wrap; gap: 4px; }}
ol.trace li {{ font-size: 12px; background: var(--accent-soft); border-radius: 4px; padding: 2px 6px; }}
ol.trace li + li::before {{ content: "→ "; color: var(--muted); }}
.variants td:nth-child(2) {{ min-width: 280px; }}
ul.tight li {{ margin: 4px 0; }}
.rec {{ counter-reset: rec; list-style: none; padding: 0; display: grid; gap: 12px; max-width: 80ch; }}
.rec li {{ counter-increment: rec; background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 14px 18px 14px 56px; position: relative; }}
.rec li::before {{ content: counter(rec); position: absolute; left: 18px; top: 12px; font: 600 24px/1 var(--f-display); color: var(--accent); }}
dl.gloss {{ display: grid; grid-template-columns: minmax(120px, max-content) 1fr; gap: 8px 20px; margin: 14px 0; max-width: 90ch; }}
dl.gloss dt {{ font-weight: 600; }}
dl.gloss dd {{ margin: 0; color: var(--ink); }}
@media (max-width: 560px) {{ dl.gloss {{ grid-template-columns: 1fr; gap: 2px; }} dl.gloss dd {{ margin-bottom: 10px; }} }}
.example {{ border: 1px dashed var(--accent); border-radius: 8px; padding: 14px 18px; margin: 18px 0; background: var(--surface); max-width: 90ch; }}
.example > .lbl {{ font: 500 11.5px/1 var(--f-mono); letter-spacing: .08em; text-transform: uppercase; color: var(--accent); }}
.example p {{ margin: 8px 0; }}
.example .tbl {{ margin: 10px 0; }}
.math {{ font: 13px/1.7 var(--f-mono); background: var(--accent-soft); border-radius: 6px; padding: 10px 14px; overflow-x: auto; white-space: pre; }}
tr.sys td {{ color: var(--muted); }}
.toc {{ display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: 14px; margin: 14px 0 0; padding: 0; list-style: none; }}
.toc a {{ color: var(--accent); text-decoration: none; border-bottom: 1px solid var(--line); }}
.toc a:hover, .toc a:focus-visible {{ border-bottom-color: var(--accent); }}
code {{ font: 12.5px var(--f-mono); background: var(--accent-soft); padding: 1px 5px; border-radius: 3px; }}
footer {{ margin-top: 64px; border-top: 1px solid var(--line); padding-top: 18px; font-size: 13px; color: var(--muted); }}
/* Annotations from the routing analysis: amber rule so they read as later notes */
.note {{ border-left: 3px solid var(--warn); background: var(--warn-bg); border-radius: 0 8px 8px 0; padding: 10px 16px; margin: 14px 0; max-width: 90ch; }}
.note > .lbl {{ font: 500 11px/1 var(--f-mono); letter-spacing: .08em; text-transform: uppercase; color: var(--warn); }}
.note p {{ margin: 6px 0; }}
.note .tbl {{ margin: 10px 0; }}
.finding.new {{ border-color: var(--warn); }}
.finding.new .k {{ color: var(--warn); }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; }} }}
</style>

<div class="wrap">
<header>
  <div class="eyebrow">Accounts payable · Process mining case · Steps 1–4</div>
  <h1>Invoice Flow Screening</h1>
  <p class="lede">How the eleven PO-type workflows for supplier invoices behave, how efficient each one is, whether they are in statistical control, and where an improvement project would pay off most.</p>
  <div class="facts">
    <span><b>{num(S['cases'])}</b> invoice cases</span>
    <span><b>{num(S['events'])}</b> events</span>
    <span><b>{S['activities']}</b> activities</span>
    <span><b>{S['users']}</b> users</span>
    <span><b>{num(sup_total)}</b> suppliers</span>
    <span>cases started <b>Jul–Nov 2025</b>, events until <b>3 Mar 2026</b></span>
  </div>
</header>

<ul class="toc">
  <li><a href="#guide">How to read this report</a></li>
  <li><a href="#findings">Key findings</a></li>
  <li><a href="#data">Data</a></li>
  <li><a href="#step1">1 · Workflows</a></li>
  <li><a href="#routing">Routing at validation</a></li>
  <li><a href="#step2">2 · KPIs</a></li>
  <li><a href="#step3">3 · Control &amp; capability</a></li>
  <li><a href="#step4">4 · Focus</a></li>
</ul>

<h2 id="guide"><span class="step">Before you start</span>How to read this report</h2>
<div class="prose">
<p>The report follows the four steps of the assignment. Each step first says what was done and why, then shows the results, then gives a worked example from a real invoice. You don't need to know process mining or statistics in advance. The terms below are used throughout.</p>
</div>
<dl class="gloss">
  <dt>Case</dt><dd>One invoice handled against one purchase-order document (<code>PO_ID</code>). Everything that happens to it, from arrival to posting, belongs to the same case.</dd>
  <dt>Event / activity</dt><dd>One recorded step in a case, such as <i>Initial validation</i> or <i>Posted manually</i>, with a timestamp and the user who did it. The <b>activity</b> is the name of the step.</dd>
  <dt>Trace</dt><dd>The ordered list of activities in one case.</dd>
  <dt>Variant</dt><dd>All cases with exactly the same trace. A process with few variants is standardised. Many variants mean many different paths, which usually means exceptions and rework.</dd>
  <dt>Process map (DFG)</dt><dd>A directly-follows graph. Boxes are activities and arrows show which activity came directly after which, with counts.</dd>
  <dt>PO type</dt><dd>The SAP purchase-order category (e.g. <code>ZNB2</code> third-party order). Different PO types follow different workflows, so they are analysed separately. The 11 types in the data cover four procurement situations: <b>standard purchasing</b> of goods for stock (ZJD1 normal, ZNB1 manual, ZJD3 automatic replenishment), <b>third-party orders</b> where the supplier ships directly to the customer (ZNB2/3/4/6, dropshipping), <b>transportation</b> services (ZNBF) and <b>returns</b> of goods already bought (NB2, ZRPC, ZRPM). The service order types ZINS and ZSER from the assignment do not occur in the log.</dd>
  <dt>System user (bot)</dt><dd>A user ID that is a program, not a person. Its events cost no manual effort.</dd>
  <dt>Touchless</dt><dd>A case where only system users did anything: zero manual activities.</dd>
  <dt>Invoice cycle</dt><dd>One invoice document within a case, identified by <code>SAP_PO_No</code>. A purchase order can receive several invoices (e.g. for partial deliveries), so one case can contain several cycles. Each cycle starts again with <i>Data management</i>.</dd>
  <dt>Rework</dt><dd>An activity that happens more than once <b>within the same invoice cycle</b>, e.g. the same invoice is validated twice. Repeats across different invoices of the same PO are not rework.</dd>
  <dt>GR, MM, WM, FI</dt><dd>SAP terms: goods receipt (proof that the goods arrived), Materials Management, Warehouse Management and Financial Accounting. Invoices that refer to a purchase order go to <b>MM</b>, where they are matched against the PO and the goods receipt. Invoices without a PO reference go to <b>FI</b> for account coding.</dd>
  <dt>Routing code</dt><dd>The value of <code>SAP_Next_Processing_Type</code> on an event: the queue SAP sends the invoice to next. On <i>Initial validation</i> it decides the road: <code>WAITGR</code> (goods-receipt check, then automatic posting), <code>ZWMGRW</code> (goods receipt from WM first), <code>CTRL1</code> (manual control queue) or <code>DELE</code> (deletion). See <a href="#routing">Routing at validation</a>.</dd>
  <dt>SLA</dt><dd>Service level: here a deadline for finishing an invoice. 10 days to get the 2% cash discount, 30 days for the legal payment deadline.</dd>
  <dt>Control chart</dt><dd>A time chart of a KPI with a centre line and limits at ±3 standard deviations. Points outside the limits, or suspicious patterns, mean something changed in the process.</dd>
  <dt>Cpu</dt><dd>Process capability against an upper limit: how many "3-sigma widths" fit between the average and the deadline. Cpu ≥ 1 is good, below 1 means many cases break the deadline.</dd>
</dl>

<h2 id="findings">Key findings</h2>
<div class="findings">
  <div class="finding"><div class="k">87%</div><p>of invoices are processed fully touchless. Two system users ({BOT_USERS[0]}, {BOT_USERS[1]}) perform {pct(S['bot_event_share'], 1)} of all events.</p></div>
  <div class="finding"><div class="k">{code('ZNB1')} {code('ZJD1')}</div><p>Manual and normal orders are only 5.5% of the cases but carry the highest-value invoices. Invoice by invoice, {pct(cap_inv.loc['ZNB1', 'observed_breach_soft10'], 0)} and {pct(cap_inv.loc['ZJD1', 'observed_breach_soft10'], 0)} take longer than 10 days, mostly those waiting for manual posting. They account for {pct(disc_focus / disc_total, 0)} of the missed cash discount.</p></div>
  <div class="finding"><div class="k">{mnok(disc_total)} NOK</div><p>of 2% cash discount is missed on {num(pr.late_invoices.sum())} invoices that were posted after more than 10 days.</p></div>
  <div class="finding"><div class="k">{code('NB2')} {pct(pr.loc['NB2', 'manual_share'], 0)}</div><p>of all manual work comes from returns to vendor. They meet the SLAs well, so this is a cost problem, not a deadline problem.</p></div>
  <div class="finding"><div class="k">10 of 11</div><p>processes are out of control on the weekly X̄–S charts. The exception, ZRPM, has only 5 cases. A large part of the signal is a drift in the case population from October onward, so read these verdicts together with the data caveats.</p></div>
  <div class="finding new"><div class="k">At validation</div><p>the road is already chosen. {rc('CTRL1')} of validations send the invoice straight to the manual control queue (<code>CTRL1</code>) and skip the goods-receipt check. That is {ctrl1('ZNB1')} of ZNB1 invoices and {ctrl1('ZJD1')} of ZJD1 invoices, so their manual posting is decided up front, not after a failed auto-post. <a href="#routing">Details</a></p></div>
</div>

<h2 id="data"><span class="step">Data preparation</span>What the event log contains</h2>
<div class="prose">
<p>Each <b>case</b> is one <code>PO_ID</code>, the invoice processing for a purchase-order document. Events are ordered by timestamp. A case can contain several <b>invoice cycles</b>: each new <code>SAP_PO_No</code> starts again with <i>Data management</i>, and <code>Item_No</code> restarts at 1. {num(S['cases_multi_cycle'])} cases ({pct(S['cases_multi_cycle'] / S['cases'], 0)}) hold more than one cycle, {num(S['invoice_cycles'])} cycles in total. That is also why <code>Item_No</code> cannot be used to order a whole case. Throughput times are calculated from the timestamps in fractional days, not taken from the pre-computed whole-day column.</p>
<ul class="tight">
  <li>{S['cases_untyped_dropped']} cases without a PO type were removed, leaving {num(S['cases'])} cases.</li>
  <li>The data contains <b>11</b> PO types, not twelve. The assignment's own table also lists 11 codes.</li>
  <li>Case volume falls sharply in October and November 2025 (14k and 3.5k cases, against about 60k in September), and those later cases are slower and more manual. This looks like an extraction effect: probably only cases that reached a certain state were included. It causes a lot of the late-period signals in the control charts.</li>
  <li>{num(S['incomplete_cases'])} cases (1%) do not end in a posting or deletion, so they may be unfinished. They are kept, because removing them would hide slow cases.</li>
</ul>
{note(f'''<p><b>No goods-receipt date.</b> The log records when SAP <i>checks</i> for the goods receipt, not when the goods arrived. Waiting for goods outside the automatic path (e.g. a clerk holding an invoice) can't be measured; it shows up only as a long gap before <i>Posted manually</i>.</p>
<p><b>{"No credit notes by sign." if NEG_VALUES == 0 else "Few negative values."}</b> {"No invoice value is negative, including all return types (NB2, ZRPC, ZRPM). Credit notes can't be told apart from invoices in this data." if NEG_VALUES == 0 else f"Only {num(NEG_VALUES)} events carry a negative invoice value, so credit notes can't be reliably told apart from invoices."}</p>
<p><b>Routing field:</b> <code>SAP_Next_Processing_Type</code> shows where SAP routes the invoice after each step. It explains why some invoices never wait for the goods receipt (see <a href="#routing">Routing at validation</a>).</p>''', "Annotation · data limits")}
</div>

<h2 id="step1"><span class="step">Step 1</span>Understand the workflows</h2>
<div class="prose"><p>Every case starts with <i>Data management → Initial validation</i>. The happy path then waits for the goods receipt (GR) and posts automatically in MM. Deviations come from three places: <b>GR problems</b> (repeated waiting, retrieving the GR from WM, transferring to MM and restarting), <b>manual posting</b> (after a failed auto-post or directly after validation), and <b>exception handling</b> (approval for price differences, asking for comments, deletion). The PO types mix these in very different proportions.</p></div>
{table(["Code", "PO type", "Situation", "Cases", "Share", "Suppliers", "Activities", "Variants", "Top variant", "Variants for 80%"], overview_rows)}
<p class="prose">The log has {num(sup_total)} suppliers in total. {num(sup_multi)} of them deliver under more than one PO type, so the supplier column adds up to more than the total. The third-party types run their volume through few suppliers ({code('ZNB2')}: {num(pr.loc['ZNB2', 'cases'])} cases from {num(pr.loc['ZNB2', 'suppliers'])} suppliers), while the large standard-purchasing and return types (ZNB1, ZJD1, ZJD3, NB2) spread over {num(pr.loc[['ZNB1', 'ZJD1', 'ZJD3', 'NB2'], 'suppliers'].min())}–{num(pr.loc[['ZNB1', 'ZJD1', 'ZJD3', 'NB2'], 'suppliers'].max())} suppliers each.</p>
{fig("figures/activity_presence.png", "Share of cases in each PO type that contain each activity. The third-party types (ZNB2–4) rarely leave the happy path, while ZNB1, ZJD3, ZNB6 and ZNBF show many repeated activities, from real rework or from several invoices per case.", "wide")}
<div class="example">
  <div class="lbl">Example · reading a process map</div>
  <p>Take the <code>NB2</code> (Return to vendor) frequency map below. Each <b>box</b> shows an activity and how often it occurred. <i>Data management</i> occurs {num(NB2_DM_EVENTS)} times in {num(NB2_CASES)} cases, so some cases go through it more than once. In NB2 that is almost always rework (the invoice is restarted), because only {pct(pr.loc["NB2", "multi_cycle_rate"], 1)} of NB2 cases hold a second invoice.</p>
  <p>Each <b>arrow</b> counts how often one activity was directly followed by another. The arrow from <i>Initial validation</i> to <i>Posted manually</i> ({num(NB2_IV_PM)}) is the road that skips the automatic posting entirely. <i>(Annotation: SAP picks this road during validation, using routing code <code>CTRL1</code>. In NB2, {pct(dn.bot_share, 0)} of these direct postings are made by the nightly batch job {BOT_USERS[1]}, not by a person. See <a href="#routing">Routing at validation</a>.)</i> The loop arrow on <i>Wait for goods receipt</i> ({num(NB2_GR_SELF)}) means the system checked for the goods receipt again because it still hadn't arrived.</p>
  <p>Darker boxes and thicker arrows mean more traffic. The <b>performance map</b> uses the same layout but labels the arrows with the median time between the two steps. That shows where invoices wait, not just where they go.</p>
</div>
<div class="example">
  <div class="lbl">Example · reading a variant</div>
  <p>A variant is shown as a chain of steps, e.g. <span class="code">Data management</span> → <span class="code">Initial validation</span> → <span class="code">Wait for goods receipt</span> → <span class="code">MM Posted Automatically</span>. That is the happy path: {pct(types.loc['ZNB2', 'top1_share'], 0)} of all ZNB2 cases follow exactly this sequence. In ZNB1 the most common variant covers only {pct(types.loc['ZNB1', 'top1_share'], 0)}, and you need {num(types.loc['ZNB1', 'variants_for_80pct'])} variants to cover 80% of the cases. The "Variants for 80%" column in the table above is a quick measure of how standardised each process is.</p>
</div>
<h3 id="routing">Where the road is chosen: routing at validation</h3>
{note(f'''<p>Why do some invoices never wait for the goods receipt? <i>Initial validation</i> records a routing code (<code>SAP_Next_Processing_Type</code>) that fixes the next queue. Across all {num(rcodes.validations.sum())} validations:</p>
<ul class="tight">
  <li><code>WAITGR</code>, {rc('WAITGR')}: goods-receipt check. {pct(rcodes.loc['WAITGR', 'top_next_share'], 1)} are followed by <i>{rcodes.loc['WAITGR', 'top_next_activity']}</i>, the road to automatic posting.</li>
  <li><code>CTRL1</code>, {rc('CTRL1')}: manual control queue. No goods-receipt check. {pct(rcodes.loc['CTRL1', 'top_next_share'], 0)} are followed directly by <i>{rcodes.loc['CTRL1', 'top_next_activity']}</i>, the rest by <i>Transfer to MM</i>, deletion, re-sequencing or approval.</li>
  <li><code>ZWMGRW</code>, {rc('ZWMGRW')}: goods receipt from Warehouse Management first, then the normal wait.</li>
  <li><code>DELE</code>, {rc('DELE')}: deletion.</li>
</ul>
<p>The share sent to the manual queue depends heavily on the PO type. Counted per invoice (first validation of each <code>SAP_PO_No</code>):</p>
{table(["Code", "CTRL1 manual queue", "WAITGR goods-receipt check", "ZWMGRW via WM", "DELE"], routing_rows)}
<p><b>"Posted manually" isn't always manual.</b> On the direct path (<i>Data management → Initial validation → Posted manually</i>, {num(da.cases)} cases), the nightly batch job {BOT_USERS[1]} makes {pct(da.bot_share, 0)} of the postings: {pct(dn.bot_share, 0)} in NB2 and almost none in other types. Bot postings take a median of {da.median_days_bot:.1f} days from validation (they wait for the next night's run). Human postings take {da.median_days_human:.1f} days, with a 90th percentile of {da.p90_days_human:.0f} days. In ZNB6 the human queue takes a median of {direct.loc['ZNB6', 'median_days_human']:.0f} days. This is consistent with the Step 2 classification of {BOT_USERS[1]} as a system user, so the manual-work figures elsewhere in the report already exclude these postings.</p>
<p><b>What the data can't say:</b> which validation rule sends an invoice to <code>CTRL1</code>. That needs the SAP configuration or the validation messages, which aren't in the log.</p>''')}

<h3>Process maps and variants per PO type</h3>
<p class="prose">Open a PO type to see its frequency map, its performance map and its five most common variants. The maps for large types hide activities and transitions that occur in fewer than 0.5% of cases (2% for types with 300–5,000 cases), so that they stay readable.</p>
{''.join(per_type)}

<h2 id="step2"><span class="step">Step 2</span>Case-level KPIs and system users</h2>
<div class="prose">
<p>The three KPIs per case are <b>trace length</b> (number of events), <b>throughput time</b> (days from first to last event) and <b>manual activities count</b> (trace length minus events by system users). To get the manual count, the system users first have to be identified.</p>
</div>
<h3>How system users were told apart from humans</h3>
<div class="prose">
<p>The assignment gives three tell-tale signs of a bot: a very high workload, very short waits between events, and activity only in a fixed night-time batch window. Every user ID was profiled on four kinds of measures:</p>
<ul class="tight">
  <li><b>Workload:</b> total events, cases touched, and events per day on which the user was active. A person working full time handles a few dozen invoice steps a day. A program handles hundreds or thousands.</li>
  <li><b>Speed:</b> median time since the previous event in the same case. A bot reacts within seconds.</li>
  <li><b>Timing:</b> share of events at night (22:00–06:00) and in the 03:00–06:00 batch window.</li>
  <li><b>Range:</b> how many different activities the user performs. Bots tend to do a few fixed steps.</li>
</ul>
<p>The final classification was made by judgment on these profiles, not by a fixed threshold. A threshold rule was used afterwards only as a cross-check (see below).</p>
</div>
{table(["", BOT_USERS[0], BOT_USERS[1], f"Typical busy human (median of {len(humans)} users)"], [
  f"<td>Events</td><td class='n'>{num(b1.events)} ({pct(b1.share_of_all_events, 0)} of all)</td><td class='n'>{num(b2.events)}</td><td class='n'>{num(hum.events)}</td>",
  f"<td>Cases touched</td><td class='n'>{num(b1.cases)} (every case)</td><td class='n'>{num(b2.cases)}</td><td class='n'>{num(hum.cases)}</td>",
  f"<td>Events per active day</td><td class='n'>{num(b1.events_per_active_day)}</td><td class='n'>{num(b2.events_per_active_day)}</td><td class='n'>{num(hum.events_per_active_day)}</td>",
  f"<td>Median wait since previous step</td><td class='n'>{num(b1.median_gap_s)} s</td><td class='n'>{num(b2.median_gap_s / 3600, 1)} h</td><td class='n'>{num(hum.median_gap_s / 3600, 1)} h</td>",
  f"<td>Share in 03:00–06:00 window</td><td class='n'>{pct(b1.batch_0306_share, 0)}</td><td class='n'>{pct(b2.batch_0306_share, 0)}</td><td class='n'>{pct(hum.batch_0306_share, 1)}</td>",
  f"<td>Distinct activities</td><td class='n'>{num(b1.activities)}</td><td class='n'>{num(b2.activities)}</td><td class='n'>{num(hum.activities)}</td>",
])}
<div class="callout">
<p><b>{BOT_USERS[0]} is the system service.</b> It works around the clock, touches every case and handles about {num(b1.events_per_active_day)} steps per day at a median of {num(b1.median_gap_s)} seconds apart, which no person could do. It only performs automated steps: data management, initial validation, waiting for the goods receipt and automatic posting.</p>
<p><b>{BOT_USERS[1]} is the nightly batch job.</b> Its median wait is long because it picks up invoices that have been sitting in a queue, so it fails the speed test. But {pct(b2.batch_0306_share, 0)} of its events fall in the 03:00–06:00 window, it runs at about {num(b2.events_per_active_day)} events per day, and it only does <i>Posted manually</i> and <i>Transfer to MM</i>. SAP labels these postings "manual", but the timing shows a scheduled job is doing them.</p>
<p><b>All other users</b> work daytime office hours at human volumes.</p>
</div>
<h4>The busiest users in detail</h4>
{table(["User", "Events", "Cases", "Activities", "Events / active day", "Median gap (h)", "Night 22–06", "03–06 batch"], brows)}
<div class="two">
{fig("figures/bot_hours.png", "Hourly event profile. User280 works almost only in a 03:00–05:00 batch window. User1587 runs around the clock and touches every case.")}
{fig("figures/bot_scatter.png", "Workload against waiting time per user. Both system users are far outside the human cluster.")}
</div>
<h4>Judgment calls and how much they matter</h4>
<div class="prose">
<ul class="tight">
  <li><b>User1550 is counted as human.</b> It made {num(U1550_EVENTS)} events, almost all <i>Invoice document deleted</i>, on just {num(bots.loc['User1550', 'active_days'])} days. That could be a script, but such a short burst looks more like an administrator running a mass clean-up. Classifying it as a bot would change the manual count in only {num(U1550_CASES)} cases ({pct(U1550_CASES / S['cases'], 1)}).</li>
  <li><b>User280 is counted as a bot,</b> and this decision matters more. If its postings were counted as manual, the total manual workload would grow by {num(U280_BY_TYPE.sum())} events, and NB2's share of all manual work would rise from {pct(pr.loc['NB2', 'manual_share'], 0)} to {pct(nb2_share_280_human, 0)}, because User280 does most of the NB2 postings.</li>
</ul>
<p><b>Rule-based cross-check.</b> A simple rule flags a user as a bot if they have at least 100 events and either more than 150 events per active day or more than half of their events between 03:00 and 06:00. This rule flags <b>{", ".join(rule_users)}</b>. It agrees with the two system users chosen above but also catches User1550, because the deletions were packed into so few days. A rule alone is not enough here. The pattern behind each user has to be looked at, which is why the final classification was done by hand.</p>
</div>
<h3>Worked example: calculating the KPIs for two real invoices</h3>
<div class="example">
  <div class="lbl">Example 1 · {EX_HAPPY} · {code('ZNB2')} happy path</div>
  {example_case(EX_HAPPY)}
  <p><b>Trace length</b> = {int(xh.trace_length)} events. <b>Throughput time</b> = last minus first timestamp = {num(xh.throughput_days, 2)} days. All steps were done by the system user, so <b>manual activities</b> = {int(xh.trace_length)} − {int(xh.bot_events)} = {int(xh.manual_count)}. The case is touchless. Most of its time was spent waiting for the goods receipt to be registered.</p>
</div>
<div class="example">
  <div class="lbl">Example 2 · {EX_REWORK} · {code('ZNB1')} with three invoices</div>
  {example_case(EX_REWORK)}
  <p>At first glance this looks like rework: validation runs three times and the invoice is posted three times. The <code>SAP_PO_No</code> column shows otherwise. These are <b>three different invoices</b> for the same purchase order, arriving on 6, 8 and 11 August. The system validated each one within seconds, then each waited for a person ({EV.loc[(EV.PO_ID == EX_REWORK) & (EV.Activity_ID == "Posted manually"), "User_ID"].iloc[0]}) to post it manually.</p>
  <p><b>Trace length</b> = {int(xr.trace_length)} events. <b>Throughput time</b> = {num(xr.throughput_days, 1)} days from the first to the last event of the whole case. <b>Manual activities</b> = {int(xr.trace_length)} − {int(xr.bot_events)} system events = {int(xr.manual_count)}. The case-level invoice value is {num(xr.invoice_value_nok)} NOK, so a 2% discount would have been worth {num(xr.invoice_value_nok * 0.02)} NOK.</p>
  <p>This case is typical of ZNB1: no real rework, but invoices that sit for one to two weeks before someone posts them. It comes back in Step 3 as an example of how the choice of end point changes whether an SLA counts as met.</p>
</div>
<p class="prose">The table below averages these KPIs over all cases in each PO type.</p>
{table(["Code", "Mean trace length", "Median throughput (days)", "Mean manual activities", "Touchless", "Invoices per case", "Rework within an invoice"], kpi_rows)}

<h2 id="step3"><span class="step">Step 3</span>High-level screening: control and capability</h2>
<div class="prose">
<p>All KPIs are transformed with log(x + 1). Cases are grouped by the week in which they started. PO types with at least 5 cases per week on average get <b>X̄–S charts</b> with limits for each subgroup size. ZNBF and ZRPM get <b>individuals (I-MR) charts</b> on single cases. The Shewhart rules applied are those of the Western Electric handbook:</p>
<ul class="tight">
<li><b>R1</b> one point beyond 3σ</li>
<li><b>R2</b> 2 of 3 consecutive points beyond 2σ on the same side</li>
<li><b>R3</b> 4 of 5 beyond 1σ on the same side</li>
<li><b>R4</b> 8 in a row on one side</li>
<li><b>R5</b> 6 in a row steadily rising or falling</li>
</ul>
<p>A process counts as in control only if no point breaks any rule on either chart.</p>
</div>
<div class="example">
  <div class="lbl">Why log(x + 1)?</div>
  <p>Throughput times range from seconds to months. Example 1 above took {num(xh.throughput_days, 3)} days, so log(1 + {num(xh.throughput_days, 3)}) = {np.log1p(xh.throughput_days):.3f}. Example 2 took {num(xr.throughput_days, 1)} days, giving {np.log1p(xr.throughput_days):.2f}. A 100-day case gives {np.log1p(100):.2f}. The log scale pulls in the long tail so that a handful of extreme cases does not dominate the mean and standard deviation, and the distribution gets closer to the normal shape that control charts assume. The "+1" avoids log(0) for same-second cases and for touchless cases with zero manual activities.</p>
</div>
<div class="example">
  <div class="lbl">Example · reading a control chart</div>
  <p>Open <code>ZNB1</code> below. Each dot on the left X̄ chart is the <b>average log throughput time of all cases that started in that week</b>. The green line is the overall average and the dashed red lines are the ±3σ limits. They get narrower in weeks with more cases, because an average of many cases varies less. The right S chart shows how spread out the cases were within each week.</p>
  <p>Red dots break a rule. In ZNB1, the weeks of 21 and 28 July both lie more than 2σ above the average (R2): invoices that started then were unusually slow. The average then falls six weeks in a row through August (R5). The week of 29 September is above the upper limit (R1). The last two November weeks are below the lower limit (R1) after a run of low weeks (R3). For ZNB1 the late cases are faster, unlike most other PO types. The change in the case population at the end of the period (see Data) is the most likely reason, but the data alone cannot confirm it. A process with no red dots on either chart is "in control": its variation comes only from everyday randomness.</p>
</div>
{table(["Code", "Chart", "Trace length", "Throughput time", "Manual activities"], ctrl_rows)}
<div class="callout">
<p><b>How to read this:</b> with thousands of cases per week, the limits of an X̄ chart become very narrow, so even small shifts between weeks count as special causes. As a robustness check, the weekly means were also charted as individual values. That treats normal week-to-week variation as common cause. Only a few charts pass even that test (marked "weekly-means OK"). The pattern repeats across PO types: stable in July and August, then rising throughput time and manual work from September, partly because of the shrinking, slower set of cases in the last two months. ZRPM passes only because it has 5 points, which proves nothing.</p>
</div>
{cc_details}

<h3>Process capability against the payment SLAs</h3>
<div class="prose"><p>Cpu = (log(USL + 1) − μ) / 3σ, with μ and σ taken from log(throughput days + 1) and σ estimated within subgroups. USL is 10 days (soft SLA, cash discount) and 30 days (hard SLA, legal payment deadline). Cpu ≥ 1 means the process meets the limit with a 3σ margin. The observed breach rates are shown next to it because log throughput is far from normal: most types have a spike at zero (same-day automatic posting) and a long right tail.</p></div>
<div class="example">
  <div class="lbl">Worked example · Cpu for the 10-day SLA (case throughput time)</div>
  <p><b>ZNB1 (Manual order).</b> The mean of log(throughput + 1) is μ = {c1.mean_log_tp:.3f}, which corresponds to a typical (geometric-mean) throughput of e<sup>μ</sup> − 1 = {np.expm1(c1.mean_log_tp):.1f} days. The upper limit on the same scale is log(10 + 1) = {usl10:.3f}.</p>
  <div class="math">Cpu = (USL − μ) / (3σ) = ({usl10:.3f} − {c1.mean_log_tp:.3f}) / (3 × {c1.sigma_within:.3f}) = {c1.Cpu_soft10:.3f}</div>
  <p>The average sits almost exactly on the deadline, so about half of the invoices miss it (observed: {pct(c1.observed_breach_soft10, 0)}).</p>
  <p><b>ZNB2 (Third-party Type 1)</b> for comparison:</p>
  <div class="math">Cpu = ({usl10:.3f} − {c2.mean_log_tp:.3f}) / (3 × {c2.sigma_within:.3f}) = {c2.Cpu_soft10:.2f}</div>
  <p>The average is far below the deadline, so even three standard deviations above it the process is still within 10 days. Only {pct(c2.observed_breach_soft10, 1)} of ZNB2 invoices take longer. Rule of thumb: Cpu ≥ 1.33 is comfortable, 1 is the minimum, and below 1 means the deadline cuts into the bulk of the cases.</p>
</div>
{table(["Code", "Cases", "Median days", "P90 days", "Cpu 10 d", "Breach > 10 d", "Cpu 30 d", "Breach > 30 d"], cap_rows)}
<div class="two">
{fig("figures/capability.png", "Cpu per PO type. Only ZNB2 and ZNB3 meet the 10-day SLA. NB2 also meets the 30-day SLA.")}
{fig("figures/throughput_distributions.png", "Distribution of log(throughput days + 1). Dashed lines mark 10 and 30 days.")}
</div>

<h3 id="sla">When does an invoice meet the SLA?</h3>
<div class="prose">
<p>Both SLAs are about <b>payment</b>. The event log does not contain payment. It ends when the invoice is <b>posted</b> in SAP, which makes it payable. Three activities do this: <i>MM Posted Automatically</i>, <i>Posted manually</i> and <i>MM Autopost after manual process</i>. Payment happens later in a payment run, when the open item is cleared in financial accounting, and that step is not in the data. <code>Case_Clearing_value_NOK</code> gives the cleared amount but no date.</p>
<p>So in this report "SLA met" means <b>posted within 10 (or 30) days</b>. That is a necessary condition for paying on time, but not proof of it. The clock also starts at the first recorded processing step, not at the invoice date, so time spent before scanning is invisible. Both effects make the real SLA performance somewhat worse than measured here. An exact measurement would need the invoice receipt date and the clearing date from the SAP accounting documents.</p>
<p>Where the clock stops makes a big difference, because a case can hold several invoices. Three end points were compared:</p>
</div>
<dl class="gloss">
  <dt>Case, last event</dt><dd>First to last event of the whole case. This is the throughput time used above. It mixes several invoices and any steps after posting.</dd>
  <dt>Case, first posting</dt><dd>First event to the first posting in the case. This is when the first invoice became payable, but later invoices on the same PO are ignored.</dd>
  <dt>Per invoice cycle</dt><dd>For each invoice (<code>SAP_PO_No</code>) separately, from its first event to its first posting. This is the closest match to a payment SLA and the recommended measure.</dd>
  <dt>Clock start for ZNB1</dt><dd>For manual orders ({code('ZNB1')}) the payment term is assumed to start at goods receipt: the clock starts at the invoice's <i>Wait for goods receipt</i> step, read as the moment the goods receipt was booked. Invoices without that step (the manual-queue route) keep the clock from their first event, as their goods are assumed to be received already. Waiting for the goods is a purchasing delay and does not count against the payment term. All other PO types keep the clock from the first event. The same applies to the case-level time to first posting.</dd>
</dl>
<div class="example">
  <div class="lbl">Example · the three end points for {EX_REWORK}</div>
  <p>Case, last event: {num(xr.throughput_days, 1)} days, so the case breaks the 10-day SLA. Case, first posting: {num(xr.time_to_post_days, 1)} days, also a breach. Per invoice: {", ".join(f"{d:.2f}" for d in EX_INV.values)} days, so all three invoices break the 10-day SLA, the last one only just. One case counts as one breach under the first two measures and as three breaches under the third.</p>
</div>
{table(["Code", "> 10 d: case, last event", "> 10 d: case, first posting", "> 10 d: per invoice", "Cpu 10 d: last event", "Cpu 10 d: per invoice", "> 30 d: per invoice", "Cpu 30 d: per invoice"], [
  f"<td>{code(c)}</td><td class='n'>{pct(cap.loc[c, 'observed_breach_soft10'])}</td><td class='n'>{pct(cap_post.loc[c, 'observed_breach_soft10'])}</td>"
  f"<td class='n'><b>{pct(cap_inv.loc[c, 'observed_breach_soft10'])}</b></td>{cpu_cell(cap.loc[c, 'Cpu_soft10'])}{cpu_cell(cap_inv.loc[c, 'Cpu_soft10'])}"
  f"<td class='n'>{pct(cap_inv.loc[c, 'observed_breach_hard30'])}</td>{cpu_cell(cap_inv.loc[c, 'Cpu_hard30'])}"
  for c in order])}
{fig("figures/capability_end_points.png", "Cpu per PO type for the three end points. The per-invoice measure (green) is the closest to the payment SLA.", "wide")}
<div class="callout">
<p><b>What changes and what doesn't.</b> Across all types, {num(cap.cases_breach_soft10.sum())} cases break the 10-day SLA when measured to the last event. Per invoice it is {num(cap_inv.cases_breach_soft10.sum())} of {num(cap_inv.posted.sum())} posted invoices ({pct(cap_inv.cases_breach_soft10.sum() / cap_inv.posted.sum())}).</p>
<p><b>ZNB1 and ZJD1 stay clearly not capable on every measure</b> (per-invoice Cpu {cap_inv.loc['ZNB1', 'Cpu_soft10']:.2f} and {cap_inv.loc['ZJD1', 'Cpu_soft10']:.2f}). The cause is manual posting. In ZNB1, {pct(m1.share, 0)} of invoices are posted by hand after a median of {m1['median']:.1f} days, and {pct(m1.late, 0)} of those take longer than 10 days. Automatically posted ZNB1 invoices are done the same day.</p>
<p><b>ZNB4 looked poor only because of the case view.</b> Its cases often hold several invoices. Per invoice, only {pct(cap_inv.loc['ZNB4', 'observed_breach_soft10'])} miss 10 days and Cpu is {cap_inv.loc['ZNB4', 'Cpu_soft10']:.2f}. NB2, ZNB2 and ZNB3 barely change.</p>
</div>

<h2 id="step4"><span class="step">Step 4</span>Where to focus</h2>
<div class="prose"><p>Need for improvement depends on how bad a process is (control, Cpu) and how much it matters (volume, manual effort, money). The discount column is the 2% cash discount missed on invoices that were posted after more than 10 days.</p>
<p>Each invoice (<code>SAP_PO_No</code>) can be followed from its first event to its posting, so lateness is counted per invoice. Its value is not in the data: <code>Case_Invoice_Value_NOK</code> is the same on every row of a PO and grows with the number of invoices, so it is the total for the PO. The value is therefore split evenly over the PO's posted invoices. For example, {EX_REWORK} from Step 2 was worth {num(xr.invoice_value_nok)} NOK; {ex_late} of its {len(EX_INV)} invoices were posted after more than 10 days, so it misses {num(xr.invoice_value_nok * 0.02 * ex_late / len(EX_INV))} NOK. Invoices that were never posted are left out.</p>
<p>Counting per PO instead, with the whole PO value lost once its last invoice is posted after 10 days, would give {mnok(disc_po_total)} NOK. That overstates the loss, because one late invoice, or invoices that simply arrive weeks apart, would put the whole PO value at loss. The exact amounts need the invoice value per document from SAP.</p>
<p>How to read the columns: <b>share of manual work</b> shows where accountants spend their time (cost). <b>Invoices &gt; 10 d</b> and <b>missed discount</b> show where deadlines are missed (money). <b>Cpu</b> and <b>KPIs in control</b> show how reliable the process is.</p></div>
{table(["Code", "Share of cases", "Share of manual work", "Invoices > 10 d", "Share of all late invoices", "Missed 2% discount (NOK)", "Cpu 10 d (per invoice)", "KPIs in control"], prio_rows)}
<h3>Recommended scope options</h3>
<ol class="rec">
  <li><b>Manual and normal orders ({code('ZNB1')} + {code('ZJD1')}), time to posting.</b> This is the strongest case. Together they have {num(cap_inv.loc[['ZNB1', 'ZJD1'], 'cases_breach_soft10'].sum())} invoices that took longer than 10 days, Cpu far below 1 on every measure, and about {mnok(disc_focus)} NOK of missed discount, because these orders carry the large invoices (average about 1.2M NOK per case). The bottleneck is manual posting: about half of their invoices are posted by hand after a median of {mj['median']:.0f}–{m1['median']:.0f} days. Questions for the project: why don't these invoices post automatically, and where do they wait before a person picks them up?
    {note(f"<p>For many of them, the decision is made at validation: {ctrl1('ZNB1')} of ZNB1 and {ctrl1('ZJD1')} of ZJD1 invoices are routed straight to the manual queue (<code>CTRL1</code>). The sharper project question is <b>which validation rule triggers <code>CTRL1</code></b> for these PO types. Answering it needs the SAP validation configuration or messages, which aren't in the event log.</p>", "Annotation · partly answered")}</li>
  <li><b>Return to vendor ({code('NB2')}), manual activities.</b> A return is a "reverse PO": it follows goods bought under another PO type that are sent back, for example because they arrived damaged, and the supplier may issue a credit note. It accounts for {pct(pr.loc['NB2', 'manual_share'], 0)} of all manual events while meeting both SLAs reasonably well. It also has the most real rework among the large types ({pct(pr.loc['NB2', 'rework_rate'], 0)} of cases). The causes to explain are the direct "Posted manually" path and the GR loop through WM and "Transfer to MM". Automating these is a cost case, not a deadline case.
    {note(f"<p>The direct “Posted manually” path is already {pct(dn.bot_share, 0)} automated by the nightly batch job {BOT_USERS[1]}. The remaining manual cost sits in the {pct(1 - dn.bot_share, 0)} posted by people and in the WM loop ({pct(routing.loc['NB2', 'ZWMGRW'], 0)} of NB2 invoices start via <code>ZWMGRW</code>). Moving more of the direct path into the batch run would cut manual work without new tooling.</p>")}</li>
  <li><b>Automatic orders ({code('ZJD3')}), the same mechanism on a smaller scale.</b> {pct(cap_inv.loc['ZJD3', 'observed_breach_soft10'], 0)} of its invoices take longer than 10 days (Cpu {cap_inv.loc['ZJD3', 'Cpu_soft10']:.2f}), and half are posted manually although the PO type is meant to be automatic. It could be added to option 1 as a comparison group.</li>
</ol>
<p class="prose">ZNB4 was a candidate in an earlier version of this report, but per invoice it meets the 10-day SLA (see <a href="#sla">When does an invoice meet the SLA?</a>). ZNB6, ZNBF and ZRPM have the worst capability, but together they make up less than 0.5% of cases. They are better kept as side notes than chosen as the project scope.</p>

<footer>
<p>Built with Python {'(conda env <code>datasci</code>)'}: pandas, pm4py, matplotlib and scipy. The scripts run in order: <code>01_prepare.py</code> → <code>02_process_maps.py</code> → <code>03_control_capability.py</code> → <code>04_prioritise.py</code> → <code>05_report.py</code>. All tables come from the CSV files in <code>output/</code>.</p>
<p>Amber annotations come from a follow-up analysis of <code>SAP_Next_Processing_Type</code> and posting users (<code>routing.csv</code>, <code>routing_codes.csv</code>, <code>direct_posting.csv</code>).</p>
</footer>
</div>
"""
(OUT / "report.html").write_text(page, encoding="utf-8")
print("wrote", OUT / "report.html", len(page) // 1024, "KB")
