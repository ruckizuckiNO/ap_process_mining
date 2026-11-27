"""Builds output/report_control_charts.html: a detailed guide to the control charts of step 3.

What the charts are for, what is charted (per posted invoice, invoice clock), how to read every panel and rule,
worked examples with the real numbers, the ZNB1 clock split, results for all PO types, and the caveats.
Reads control_chart_results.csv and control_chart_clock_views.csv (run 03_control_capability.py first),
invoices.csv and events.pkl, and reuses the stylesheet of 05_report.py.
"""
import html
import re

import numpy as np
import pandas as pd

from common import (OUT, ROOT, PO_TYPES, CHART_KPIS, GR_CLOCK_PO_TYPES, SLA_SOFT_DAYS, load_cases, load_events,
                    load_invoices)

CSS = re.search(r"<style>(.*?)</style>", (ROOT / "05_report.py").read_text(), re.S).group(1)
CSS = CSS.replace("{{", "{").replace("}}", "}")

ctrl = pd.read_csv(OUT / "control_chart_results.csv")
clockv = pd.read_csv(OUT / "control_chart_clock_views.csv")
cases = load_cases()
ev = load_events()

# Same chart units as 03_control_capability.py
inv_all = load_invoices()
bots = ev.groupby(["PO_ID", "SAP_PO_No"]).is_bot.sum().rename("bot_events")
inv_all = inv_all.join(bots, on=["PO_ID", "SAP_PO_No"])
inv_all["manual_count"] = inv_all.events - inv_all.bot_events
inv_all["clock"] = np.where(inv_all.clock_start > inv_all.start, "GR", "arrival")
inv_all["clock_week"] = inv_all.clock_start.dt.to_period("W-SUN").dt.start_time
inv_all["log_ttp"] = np.log1p(inv_all.time_to_post_days)
# Chart window as used by 03 (all posted invoices except the incomplete last week of the log)
FIRST, LAST, FOLLOWUP_START = (pd.Timestamp(ctrl[k].iloc[0]) for k in ("window_first", "window_last", "followup_start"))
PO_START_FIRST, PO_START_LAST, LOG_END = cases.start.min(), cases.start.max(), ev.ts.max()
posted = inv_all[inv_all.posted]
inv = posted[posted.clock_start <= LAST + pd.Timedelta(days=1)]
n_all, n_unposted, n_after = len(inv_all), int((~inv_all.posted).sum()), len(posted) - len(inv)
n_followup = int((inv.clock_start >= FOLLOWUP_START).sum())


def pct(x, d=1):
    return "–" if pd.isna(x) else f"{x * 100:.{d}f}%"


def num(x, d=0):
    return "–" if pd.isna(x) else f"{x:,.{d}f}"


def code(c):
    return f'<span class="code">{c}</span>'


def table(head, rows, cls=""):
    th = "".join(f"<th>{h}</th>" for h in head)
    body = "".join(f"<tr>{r}</tr>" for r in rows)
    return f'<div class="tbl {cls}"><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def fig(src, cap_text, cls=""):
    return (f'<figure class="{cls}"><div class="plate"><img src="{src}" alt="{html.escape(cap_text)}" loading="lazy"></div>'
            f"<figcaption>{cap_text}</figcaption></figure>")


# ---------------------------------------------------------------- worked examples (ZNB1)
T = "ZNB1"
z = inv[inv.po_type == T]
zr = ctrl[(ctrl.po_type == T) & (ctrl.kpi == "time_to_post_days")].iloc[0]
wk = z.groupby("clock_week").log_ttp.agg(["mean", "std", "size"])
wk = wk[wk["size"] >= 2]
center, sigma = zr.center, zr.sigma_within
wk["se"] = sigma / np.sqrt(wk["size"])
wk["ucl"], wk["lcl"] = center + 3 * wk.se, center - 3 * wk.se
wk["z"] = (wk["mean"] - center) / wk.se
hi = wk.loc[wk.z.idxmax()]
hi_w = wk.z.idxmax()
lo_w = wk.z.idxmin()
lo = wk.loc[lo_w]
first_w = wk.index[0]
fw = wk.loc[first_w]
gr_share = z.groupby(z.clock_start.dt.to_period("M")).apply(lambda d: (d.clock == "GR").mean())


# One PO with a GR-clock and an arrival-clock invoice
EX = "PurDoc184235"
ex = inv_all[inv_all.PO_ID == EX].sort_values("start")
ex_rows = [
    f"<td>{r.SAP_PO_No}</td><td>{r.start:%d %b %Y %H:%M}</td>"
    f"<td>{'–' if pd.isna(r.goods_receipt) else f'{r.goods_receipt:%d %b %Y %H:%M}'}</td>"
    f"<td>{r.clock}</td><td>{r.clock_start:%d %b %Y %H:%M}</td>"
    f"<td>{'–' if pd.isna(r.first_post) else f'{r.first_post:%d %b %Y %H:%M}'}</td>"
    f"<td class='n'>{num(r.time_to_post_days, 2)}</td><td class='n'>{num(r.log_ttp, 3)}</td>"
    f"<td class='n'>{int(r.events)}</td><td class='n'>{int(r.manual_count)}</td>"
    for r in ex.itertuples()]

# ---------------------------------------------------------------- results tables
order = ctrl.drop_duplicates("po_type").sort_values("invoices", ascending=False).po_type.tolist()
kpi_short = {"events": "Events", "time_to_post_days": "Days to posting", "manual_count": "Manual events"}
res_rows = []
for c in order:
    sub = ctrl[ctrl.po_type == c].set_index("kpi")
    cells = []
    for k in CHART_KPIS:
        r = sub.loc[k]
        rules = ", ".join(rr for rr in ["R1", "R2", "R3", "R4", "R5"] if r[rr] > 0) or "none"
        wk_ok = r.weekly_I_in_control in (True, "True")
        pill = ' <span class="pill ok">weekly-means OK</span>' if wk_ok else ""
        cells.append(f"<td class='{'ok' if r.in_control else 'bad'}'>{int(r.points_flagged)}/{int(r.subgroups)}"
                     f"<br><small>{rules}{'; S/MR R1: ' + str(int(r.s_chart_R1)) if r.s_chart_R1 else ''}</small>{pill}</td>")
    ttp = sub.loc["time_to_post_days"]
    res_rows.append(f"<td>{code(c)}<br><small>{PO_TYPES[c]}</small></td><td><small>{sub.chart.iloc[0]}</small></td>"
                    f"<td class='n'>{num(sub.invoices.iloc[0])}</td><td class='n'>{num(ttp.center_days, 2)}</td>{''.join(cells)}")
n_out = int((~ctrl.groupby("po_type").in_control.all()).sum())
wk_ok_n = int((ctrl.weekly_I_in_control.astype(str) == "True").sum())
wk_n = int(ctrl.weekly_I_in_control.notna().sum())

cv_names = ["Headline (invoice clock)", "GR clock only", "Arrival clock only", "Arrival clock for all (upper bound)"]
cv_rows = [f"<td>{n}</td><td class='n'>{num(r.invoices)}</td><td class='n'>{num(r.center_days, 2)}</td>"
           f"<td class='n'>{int(r.points_flagged)}/{int(r.subgroups)}</td>"
           f"<td><small>{', '.join(rr for rr in ['R1', 'R2', 'R3', 'R4', 'R5'] if r[rr] > 0) or 'none'}</small></td>"
           f"<td class='n'>{int(r.weekly_I_violations)}</td>"
           for n, (_, r) in zip(cv_names, clockv.iterrows())]
cvh, cvg, cva, cvu = (clockv.iloc[i] for i in range(4))

galleries = "".join(
    f'<details class="type"><summary>{code(c)} <b>{PO_TYPES[c]}</b><span class="meta">{ctrl[ctrl.po_type == c].chart.iloc[0]} · '
    f'{num(ctrl[ctrl.po_type == c].invoices.iloc[0])} invoices</span></summary>'
    f'<div class="type-body">{fig(f"figures/control_charts/cc_{c}.png", f"{c}: rows = events, days to first posting, manual events per invoice; left = X̄ (or individuals), right = S (or moving range).", "wide")}</div></details>'
    for c in order)

# ---------------------------------------------------------------- schematic of zones and rules (inline SVG)
W, H, PAD = 720, 250, 40
ys = {k: 125 - k * 30 for k in (-3, -2, -1, 0, 1, 2, 3)}


def pts(vals, x0, dx=18):
    return [(x0 + i * dx, 125 - v * 30) for i, v in enumerate(vals)]


def poly(p, col):
    d = " ".join(f"{x:.0f},{y:.0f}" for x, y in p)
    dots = "".join(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="3.5" fill="{col}"/>' for x, y in p)
    return f'<polyline points="{d}" fill="none" stroke="{col}" stroke-width="1.5"/>{dots}'


groups = [("R1", [0.2, -0.4, 0.5, 3.4, 0.1], 70), ("R2", [0.3, 2.3, 1.2, 2.4, 0.4], 190),
          ("R4", [0.4, 0.7, 0.2, 0.9, 0.5, 0.3, 0.8, 0.6], 310), ("R5", [-1.4, -0.9, -0.5, 0.1, 0.6, 1.1], 490)]
svg_parts = []
for k, lab in ((3, "UCL +3σ"), (2, "+2σ"), (1, "+1σ"), (0, "centre line"), (-1, "−1σ"), (-2, "−2σ"), (-3, "LCL −3σ")):
    dash = "" if k in (0,) else ' stroke-dasharray="5 4"'
    col = "#1f6f5c" if k == 0 else ("#b0333f" if abs(k) == 3 else "#9aa8a2")
    svg_parts.append(f'<line x1="{PAD}" x2="{W - 110}" y1="{ys[k]}" y2="{ys[k]}" stroke="{col}" stroke-width="1"{dash}/>'
                     f'<text x="{W - 104}" y="{ys[k] + 4}" font-size="11" fill="#5a6a63">{lab}</text>')
for k, (y0, y1) in (("C", (ys[1], ys[-1])), ("B", (ys[2], ys[1])), ("A", (ys[3], ys[2]))):
    svg_parts.append(f'<text x="{PAD - 18}" y="{(y0 + y1) / 2 + 4:.0f}" font-size="11" fill="#5a6a63">{k}</text>')
for name, vals, x0 in groups:
    p = pts(vals, x0)
    svg_parts.append(poly(p, "#b0333f" if name else "#4a6fa5"))
    svg_parts.append(f'<text x="{x0 + (len(vals) - 1) * 9:.0f}" y="{H - 8}" font-size="12" font-weight="600" text-anchor="middle" fill="#16201c">{name}</text>')
svg = (f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="Schematic control chart with σ zones and four '
       f'rule patterns: R1 one point beyond 3σ, R2 two of three beyond 2σ, R4 eight in a row above the centre, R5 six rising">'
       + "".join(svg_parts) + "</svg>")

# ---------------------------------------------------------------- page
page = f"""<meta charset="utf-8">
<title>Control Charts Explained</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,500;6..72,600&family=Instrument+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap">
<style>{CSS}
svg text {{ font-family: var(--f-body); }}
</style>

<div class="wrap">
<header>
  <div class="eyebrow">Accounts payable · Process mining case · Step 3 guide</div>
  <h1>Control Charts Explained</h1>
  <p class="lede">What the control charts in this project are for, what exactly they measure under the invoice clock, how to read every panel and rule, and what they can and cannot tell us.</p>
  <div class="facts">
    <span><b>{num(len(inv))}</b> posted invoices charted</span>
    <span><b>11</b> PO types</span>
    <span><b>3</b> KPIs per invoice</span>
    <span>clock starts <b>{FIRST:%d %b %Y} – {LAST:%d %b %Y}</b></span>
    <span><b>{n_out} of 11</b> PO types out of control</span>
  </div>
</header>

<ul class="toc">
  <li><a href="#why">1 · What they are for</a></li>
  <li><a href="#what">2 · What is charted</a></li>
  <li><a href="#read">3 · Reading a figure</a></li>
  <li><a href="#rules">4 · The rules</a></li>
  <li><a href="#example">5 · Worked example</a></li>
  <li><a href="#clock">6 · ZNB1 by clock</a></li>
  <li><a href="#results">7 · Results</a></li>
  <li><a href="#caveats">8 · Caveats</a></li>
  <li><a href="#changed">9 · What changed</a></li>
</ul>

<h2 id="why"><span class="step">1</span>What control charts are for</h2>
<div class="prose">
<p>Every process varies. Some invoices post in minutes, others take weeks. A control chart separates two kinds of variation:</p>
<dl class="gloss">
  <dt>Common cause</dt><dd>The everyday noise of a stable process: different suppliers, different clerks, a busy Monday. It is part of how the process is built and only changes if the process is redesigned.</dd>
  <dt>Special cause</dt><dd>Something that changed: a backlog, a new rule in SAP, a holiday, a shift in which invoices arrive. It shows up as points outside the limits or as patterns that randomness rarely produces.</dd>
</dl>
<p>A process whose chart shows only common-cause variation is <b>in statistical control</b>: it is predictable, and its past behaviour is a fair forecast of the next weeks. In this project the charts have three jobs:</p>
<ol>
  <li><b>Screening.</b> Which PO types behave predictably, and which drift? This is part of the high-level screening in Step 3 of the report.</li>
  <li><b>A precondition for capability.</b> Cpu (can the process meet the 10- and 30-day SLAs?) assumes a stable process. If a process is out of control, Cpu describes an average over changing conditions, not a stable property.</li>
  <li><b>Pointing at when things changed.</b> Flagged weeks tell us where to look for causes (a backlog, a change in routing, the composition of the invoice population).</li>
</ol>
</div>
<div class="callout"><p><b>In control does not mean good.</b> A process can be perfectly stable and still miss the SLA on half of its invoices. Control is about predictability; capability (Cpu, breach rates) is about meeting the deadline. The report shows both for this reason.</p></div>

<h2 id="what"><span class="step">2</span>What is charted: invoices on the invoice clock</h2>
<div class="prose">
<p>The charts use the same unit and the same clock as the SLA and missed-discount analysis, so the control verdicts and the money figures describe the same thing.</p>
</div>
<dl class="gloss">
  <dt>Unit</dt><dd>One <b>posted invoice</b>: a PO_ID × <code>SAP_PO_No</code> cycle. A PO can hold several invoices; each gets its own clock. Invoices that were never posted ({num(n_unposted)} of {num(n_all)}) have no end point and are left out.</dd>
  <dt>Clock start</dt><dd>The invoice's first event (arrival in accounts payable). For {', '.join(code(c) for c in GR_CLOCK_PO_TYPES)} only, the clock starts at <b>goods receipt</b>: the first <i>Wait for goods receipt</i> event with object code <code>WAIT_GR</code>, read as the moment the goods receipt was booked. <code>END_WAIT_GR</code> (wait ended without a goods receipt) does not count, and ZNB1 invoices without a <code>WAIT_GR</code> step keep the arrival clock. The log has no goods-receipt date, so this is a proxy.</dd>
  <dt>Clock stop</dt><dd>The <b>first posting</b> (<i>MM Posted Automatically</i>, <i>Posted manually</i> or <i>MM Autopost after manual process</i>): the moment the invoice became payable. The log ends at posting, not payment.</dd>
  <dt>KPIs</dt><dd><b>Events per invoice</b> (all steps of that invoice cycle), <b>days to first posting</b> (clock start → first posting) and <b>manual events per invoice</b> (events not done by the two system users). Only the time KPI depends on the clock; the two counts describe the same invoices so that all three rows of a figure share one population.</dd>
  <dt>Scale</dt><dd>Every KPI is transformed with log(x + 1). Times range from seconds to months; the log pulls in the long tail so a few extreme invoices don't dominate the mean and the standard deviation. The +1 keeps zero values defined. To read a value back in days: days = e<sup>value</sup> − 1. The centre line then is a <b>geometric mean</b>, which sits below the ordinary mean for skewed data.</dd>
  <dt>Subgroups</dt><dd>Invoices are grouped by the <b>week in which their clock started</b> (Monday to Sunday). For ZNB1 that is the goods-receipt week where there is one: an invoice belongs to the week its payment term began.</dd>
  <dt>Period</dt><dd>The event log runs from {PO_START_FIRST:%d %b %Y} to {LOG_END:%d %b %Y}, and the charts use all of it: clock starts from {FIRST:%d %b %Y} to {LAST:%d %b %Y}. Only the incomplete last week (the log ends on a {LOG_END:%A}) is left out, {num(n_after)} invoices. One thing changes along the way: POs only <i>start</i> between {PO_START_FIRST:%d %b} and {PO_START_LAST:%d %b %Y}. From {FOLLOWUP_START:%d %b %Y} on, every clock belongs to a follow-up invoice of an older PO, or (for ZNB1) to a goods receipt booked after November for an older invoice. These {num(n_followup)} invoices are kept, so the charts match the SLA and missed-discount figures, but that period is <b>shaded</b> in the charts because the population is different.</dd>
</dl>

<div class="example">
  <div class="lbl">Example · the two clocks on one PO ({EX}, ZNB1)</div>
  <p>This PO has an invoice that waited for goods receipt and one that didn't. The first gets the goods-receipt clock, the second the arrival clock. Each becomes one point in the chart data, in the week its clock started.</p>
  {table(["Invoice", "First event", "WAIT_GR", "Clock", "Clock start", "First posting", "Days", "log(days+1)", "Events", "Manual"], ex_rows)}
</div>

<h2 id="read"><span class="step">3</span>Reading a figure</h2>
<div class="prose">
<p>Each PO type has one figure (<code>output/figures/control_charts/cc_&lt;type&gt;.png</code>) with three rows, one per KPI: events, days to first posting, manual events. Each row has two panels.</p>
</div>
{fig(f"figures/control_charts/cc_{T}.png", "ZNB1, the figure used in the worked example below. Rows: events, days to first posting, manual events per invoice. Left: X̄ chart of weekly means. Right: S chart of the spread within each week.", "wide")}
<h3>Left panel: the X̄ chart (where is the process?)</h3>
<div class="prose">
<ul class="tight">
  <li><b>Blue dots</b>: the mean of log(KPI + 1) over all invoices whose clock started in that week.</li>
  <li><b>Green line</b>: the centre line, the overall mean weighted by week size (X̿).</li>
  <li><b>Dashed red steps</b>: the control limits at ±3 standard errors. They are wide in weeks with few invoices and narrow in busy weeks, because an average over many invoices varies less. That is why the limits funnel in during July, when only a handful of invoices had started, and widen again in the shaded follow-up period.</li>
  <li><b>Shaded area</b>: clock starts from {FOLLOWUP_START:%d %b %Y}, when only follow-up invoices of older POs remain (no new POs start).</li>
  <li><b>Red dots</b>: weeks that break at least one rule (section 4).</li>
</ul>
</div>
<div class="math">σ   = pooled within-week standard deviation = √( Σ (nᵢ − 1)·sᵢ² / Σ (nᵢ − 1) )
UCLᵢ = X̿ + 3·σ/√nᵢ        LCLᵢ = X̿ − 3·σ/√nᵢ        zᵢ = (x̄ᵢ − X̿) / (σ/√nᵢ)</div>
<h3>Right panel: the S chart (how consistent is the process?)</h3>
<div class="prose">
<p>Each dot is the standard deviation of the invoices <i>within</i> one week. A high point means that week mixed very fast and very slow invoices; a low point means the invoices were unusually alike. The centre line is c₄(n)·σ and the limits are c₄(n)·σ ± 3·σ·√(1 − c₄(n)²), where c₄ is the usual small-sample correction. Only rule R1 is applied to the S chart.</p>
</div>
<h3>Small PO types: individuals and moving-range charts</h3>
<div class="prose">
<p>PO types with fewer than 5 invoices per week on average (ZNBF and ZRPM) get an <b>I-MR chart</b> instead: every invoice is its own point, in order of clock start (the x-axis is the invoice number, not a date). σ is estimated from the average moving range between neighbouring invoices (MR̄ / 1.128). The right panel is the moving-range chart, with an upper limit of 3.267 · MR̄. With 9 invoices (ZRPM) such a chart proves very little either way.</p>
</div>
<h3>Robustness check: weekly means as individuals</h3>
<div class="prose">
<p>With hundreds or thousands of invoices a week, σ/√n becomes tiny and the X̄ limits very narrow, so even a small, harmless shift between weeks breaks them. The pipeline therefore also charts the weekly means as individual values (an I-chart on the weekly means, σ from their moving range). That treats normal week-to-week variation as common cause and only flags shifts that stand out from it. It is not drawn, but its verdict is in the results table ("weekly-means OK"). If a KPI fails even this check, the change is large, not just statistically detectable.</p>
</div>

<h2 id="rules"><span class="step">4</span>The rules (Western Electric)</h2>
<div class="prose">
<p>The rules are applied to the standardised values z. The band between the centre line and ±1σ is zone C, ±1–2σ zone B, ±2–3σ zone A. Under a stable, roughly normal process each rule fires by chance only rarely (well under 1% per point), so a firing rule is evidence of a special cause.</p>
</div>
<figure><div class="plate">{svg}</div><figcaption>Schematic: the σ zones and four of the patterns the rules look for. Red lines are pattern examples, not data.</figcaption></figure>
{table(["Rule", "Pattern", "What it usually means here"], [
  "<td><b>R1</b></td><td>One point beyond ±3σ</td><td>A single unusual week: a backlog, a holiday week, a batch of odd invoices.</td>",
  "<td><b>R2</b></td><td>2 of 3 consecutive points beyond 2σ, same side</td><td>A short-lived shift, often two or three slow weeks in a row.</td>",
  "<td><b>R3</b></td><td>4 of 5 consecutive points beyond 1σ, same side</td><td>A moderate shift that lasts about a month.</td>",
  "<td><b>R4</b></td><td>8 points in a row on one side of the centre line</td><td>A sustained level change: the process (or the invoice mix) is now faster or slower than the period average.</td>",
  "<td><b>R5</b></td><td>6 points in a row steadily rising or falling</td><td>A trend: a backlog building up or being worked off.</td>",
])}
<div class="prose"><p>A KPI counts as <b>in control</b> only if no point breaks any rule on the left chart and no point breaks R1 on the right chart. A PO type counts as in control only if all three KPIs are.</p></div>

<h2 id="example"><span class="step">5</span>Worked example: ZNB1, days to first posting</h2>
<div class="example">
  <div class="lbl">Step by step · middle row of the ZNB1 figure</div>
  <p><b>Centre line.</b> Over {num(zr.invoices)} posted ZNB1 invoices the weighted mean of log(days + 1) is X̿ = {center:.3f}, a geometric mean of e<sup>{center:.3f}</sup> − 1 = {zr.center_days:.2f} days.</p>
  <p><b>Spread.</b> The pooled within-week standard deviation is σ = {sigma:.3f} on the log scale.</p>
  <p><b>The slowest week.</b> In the week of {hi_w:%d %b} {int(hi['size'])} invoices started their clock. Their mean was {hi['mean']:.3f} ({np.expm1(hi['mean']):.1f} days).</p>
  <div class="math">σ/√n = {sigma:.3f}/√{int(hi['size'])} = {hi.se:.3f}
UCL   = {center:.3f} + 3 × {hi.se:.3f} = {hi.ucl:.3f}
z     = ({hi['mean']:.3f} − {center:.3f}) / {hi.se:.3f} = {hi.z:.1f}   → beyond 3σ, R1</div>
  <p><b>A small week.</b> In the first week (labelled {first_w:%d %b}, the Monday of the week in which the log starts on {PO_START_FIRST:%d %b}) only {int(fw['size'])} invoices started, so the limits are wide: UCL = {fw.ucl:.3f}, LCL = {fw.lcl:.3f}. Its mean of {fw['mean']:.3f} is inside them (z = {fw.z:.1f}).</p>
  <p><b>The lowest week.</b> {lo_w:%d %b}: {int(lo['size'])} invoices, mean {lo['mean']:.3f} ({np.expm1(lo['mean']):.1f} days), z = {lo.z:.1f}.</p>
  <p><b>The pattern.</b> Four phases are visible. From late July to late September the weekly means lie above the centre line, many beyond the upper limit and in runs of eight or more (R4). In October and November they lie below it. In the shaded follow-up period (December to early February) they rise above it again, around Christmas and in late January. The last two weeks fall far below, but that is mostly an artefact: an invoice whose clock started in late February had only days to be posted before the log ends, so only the fast ones are in the data. The process is not stable over the period.</p>
  <p><b>Back in days.</b> The UCL of {hi.ucl:.3f} in that week means e<sup>{hi.ucl:.3f}</sup> − 1 = {np.expm1(hi.ucl):.2f} days as a weekly geometric mean. The limits concern weekly averages, not single invoices; they are not an SLA.</p>
</div>

<h2 id="clock"><span class="step">6</span>ZNB1 split by clock</h2>
<div class="prose">
<p>The ZNB1 headline chart mixes two groups. Invoices with a <code>WAIT_GR</code> step (mostly on the goods-receipt road, <code>WAITGR</code>) run on the goods-receipt clock and post quickly after it. Invoices without one (mostly on the manual control road, <code>CTRL1</code>) keep the arrival clock and wait in the manual queue. The share of invoices on the goods-receipt clock grows over the period:</p>
</div>
{table(["Clock-start month", *[p.strftime("%b %Y") for p in gr_share.index]], ["<td>Share on GR clock</td>" + "".join(f"<td class='n'>{pct(v, 0)}</td>" for v in gr_share.values)])}
<div class="prose"><p>So part of the summer-to-autumn fall in the headline chart is a change of mix: more fast GR-clock invoices, not necessarily faster work. The extra figure separates the two groups:</p></div>
{fig(f"figures/control_charts/cc_{T}_by_clock.png", "ZNB1 days to first posting, X̄ charts with weekly subgroups by clock start. Top left: headline invoice clock. Top right: only GR-clock invoices. Bottom left: only arrival-clock invoices. Bottom right: arrival clock for every invoice (the upper-bound view). Same y-axis in all four panels.", "wide")}
{table(["View", "Invoices", "Centre (days, geo. mean)", "Weeks flagged", "Rules", "Weekly-means flags"], cv_rows)}
<div class="prose">
<p>How to read it:</p>
<ul class="tight">
  <li><b>GR clock only</b>: a geometric mean of {cvg.center_days:.2f} days. Once the goods are in, these invoices post almost immediately. The chart still breaks rules ({int(cvg.points_flagged)} of {int(cvg.subgroups)} weeks), but on a low level: almost zero in July and August, slowly rising through the autumn, and clearly slower in the follow-up period from December to late January.</li>
  <li><b>Arrival clock only</b>: {cva.center_days:.1f} days. High from July to September, lower in October and November, back near the centre line in the follow-up period, then the end-of-log drop. This is the manual queue, and it does show a real change over time, not only a mix effect.</li>
  <li><b>Arrival clock for all</b>: {cvu.center_days:.1f} days. This is the old view and the upper bound for the money at risk. It shows the same summer-to-autumn fall and the end-of-log drop.</li>
  <li>Do not read the gap between the two groups as the effect of the route alone: they differ in clock <i>and</i> in route. A CTRL1 invoice may also have waited for goods that the log can't see (see <code>docs/assumptions.html</code> §3).</li>
</ul>
</div>

<h2 id="results"><span class="step">7</span>Results for all PO types</h2>
<div class="prose"><p>Each cell shows the number of flagged points out of all points, the rules that fired on the X̄ (or individuals) chart, and how many points broke R1 on the S (or moving-range) chart. Green = in control. "weekly-means OK" = passes the robustness check. The centre column is the geometric-mean days to first posting on the invoice clock.</p></div>
{table(["PO type", "Chart", "Invoices", "Centre (days)", "Events per invoice", "Days to first posting", "Manual events per invoice"], res_rows)}
<div class="callout"><p><b>Verdict:</b> {n_out} of 11 PO types are out of control on at least one KPI. Only {wk_ok_n} of {wk_n} weekly-means checks pass, so most of the signals are large shifts, not just narrow limits. ZRPM ({num(ctrl[ctrl.po_type == 'ZRPM'].invoices.iloc[0])} invoices) is too small to judge either way.</p></div>
{galleries}

<h2 id="caveats"><span class="step">8</span>Caveats: what the charts can't tell us</h2>
<div class="prose">
<ul>
  <li><b>The invoice population is not constant.</b> Only POs that started between July and November are in the log. In early July few invoices exist yet, from October the volume falls (the report's data section suspects an extraction effect), and from December only follow-up invoices of older POs remain (shaded). A changing population alone can move the weekly means, so a signal is a reason to investigate, not proof that the work changed.</li>
  <li><b>Less time to be posted at the end.</b> The log ends on {LOG_END:%d %b %Y}, and invoices that were not posted by then are excluded. An invoice whose clock started in February had only a few weeks, or days, to be posted, so slow late invoices are missing and the last weeks look faster than they were. Don't read the drop at the right edge of the charts as an improvement.</li>
  <li><b>Narrow limits.</b> With hundreds of invoices a week, statistically significant is not the same as practically important. Use the weekly-means check and the size of the shift in days.</li>
  <li><b>Not normal, even on the log scale.</b> Many PO types have a spike at same-minute automatic postings and a long tail of manual ones. The X̄ chart is fairly robust to this because it charts averages, but the S chart and the exact false-alarm rates are less reliable.</li>
  <li><b>The goods-receipt clock is a proxy.</b> The log has no goods-receipt date; the first <code>WAIT_GR</code> event stands in for it, and it exists almost only on the WAITGR road. Always read the ZNB1 headline next to the arrival-clock view.</li>
  <li><b>Timestamps mark completion.</b> Each event has one timestamp, the end of a step, so waiting and working time can't be separated inside a step.</li>
  <li><b>Control is not capability.</b> Whether the 10-day and 30-day SLAs are met is answered by Cpu and the observed breach rates in the report, not by these charts.</li>
</ul>
</div>

<h2 id="changed"><span class="step">9</span>What changed from the first version</h2>
{table(["", "First version", "Now"], [
  "<td>Unit</td><td>One PO (case), all its invoices together</td><td>One posted invoice (PO × <code>SAP_PO_No</code>)</td>",
  "<td>Time KPI</td><td>Throughput: first to last event of the whole PO, including rework after posting</td><td>Days from clock start to first posting</td>",
  "<td>Clock start</td><td>First event of the PO for every type</td><td>First event of the invoice; for ZNB1 goods receipt (first <code>WAIT_GR</code>) where there is one</td>",
  "<td>Count KPIs</td><td>Trace length and manual activities per PO</td><td>Events and manual events per invoice</td>",
  "<td>Subgroup</td><td>Week the PO started</td><td>Week the invoice clock started, whole log (follow-up period shaded)</td>",
  "<td>Extra</td><td>–</td><td>ZNB1 split by clock (<code>cc_ZNB1_by_clock.png</code>)</td>",
])}
<div class="prose"><p>The capability figures (Cpu) in the report are unchanged; the per-invoice Cpu there already used the invoice clock.</p></div>

<footer>
<p>Built by <code>16_report_control_charts.py</code> from <code>control_chart_results.csv</code>, <code>control_chart_clock_views.csv</code>, <code>invoices.csv</code> and <code>events.pkl</code>. Run <code>01_prepare.py</code> and <code>03_control_capability.py</code> first. The charts themselves are drawn by <code>03_control_capability.py</code> into <code>output/figures/control_charts/</code>.</p>
</footer>
</div>
"""
(OUT / "report_control_charts.html").write_text(page, encoding="utf-8")
print("wrote", OUT / "report_control_charts.html", len(page) // 1024, "KB")
