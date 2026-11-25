"""Builds output/report_zjd1_znb1_clock.html: ZJD1 vs ZNB1 per invoice, with the SLA clock from arrival and from goods receipt.

A side comparison: the pipeline (GR_CLOCK_PO_TYPES) still starts the goods-receipt clock for ZNB1 only. Here the same rule
is applied to both PO types so they can be compared like for like. Reads output/invoices.csv and cases.csv; reuses the
stylesheet of 05_report.py. report.html and report_znb1.html are not touched.
"""
import re

import pandas as pd

from clock_doc import logic_html, feasibility
from common import (OUT, ROOT, PO_TYPES, POSTING_ACTIVITIES, SLA_SOFT_DAYS, SLA_HARD_DAYS, GR_ACTIVITY, GR_OBJECT_CODE,
                    GR_CLOCK_PO_TYPES, load_invoices, load_cases)

TYPES = ["ZJD1", "ZNB1"]
CSS = re.search(r"<style>(.*?)</style>", (ROOT / "05_report.py").read_text(), re.S).group(1)
CSS = CSS.replace("{{", "{").replace("}}", "}")


def pct(x, d=1):
    return "–" if pd.isna(x) else f"{x * 100:.{d}f}%"


def num(x, d=0):
    return "–" if pd.isna(x) else f"{x:,.{d}f}"


def mnok(x):
    return f"{x / 1e6:,.1f}M"


def code(c):
    return f'<span class="code">{c}</span>'


def table(head, rows, cls=""):
    th = "".join(f"<th>{h}</th>" for h in head)
    body = "".join(f"<tr>{r}</tr>" for r in rows)
    return f'<div class="tbl {cls}"><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def bar(x, top):
    return f'<div class="bar"><span style="width:{x / top * 100:.1f}%"></span></div>'


# ---------------------------------------------------------------- data
inv = load_invoices()
inv = inv[inv.po_type.isin(TYPES)].copy()
cases = load_cases().set_index("PO_ID")
not_posted = inv[~inv.posted].groupby("po_type").size()
p = inv[inv.posted].copy()
# Invoice value: the PO value split evenly over the PO's posted invoices (same rule as 04_prioritise.py)
p["val"] = cases.invoice_value_nok.reindex(p.PO_ID).values / p.groupby("PO_ID").PO_ID.transform("size").values
p["has_gr"] = p.goods_receipt.notna()
p["arr"] = (p.first_post - p.start).dt.total_seconds() / 86400
p["gr"] = (p.first_post - p.goods_receipt.fillna(p.start)).dt.total_seconds() / 86400
assert (p.gr >= 0).all()
CLOCKS = {"arr": "From arrival (first event)", "gr": "From goods receipt"}


def stats(g, col):
    x = g[col]
    l10, l30 = x > SLA_SOFT_DAYS, x > SLA_HARD_DAYS
    return dict(n=len(g), pos=g.PO_ID.nunique(), median=x.median(), n10=int(l10.sum()), s10=l10.mean(),
                n30=int(l30.sum()), s30=l30.mean(), missed=0.02 * g.val[l10].sum(), value=g.val.sum())


R = {(t, c): stats(p[p.po_type == t], c) for t in TYPES for c in CLOCKS}
SPLIT = {(t, h): stats(p[(p.po_type == t) & (p.has_gr == h)], "gr") for t in TYPES for h in (True, False)}
GRSHARE = p.groupby("po_type").has_gr.mean()
# Sanity check: ZNB1 with the GR clock must reproduce the pipeline's headline (time_to_post_days uses GR for ZNB1)
_z = p[p.po_type == "ZNB1"]
assert abs(R[("ZNB1", "gr")]["missed"] - 0.02 * _z.val[_z.time_to_post_days > SLA_SOFT_DAYS].sum()) < 1
a1, a2, g1, g2 = R[("ZJD1", "arr")], R[("ZNB1", "arr")], R[("ZJD1", "gr")], R[("ZNB1", "gr")]
s1t, s1f, s2t, s2f = SPLIT[("ZJD1", True)], SPLIT[("ZJD1", False)], SPLIT[("ZNB1", True)], SPLIT[("ZNB1", False)]

# Main comparison table: one row per clock and PO type
top10 = max(r["s10"] for r in R.values())
main_rows = []
for c, lab in CLOCKS.items():
    for t in TYPES:
        r = R[(t, c)]
        hl = " class='hl'" if c == "gr" else ""
        main_rows.append(f"<td{hl}>{lab if t == TYPES[0] else ''}</td><td>{code(t)}</td><td class='n'>{num(r['n'])}</td>"
                         f"<td class='n'>{num(r['median'], 2)}</td><td class='n'><b>{pct(r['s10'])}</b></td><td class='n'>{num(r['n10'])}</td>"
                         f"<td>{bar(r['s10'], top10)}</td><td class='n'><b>{pct(r['s30'])}</b></td><td class='n'>{num(r['n30'])}</td>"
                         f"<td class='n'>{mnok(r['missed'])}</td>")

split_rows = []
for h, lab in [(True, "With a goods-receipt step"), (False, "Without (clock from first event)")]:
    for t in TYPES:
        r = SPLIT[(t, h)]
        tot = R[(t, "gr")]
        split_rows.append(f"<td>{lab if t == TYPES[0] else ''}</td><td>{code(t)}</td><td class='n'>{num(r['n'])}</td>"
                          f"<td class='n'>{pct(r['n'] / tot['n'], 0)}</td><td class='n'>{num(r['median'], 2)}</td>"
                          f"<td class='n'><b>{pct(r['s10'])}</b></td><td class='n'>{pct(r['s30'])}</td>"
                          f"<td class='n'>{mnok(r['missed'])}</td><td class='n'>{pct(r['missed'] / tot['missed'], 0)}</td>")

# Clock timeline (start and stop points along an invoice)
STEPS = [("Invoice arrives", "Data management", "start-a"), ("Initial validation", "routing code set", ""),
         ("Wait for goods receipt", f"object code {GR_OBJECT_CODE}", "start-g"), ("First posting", "invoice payable", "stop"),
         ("Payment run", "not in the log", "out")]
timeline = '<ol class="clock">' + "".join(
    f'<li class="{cls}"><b>{a}</b><small>{b}</small>'
    + ('<span class="tag a">start · arrival clock</span>' if cls == "start-a" else "")
    + ('<span class="tag g">start · goods-receipt clock</span>' if cls == "start-g" else "")
    + ('<span class="tag s">stop · both clocks</span>' if cls == "stop" else "")
    + ('<span class="tag o">real SLA end, not measurable</span>' if cls == "out" else "")
    + "</li>" for a, b, cls in STEPS) + "</ol>"

LOGIC = logic_html(two_clocks=True)
FEAS, FS = feasibility(TYPES)

EXTRA_CSS = """
.bar { background: var(--accent-soft); border-radius: 3px; height: 10px; min-width: 90px; }
.bar span { display: block; height: 100%; background: var(--accent); border-radius: 3px; }
td.hl { font-weight: 600; }
ol.clock { list-style: none; padding: 0; margin: 18px 0; display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 170px), 1fr)); gap: 10px; }
ol.clock li { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 12px 14px; position: relative; min-width: 0; }
ol.clock li b { display: block; font-size: 14px; }
ol.clock li small { display: block; color: var(--muted); font: 12px var(--f-mono); margin-top: 2px; }
ol.clock li.out { border-style: dashed; color: var(--muted); }
ol.clock .tag { display: inline-block; margin-top: 8px; font: 500 10.5px/1.2 var(--f-mono); padding: 3px 6px; border-radius: 4px; }
.tag.a { background: var(--warn-bg); color: var(--warn); }
.tag.g { background: var(--ok-bg); color: var(--ok); }
.tag.s { background: var(--accent-soft); color: var(--accent); }
.tag.o { border: 1px dashed var(--line); color: var(--muted); }
"""

page = f"""<meta charset="utf-8">
<title>ZJD1 vs ZNB1 Lead Time</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,500;6..72,600&family=Instrument+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap">
<style>{CSS}{EXTRA_CSS}</style>

<div class="wrap">
<header>
  <div class="eyebrow">Accounts payable · Process mining case · Side comparison</div>
  <h1>ZJD1 vs ZNB1: invoices beyond 10 and 30 days</h1>
  <p class="lede">Normal orders ({code('ZJD1')}) and manual orders ({code('ZNB1')}), counted per invoice, with the SLA clock started at arrival and at goods receipt.</p>
  <div class="facts">
    <span><b>{num(a1['pos'])}</b> ZJD1 POs · <b>{num(a1['n'])}</b> invoices</span>
    <span><b>{num(a2['pos'])}</b> ZNB1 POs · <b>{num(a2['n'])}</b> invoices</span>
    <span>SLA <b>{SLA_SOFT_DAYS} d</b> (2% discount) and <b>{SLA_HARD_DAYS} d</b> (payment deadline)</span>
  </div>
</header>

<h2 id="answer">Short answer</h2>
<div class="findings">
  <div class="finding"><div class="k">{pct(g1['s10'])} vs {pct(g2['s10'])}</div><p>of invoices take longer than 10 days in ZJD1 and ZNB1 when both clocks start at goods receipt. ZNB1 is late about twice as often.</p></div>
  <div class="finding"><div class="k">{pct(g1['s30'])} vs {pct(g2['s30'])}</div><p>take longer than 30 days. On the legal payment deadline the two PO types are about equal.</p></div>
  <div class="finding"><div class="k">{mnok(g1['missed'])} vs {mnok(g2['missed'])}</div><p>NOK of 2% cash discount is missed. ZNB1 loses more although its POs are worth slightly less on average.</p></div>
  <div class="finding"><div class="k">{pct(s1t['s10'])} = {pct(s2t['s10'])}</div><p>With a goods-receipt step both are equally fast. The whole difference sits in the invoices without one, mostly the manual-queue route: {pct(s1f['s10'])} late in ZJD1, {pct(s2f['s10'])} in ZNB1.</p></div>
</div>

<h2 id="clock"><span class="step">Assumption</span>Where the clock starts and stops</h2>
<div class="prose">
<p>Both SLAs are about <b>payment</b>, but the event log doesn't contain payment or the invoice date. Lead time is therefore measured <b>per invoice</b> (<code>SAP_PO_No</code>) between two events that are in the log.</p>
</div>
{timeline}
{LOGIC}
<dl class="gloss">
  <dt>Pipeline vs this page</dt><dd>The pipeline and the other reports use clock B only for {", ".join(GR_CLOCK_PO_TYPES)} (decided 2026-11-19), and clock A for ZJD1. Comparing those two numbers mixes two clocks. This page applies the same clock to both types.</dd>
  <dt>Never posted</dt><dd>Invoices without a posting are left out: {num(not_posted.get('ZJD1', 0))} in ZJD1, {num(not_posted.get('ZNB1', 0))} in ZNB1.</dd>
</dl>

<h2 id="compare"><span class="step">Result</span>Per-invoice comparison</h2>
{table(["Clock start", "Type", "Invoices", "Median days", "&gt; 10 d", "Count", "", "&gt; 30 d", "Count", "Missed 2% (NOK)"], main_rows)}
<div class="callout">
<p><b>The clock decides the ranking.</b> From arrival, ZNB1 looks worse ({pct(a2['s10'])} vs {pct(a1['s10'])} above 10 days). The pipeline's mixed setup (ZNB1 from goods receipt, ZJD1 from arrival) turns that around: {pct(g2['s10'])} vs {pct(a1['s10'])}. With the same goods-receipt clock for both, ZNB1 is worse again: {pct(g2['s10'])} vs {pct(g1['s10'])}.</p>
<p><b>ZJD1 gains most from the goods-receipt clock.</b> Its share above 10 days falls from {pct(a1['s10'])} to {pct(g1['s10'])} and the missed discount from {mnok(a1['missed'])} to {mnok(g1['missed'])} NOK. Most of ZJD1's waiting is for the goods, which is a purchasing delay, not an accounts-payable one.</p>
</div>

<h3>Split by goods-receipt step (clock from goods receipt)</h3>
{table(["Invoices", "Type", "Count", "Share of type", "Median days", "&gt; 10 d", "&gt; 30 d", "Missed 2% (NOK)", "Share of type's missed"], split_rows)}
<div class="callout">
<p><b>With a goods-receipt step, ZJD1 and ZNB1 behave the same:</b> {pct(s1t['s10'])} and {pct(s2t['s10'])} above 10 days. Once the goods are booked, both post almost on time.</p>
<p><b>Without one, ZNB1 is much slower.</b> In both types 92–93% of these invoices were sent to the manual control queue (<code>CTRL1</code>) at validation and waited for a clerk. ZNB1 has more of them ({pct(s2f['n'] / g2['n'], 0)} vs {pct(s1f['n'] / g1['n'], 0)} of invoices), they're late more often ({pct(s2f['s10'])} vs {pct(s1f['s10'])}), and they cause {pct(s2f['missed'] / g2['missed'], 0)} of ZNB1's missed discount. For these invoices the clock still starts at arrival, so part of their delay may be waiting for goods that the log can't show.</p>
</div>

{FEAS}

<footer>
<p>Built with Python (conda env <code>datasci</code>) by <code>07_report_clock_compare.py</code> from <code>output/invoices.csv</code> and <code>cases.csv</code>. Run steps 01–04 first. The ZNB1 goods-receipt figures reproduce the headline in the main reports ({num(g2['n10'])} late invoices, {mnok(g2['missed'])} NOK).</p>
</footer>
</div>
"""
(OUT / "report_zjd1_znb1_clock.html").write_text(page, encoding="utf-8")
print("wrote", OUT / "report_zjd1_znb1_clock.html", len(page) // 1024, "KB")
for k, r in R.items():
    print(k, f"{r['s10']:.1%} {r['s30']:.1%} {r['missed'] / 1e6:.1f}M")
