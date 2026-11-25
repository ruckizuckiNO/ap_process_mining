"""Builds output/report_invoice_clock.html: the per-invoice SLA clock, explained step by step, and ZJD1 vs ZNB1 under it.

The clock starts at goods receipt (first WAIT_GR event of the invoice) or, without one, at the invoice's first event, and
stops at the invoice's first posting. Applied to ZJD1 and ZNB1 alike. Note: the pipeline (GR_CLOCK_PO_TYPES) uses the
goods-receipt start for ZNB1 only, so ZJD1 figures here differ from the other reports. Reads output/invoices.csv,
cases.csv and events.pkl; reuses the stylesheet of 05_report.py. report_zjd1_znb1_clock.html is not touched.
"""
import html
import re

import pandas as pd

from clock_doc import logic_html, feasibility
from common import (OUT, ROOT, POSTING_ACTIVITIES, SLA_SOFT_DAYS, SLA_HARD_DAYS, GR_ACTIVITY, GR_OBJECT_CODE,
                    load_invoices, load_cases, load_events)

TYPES = ["ZJD1", "ZNB1"]
EX = "PurDoc184235"  # ZNB1, two invoices: one with a goods-receipt step, one without (late)
DISCOUNT = 0.02
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


def ts(t):
    return f"{t:%d %b %Y %H:%M:%S}"


# ---------------------------------------------------------------- the clock, per invoice
inv = load_invoices()
inv = inv[inv.po_type.isin(TYPES)].copy()
cases = load_cases().set_index("PO_ID")
inv["has_gr"] = inv.goods_receipt.notna()
inv["clock_start"] = inv.goods_receipt.fillna(inv.start)                                       # step 3
inv["days"] = (inv.first_post - inv.clock_start).dt.total_seconds() / 86400                   # step 5
p = inv[inv.posted].copy()                                                                    # step 6
p["n_posted_on_po"] = p.groupby("PO_ID").PO_ID.transform("size")
p["po_value"] = cases.invoice_value_nok.reindex(p.PO_ID).values
p["inv_value"] = p.po_value / p.n_posted_on_po                                                 # step 8
p["late10"], p["late30"] = p.days > SLA_SOFT_DAYS, p.days > SLA_HARD_DAYS                      # step 7
p["missed"] = DISCOUNT * p.inv_value * p.late10
assert (p.days >= 0).all()
# ZNB1 must reproduce the pipeline's headline (the pipeline uses this same clock for ZNB1)
_z = p[p.po_type == "ZNB1"]
assert abs(_z.missed.sum() - DISCOUNT * _z.inv_value[_z.time_to_post_days > SLA_SOFT_DAYS].sum()) < 1


def stats(g):
    return dict(pos=g.PO_ID.nunique(), n=len(g), median=g.days.median(), p90=g.days.quantile(0.9),
                n10=int(g.late10.sum()), s10=g.late10.mean(), n30=int(g.late30.sum()), s30=g.late30.mean(),
                missed=g.missed.sum(), value=g.inv_value.sum())


R = {t: stats(p[p.po_type == t]) for t in TYPES}
SPLIT = {(t, h): stats(p[(p.po_type == t) & (p.has_gr == h)]) for t in TYPES for h in (True, False)}
NP = inv[~inv.posted].groupby("po_type").size()
ALL_INV = inv.groupby("po_type").size()
GRS = p.groupby("po_type").has_gr.mean()
r1, r2 = R["ZJD1"], R["ZNB1"]

top10 = max(r["s10"] for r in R.values())
top30 = max(r["s30"] for r in R.values())
cmp_rows = [
    ("Purchase orders (POs)", lambda r, t: num(r["pos"])),
    ("Invoices on these POs", lambda r, t: num(ALL_INV[t])),
    ("Never posted (left out)", lambda r, t: num(NP.get(t, 0))),
    ("Posted invoices (the base)", lambda r, t: f"<b>{num(r['n'])}</b>"),
    ("Posted invoices per PO", lambda r, t: num(r["n"] / r["pos"], 2)),
    ("With a goods-receipt step", lambda r, t: pct(GRS[t], 0)),
    ("Median days to posting", lambda r, t: num(r["median"], 2)),
    ("90th percentile, days", lambda r, t: num(r["p90"], 1)),
    ("Invoices &gt; 10 days", lambda r, t: num(r["n10"])),
    ("<b>Share &gt; 10 days</b>", lambda r, t: f"<b>{pct(r['s10'])}</b>"),
    ("Invoices &gt; 30 days", lambda r, t: num(r["n30"])),
    ("<b>Share &gt; 30 days</b>", lambda r, t: f"<b>{pct(r['s30'])}</b>"),
    ("Missed 2% discount (NOK)", lambda r, t: mnok(r["missed"])),
    ("Missed discount per posted invoice (NOK)", lambda r, t: num(r["missed"] / r["n"])),
]
cmp_html = table(["", code("ZJD1"), code("ZNB1")],
                 [f"<td>{lab}</td>" + "".join(f"<td class='n'>{f(R[t], t)}</td>" for t in TYPES) for lab, f in cmp_rows])
bar_rows = [f"<td>{code(t)}</td><td class='n'>{pct(R[t]['s10'])}</td><td>{bar(R[t]['s10'], top10)}</td>"
            f"<td class='n'>{pct(R[t]['s30'])}</td><td>{bar(R[t]['s30'], top10)}</td>" for t in TYPES]

split_rows = []
for h, lab in [(True, "With a goods-receipt step (clock from goods receipt)"), (False, "Without (clock from first event)")]:
    for t in TYPES:
        r, tot = SPLIT[(t, h)], R[t]
        split_rows.append(f"<td>{lab if t == TYPES[0] else ''}</td><td>{code(t)}</td><td class='n'>{num(r['n'])}</td>"
                          f"<td class='n'>{pct(r['n'] / tot['n'], 0)}</td><td class='n'>{num(r['median'], 2)}</td>"
                          f"<td class='n'><b>{pct(r['s10'])}</b></td><td class='n'>{pct(r['s30'])}</td>"
                          f"<td class='n'>{mnok(r['missed'])}</td><td class='n'>{pct(r['missed'] / tot['missed'], 0)}</td>")
s1t, s1f, s2t, s2f = SPLIT[("ZJD1", True)], SPLIT[("ZJD1", False)], SPLIT[("ZNB1", True)], SPLIT[("ZNB1", False)]

# ---------------------------------------------------------------- worked example
ev = load_events()
ev = ev[ev.PO_ID == EX]
ex_type = ev.SAP_PO_Type_Code.iloc[0]
xi = p[p.PO_ID == EX].set_index("SAP_PO_No")
first_ev = ev.groupby("SAP_PO_No").Event_ID.first()
gr_ev = ev[(ev.Activity_ID == GR_ACTIVITY) & (ev.SAP_Object_Code == GR_OBJECT_CODE)].groupby("SAP_PO_No").Event_ID.first()
post_ev = ev[ev.Activity_ID.isin(POSTING_ACTIVITIES)].groupby("SAP_PO_No").Event_ID.first()
inv_letter = {no: chr(65 + i) for i, no in enumerate(ev.drop_duplicates("SAP_PO_No").SAP_PO_No)}
ex_rows = []
for i, (_, r) in enumerate(ev.iterrows(), 1):
    no = r.SAP_PO_No
    role = ""
    if r.Event_ID == gr_ev.get(no):
        role = '<span class="tag g">clock start (goods receipt)</span>'
    elif r.Event_ID == first_ev.get(no) and no not in gr_ev.index:
        role = '<span class="tag a">clock start (first event, no goods receipt)</span>'
    elif r.Event_ID == first_ev.get(no):
        role = '<span class="tag n">first event, not used</span>'
    elif r.Event_ID == post_ev.get(no):
        role = '<span class="tag s">clock stop (first posting)</span>'
    elif r.Activity_ID == "Data management":
        role = '<span class="tag n">restart of the same invoice, not used</span>'
    elif r.Activity_ID == GR_ACTIVITY:
        oc = "no object code" if pd.isna(r.SAP_Object_Code) else html.escape(r.SAP_Object_Code)
        role = f'<span class="tag n">{oc}: doesn\'t count</span>'
    obj = "" if pd.isna(r.SAP_Object_Code) else f"<code>{html.escape(r.SAP_Object_Code)}</code>"
    ex_rows.append(f"<td class='n'>{i}</td><td class='n'>{r.ts:%d %b %H:%M:%S}</td><td>{html.escape(r.Activity_ID)}</td>"
                   f"<td class='n'>{no} ({inv_letter[no]})</td><td>{obj}</td><td>{role}</td>")

ex_calc = []
for no, r in xi.iterrows():
    L = inv_letter[no]
    src = "goods receipt" if r.has_gr else "first event (no WAIT_GR step)"
    ex_calc.append(f"Invoice {L} ({no})\n"
                   f"  clock start = {ts(r.clock_start)}   ({src})\n"
                   f"  clock stop  = {ts(r.first_post)}   (first posting)\n"
                   f"  days        = {(r.first_post - r.clock_start).total_seconds():,.0f} s / 86,400 = {r.days:.3f} days\n"
                   f"  > 10 days?  {'yes' if r.late10 else 'no'}      > 30 days?  {'yes' if r.late30 else 'no'}")
ex_value = xi.po_value.iloc[0]
ex_n = len(xi)
ex_late = xi[xi.late10]
ex_arr0 = (xi.first_post - xi.start).dt.total_seconds() / 86400
ex_calc.append(f"Discount\n"
               f"  PO value {num(ex_value, 2)} NOK / {ex_n} posted invoices = {num(ex_value / ex_n, 2)} NOK per invoice\n"
               f"  missed   = 2% x {num(ex_value / ex_n, 2)} x {len(ex_late)} late invoice(s) = {num(xi.missed.sum(), 2)} NOK")
ex_math = html.escape("\n\n".join(ex_calc))
gr_inv = xi[xi.has_gr].iloc[0]
gr_letter = inv_letter[xi[xi.has_gr].index[0]]

LOGIC = logic_html(two_clocks=False)
FEAS, FS = feasibility(TYPES)

# ---------------------------------------------------------------- page
EXTRA_CSS = """
.bar { background: var(--accent-soft); border-radius: 3px; height: 10px; min-width: 90px; }
.bar span { display: block; height: 100%; background: var(--accent); border-radius: 3px; }
.tag { display: inline-block; font: 500 10.5px/1.2 var(--f-mono); padding: 3px 6px; border-radius: 4px; white-space: nowrap; }
.tag.a { background: var(--warn-bg); color: var(--warn); }
.tag.g { background: var(--ok-bg); color: var(--ok); }
.tag.s { background: var(--accent-soft); color: var(--accent); }
.tag.n { border: 1px dashed var(--line); color: var(--muted); }
.tag.o { border: 1px dashed var(--line); color: var(--muted); }
ol.clock { list-style: none; padding: 0; margin: 18px 0; display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 170px), 1fr)); gap: 10px; }
ol.clock li { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 12px 14px; min-width: 0; }
ol.clock li b { display: block; font-size: 14px; }
ol.clock li small { display: block; color: var(--muted); font: 12px var(--f-mono); margin: 2px 0 8px; }
ol.clock li.out { border-style: dashed; color: var(--muted); }
.rec li code { white-space: nowrap; }
"""

timeline = f"""<ol class="clock">
  <li><b>Data management</b><small>invoice arrives in AP</small><span class="tag a">start if no goods-receipt step</span></li>
  <li><b>Initial validation</b><small>routing code set</small></li>
  <li><b>{GR_ACTIVITY}</b><small>object code {GR_OBJECT_CODE}</small><span class="tag g">start: goods receipt</span></li>
  <li><b>First posting</b><small>invoice payable</small><span class="tag s">stop</span></li>
  <li class="out"><b>Payment run</b><small>not in the log</small><span class="tag o">real SLA end, not measurable</span></li>
</ol>"""

page = f"""<meta charset="utf-8">
<title>Invoice Clock ZJD1 ZNB1</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,500;6..72,600&family=Instrument+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap">
<style>{CSS}{EXTRA_CSS}</style>

<div class="wrap">
<header>
  <div class="eyebrow">Accounts payable · Process mining case · Invoice clock</div>
  <h1>The invoice clock: ZJD1 vs ZNB1</h1>
  <p class="lede">How the time from goods receipt to posting is measured for each invoice, step by step, and how normal orders ({code('ZJD1')}) and manual orders ({code('ZNB1')}) compare against the 10-day and 30-day limits.</p>
  <div class="facts">
    <span><b>{num(r1['n'])}</b> ZJD1 invoices</span>
    <span><b>{num(r2['n'])}</b> ZNB1 invoices</span>
    <span>limits <b>{SLA_SOFT_DAYS} d</b> (2% discount) and <b>{SLA_HARD_DAYS} d</b> (payment deadline)</span>
  </div>
</header>

<h2 id="answer">Short answer</h2>
<div class="findings">
  <div class="finding"><div class="k">{pct(r1['s10'])} vs {pct(r2['s10'])}</div><p>of posted invoices take longer than 10 days in ZJD1 and ZNB1. ZNB1 is late about twice as often.</p></div>
  <div class="finding"><div class="k">{pct(r1['s30'])} vs {pct(r2['s30'])}</div><p>take longer than 30 days. On the payment deadline the two are about equal.</p></div>
  <div class="finding"><div class="k">{mnok(r1['missed'])} vs {mnok(r2['missed'])}</div><p>NOK of 2% cash discount is missed on the late invoices.</p></div>
  <div class="finding"><div class="k">{pct(s1t['s10'])} = {pct(s2t['s10'])}</div><p>With a goods-receipt step both types are equally fast. The gap is in the invoices without one: {pct(s1f['s10'])} late in ZJD1, {pct(s2f['s10'])} in ZNB1.</p></div>
</div>

<h2 id="logic"><span class="step">1</span>How the invoice clock is calculated</h2>
<div class="prose"><p>The cash discount and the payment deadline both run from when the supplier's claim is due, which is assumed to be the <b>goods receipt</b>. They end at <b>payment</b>. The event log has neither a goods-receipt date nor a payment date, so the clock uses the closest events that are in the log:</p></div>
{timeline}
{LOGIC}

<h2 id="example"><span class="step">2</span>Worked example: {EX} ({code(ex_type)})</h2>
<div class="prose"><p>This PO received two invoices. Every event in the log for it:</p></div>
{table(["#", "Timestamp", "Activity", "Invoice", "Object code", "Role in the clock"], ex_rows)}
<div class="math">{ex_math}</div>
<div class="prose">
<p>Invoice {gr_letter} shows why the start matters: it arrived {ex_arr0[gr_inv.name]:.1f} days before the goods receipt and was posted seconds after it. From goods receipt it's on time; from arrival it would count the wait for the goods against accounts payable.</p>
<p>The other invoice had <i>{GR_ACTIVITY}</i> steps too, but never with <code>{GR_OBJECT_CODE}</code>: one has no object code and the other is <code>END_WAIT_GR</code>. The wait ended without a goods receipt, the invoice went to the manual queue and was posted by a person two weeks later. Without a <code>{GR_OBJECT_CODE}</code> event its clock starts at its first event, and it misses the 10-day limit.</p>
</div>

<h2 id="compare"><span class="step">3</span>ZJD1 vs ZNB1</h2>
<div class="prose"><p>The same clock applied to both PO types:</p></div>
{cmp_html}
{table(["Type", "&gt; 10 days", "", "&gt; 30 days", ""], bar_rows)}
<div class="callout">
<p><b>ZNB1 misses the 10-day limit about twice as often</b> ({pct(r2['s10'])} vs {pct(r1['s10'])}) and loses more discount ({mnok(r2['missed'])} vs {mnok(r1['missed'])} NOK), although its POs are worth slightly less on average. Above 30 days the two are about the same ({pct(r1['s30'])} vs {pct(r2['s30'])}).</p>
</div>

<h3>Split by clock start</h3>
{table(["Clock starts at", "Type", "Invoices", "Share of type", "Median days", "&gt; 10 d", "&gt; 30 d", "Missed 2% (NOK)", "Share of type's missed"], split_rows)}
<div class="callout">
<p><b>With a goods-receipt step, ZJD1 and ZNB1 are equally fast:</b> {pct(s1t['s10'])} and {pct(s2t['s10'])} above 10 days. Once the goods are booked, both post almost at once.</p>
<p><b>Without one, ZNB1 is much slower.</b> In both types these are mostly invoices sent to the manual control queue (<code>CTRL1</code>) at validation. ZNB1 has more of them ({pct(s2f['n'] / r2['n'], 0)} vs {pct(s1f['n'] / r1['n'], 0)} of invoices), they're late more often ({pct(s2f['s10'])} vs {pct(s1f['s10'])}), and they account for {pct(s2f['missed'] / r2['missed'], 0)} of ZNB1's missed discount. Part of that gap may be waiting for goods that the log can't show for these invoices (see <a href="#feasible">Is the goods-receipt clock fair?</a>).</p>
</div>

{FEAS}

<h2 id="limits"><span class="step">4</span>Assumptions and limits</h2>
<ul class="tight prose">
  <li><b>Goods receipt is a proxy.</b> The <code>{GR_OBJECT_CODE}</code> event marks when SAP found the goods receipt, not necessarily when it was booked. For invoices without that step the clock starts at arrival, so any wait for goods before arrival is invisible, and any wait after arrival is counted.</li>
  <li><b>Posting isn't payment.</b> "Within 10 days" means posted within 10 days, which is needed to pay on time but doesn't prove it.</li>
  <li><b>Invoice value is estimated</b> by splitting the PO value evenly over its posted invoices. The real amount per invoice is in SAP.</li>
  <li><b>Not the pipeline default for ZJD1.</b> The other reports start the ZJD1 clock at arrival (the goods-receipt start is set for ZNB1 only), so their ZJD1 figures are higher. The ZNB1 figures here match the other reports exactly ({num(r2['n10'])} late invoices, {mnok(r2['missed'])} NOK).</li>
</ul>

<footer>
<p>Built with Python (conda env <code>datasci</code>) by <code>10_report_invoice_clock.py</code> from <code>output/invoices.csv</code> (columns <code>start</code>, <code>goods_receipt</code>, <code>first_post</code>, built in <code>01_prepare.py</code>), <code>cases.csv</code> and <code>events.pkl</code>.</p>
</footer>
</div>
"""
(OUT / "report_invoice_clock.html").write_text(page, encoding="utf-8")
print("wrote", OUT / "report_invoice_clock.html", len(page) // 1024, "KB")
for t in TYPES:
    print(t, {k: round(v, 3) for k, v in R[t].items()})
print(html.unescape(ex_math))
