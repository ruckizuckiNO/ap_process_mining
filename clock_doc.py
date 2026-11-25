"""Shared documentation of the per-invoice SLA clock, used by 07_report_clock_compare.py and 10_report_invoice_clock.py.

- logic_html(): the step-by-step calculation (one clock or both clocks)
- feasibility(): evidence for and against starting the clock at WAIT_GR, per PO type, as an HTML section
"""
import numpy as np
import pandas as pd

from common import POSTING_ACTIVITIES, GR_ACTIVITY, GR_OBJECT_CODE, SLA_SOFT_DAYS, load_events

KEY = ["PO_ID", "SAP_PO_No"]


def _table(head, rows):
    th = "".join(f"<th>{h}</th>" for h in head)
    body = "".join(f"<tr>{r}</tr>" for r in rows)
    return f'<div class="tbl"><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def _pct(x, d=1):
    return "–" if pd.isna(x) else f"{x * 100:.{d}f}%"


def _code(c):
    return f'<span class="code">{c}</span>'


def logic_html(two_clocks=False):
    """Numbered steps and formulas. two_clocks=True also documents the arrival clock (clock A)."""
    if two_clocks:
        start = f"""<li><b>Clock start, two variants.</b>
    <ul class="tight">
      <li><b>A · from arrival:</b> the invoice's <b>first event</b> (<i>Data management</i>, arrival in accounts payable). The default in the pipeline for every PO type except ZNB1.</li>
      <li><b>B · from goods receipt:</b> the invoice's <b>first</b> <i>{GR_ACTIVITY}</i> event with <code>SAP_Object_Code</code> = <code>{GR_OBJECT_CODE}</code>. SAP logs this event when the goods-receipt check finds the booked goods receipt, so it is read as the goods-receipt date. A <i>{GR_ACTIVITY}</i> event with <code>END_WAIT_GR</code> (the wait ended <i>without</i> a goods receipt) or with no object code does <b>not</b> count. If the invoice has no <code>{GR_OBJECT_CODE}</code> event, clock B falls back to clock A.</li>
    </ul></li>"""
        formula_start = (f"clock_start A = first ts of the invoice\n"
                         f"clock_start B = first ts of ({GR_ACTIVITY}, {GR_OBJECT_CODE})  if the invoice has one,  else clock_start A")
    else:
        start = f"""<li><b>Clock start.</b> The timestamp of the invoice's <b>first</b> <i>{GR_ACTIVITY}</i> event with <code>SAP_Object_Code</code> = <code>{GR_OBJECT_CODE}</code>. SAP logs this event when the goods-receipt check finds the booked goods receipt, so it is read as the goods-receipt date.
    <ul class="tight">
      <li>A <i>{GR_ACTIVITY}</i> event with <code>END_WAIT_GR</code> (the wait ended <i>without</i> a goods receipt and the invoice was released to the manual queue) or with no object code does <b>not</b> count.</li>
      <li>If the invoice has no <code>{GR_OBJECT_CODE}</code> event, the clock starts at its <b>first event</b> (<i>Data management</i>, arrival in accounts payable). These are mostly invoices sent straight to the manual queue at validation. The log gives no goods-receipt date for them (see <a href="#feasible">Is this clock fair?</a>).</li>
    </ul></li>"""
        formula_start = (f"clock_start = first ts of ({GR_ACTIVITY}, {GR_OBJECT_CODE})  if the invoice has one,  "
                         f"else first ts of the invoice")
    return f"""<ol class="rec">
  <li><b>One clock per invoice.</b> A purchase order (<code>PO_ID</code>) can receive several invoices, e.g. one per partial delivery. Each invoice has its own document number <code>SAP_PO_No</code> and starts again with <i>Data management</i>. The unit is one invoice = one <code>PO_ID</code> × <code>SAP_PO_No</code> pair; all events with that pair belong to it.</li>
  <li><b>Order the events.</b> By timestamp, with <code>Event_ID</code> breaking ties. <code>Item_No</code> isn't used, because it isn't in time order for about 45,000 events.</li>
  {start}
  <li><b>Clock stop.</b> The invoice's <b>first posting</b>: the earliest of <i>{"</i>, <i>".join(POSTING_ACTIVITIES)}</i>. Posting makes the invoice payable. Later postings of the same invoice don't move the stop. The payment itself isn't in the log.</li>
  <li><b>Lead time in days.</b> Stop minus start in seconds, divided by 86,400. Calendar days with fractions, so weekends and holidays count, as they do for payment terms.</li>
  <li><b>Base.</b> Only <b>posted</b> invoices have a stop time. Invoices that were never posted are left out of every share and count.</li>
  <li><b>Late or not.</b> Late means a lead time of <b>more than</b> 10 days (cash discount) or more than 30 days (payment deadline). Share = late invoices ÷ posted invoices.</li>
  <li><b>Missed discount.</b> The value is only given per PO (<code>Case_Invoice_Value_NOK</code>, the same on every row of a PO). It's split evenly over the PO's posted invoices, and each invoice later than 10 days misses 2% of its share.</li>
</ol>
<div class="math">{formula_start}
clock_stop  = first ts of a posting activity
days        = (clock_stop − clock_start) in seconds / 86,400
late_10     = days &gt; 10          late_30 = days &gt; 30
share_10    = Σ late_10 / number of posted invoices
missed      = Σ over late_10 invoices of  2% × PO value / posted invoices on the PO</div>"""


def _invoice_table(types):
    ev = load_events()
    ev = ev[ev.SAP_PO_Type_Code.isin(types)].sort_values(["PO_ID", "ts", "Event_ID"])
    g = ev.groupby(KEY)
    is_gr = (ev.Activity_ID == GR_ACTIVITY) & (ev.SAP_Object_Code == GR_OBJECT_CODE)
    d = pd.DataFrame({"po_type": g.SAP_PO_Type_Code.first(), "start": g.ts.min(),
                      "gr": ev[is_gr].groupby(KEY).ts.first(),
                      "post": ev[ev.Activity_ID.isin(POSTING_ACTIVITIES)].groupby(KEY).ts.first(),
                      "route": ev[ev.Activity_ID == "Initial validation"].groupby(KEY).SAP_Next_Processing_Type.first()})
    d = d[d.post.notna()].copy()
    d["has_gr"] = d.gr.notna()
    days = lambda a, b: (b - a).dt.total_seconds() / 86400
    d["arr"], d["arr_to_gr"], d["gr_to_post"] = days(d.start, d.post), days(d.start, d.gr), days(d.gr, d.post)
    d["grc"] = np.where(d.has_gr, d.gr_to_post, d.arr)
    nxt = ev.groupby(KEY).Activity_ID.shift(-1)
    after_gr = nxt[is_gr].groupby(ev.loc[is_gr, "SAP_PO_Type_Code"]).agg(lambda s: (s == "MM Posted Automatically").mean())
    return d, after_gr


def feasibility(types=("ZJD1", "ZNB1")):
    """HTML section 'Is this clock fair?' plus the numbers behind it."""
    types = list(types)
    d, after_gr = _invoice_table(types)
    ev_rows, route_rows, S = [], [], {}
    for t in types:
        x = d[(d.po_type == t) & d.has_gr]
        q = x.arr_to_gr.quantile([0.5, 0.75, 0.9])
        S[t] = dict(n_gr=len(x), within_min=(x.gr_to_post < 1 / 1440).mean(), auto_next=after_gr.get(t, np.nan),
                    q50=q[0.5], q75=q[0.75], q90=q[0.9], gt30=(x.arr_to_gr > 30).mean())
        s = S[t]
        ev_rows.append(f"<td>{_code(t)}</td><td class='n'>{s['n_gr']:,}</td><td class='n'>{_pct(s['within_min'], 0)}</td>"
                       f"<td class='n'>{_pct(s['auto_next'], 0)}</td><td class='n'>{s['q50']:.1f}</td><td class='n'>{s['q75']:.1f}</td>"
                       f"<td class='n'>{s['q90']:.1f}</td>")
        for r in ["WAITGR", "CTRL1"]:
            y = d[(d.po_type == t) & (d.route == r)]
            a10, g10 = (y.arr > SLA_SOFT_DAYS).mean(), (y.grc > SLA_SOFT_DAYS).mean()
            S[(t, r)] = dict(n=len(y), has_gr=y.has_gr.mean(), a10=a10, g10=g10)
            label = "goods-receipt check" if r == "WAITGR" else "manual queue"
            route_rows.append(f"<td>{_code(t) if r == 'WAITGR' else ''}</td><td><code>{r}</code> <small>{label}</small></td>"
                              f"<td class='n'>{len(y):,}</td><td class='n'>{_pct(y.has_gr.mean(), 0)}</td>"
                              f"<td class='n'>{_pct(a10)}</td><td class='n'><b>{_pct(g10)}</b></td>")
    n1 = S[("ZNB1", "WAITGR")] if "ZNB1" in types else S[(types[0], "WAITGR")]
    c1 = S[("ZNB1", "CTRL1")] if "ZNB1" in types else S[(types[0], "CTRL1")]
    html_ = f"""<h2 id="feasible"><span class="step">Check</span>Is the goods-receipt clock fair?</h2>
<div class="prose"><p>Starting the clock at <code>{GR_OBJECT_CODE}</code> instead of arrival removes the time an invoice waits for its goods, which accounts payable can't speed up. Two questions decide whether that's sound: is the event a good goods-receipt date, and can it be applied to every invoice?</p></div>
<h3>1 · <code>{GR_OBJECT_CODE}</code> is a usable goods-receipt date</h3>
{_table(["Type", "Posted invoices with WAIT_GR", "Posted &lt; 1 min after it", "Next step is automatic posting", "Arrival → WAIT_GR, median days", "75th pct", "90th pct"], ev_rows)}
<div class="prose"><p>Most invoices are posted within a minute of the event, and the next step is usually <i>MM Posted Automatically</i>: the event is the moment SAP found the goods receipt and released the invoice. The lag between the real booking and SAP's check is probably at most about a day (the median wait looks like a daily check), small against a 10-day limit. The wait before it is real: a quarter of these invoices wait several days for their goods, and one in ten waits weeks.</p></div>
<h3>2 · But it exists only on one route</h3>
<div class="prose"><p>At <i>Initial validation</i> SAP sends an invoice either to the goods-receipt check (<code>WAITGR</code>) or to the manual control queue (<code>CTRL1</code>). Only the first road produces a <code>{GR_OBJECT_CODE}</code> event. Share of posted invoices taking longer than 10 days, by route and clock:</p></div>
{_table(["Type", "Route at validation", "Posted invoices", "Have WAIT_GR", "&gt; 10 d, clock from arrival", "&gt; 10 d, clock from goods receipt"], route_rows)}
<div class="callout">
<p><b>The two routes are treated differently.</b> From arrival, ZNB1's two routes are late about equally often ({_pct(n1['a10'])} and {_pct(c1['a10'])}). The goods-receipt clock removes the wait for goods on the <code>WAITGR</code> route ({_pct(n1['g10'])}), but it can't on the <code>CTRL1</code> route ({_pct(c1['g10'])}), because only {_pct(c1['has_gr'])} of those invoices have a goods-receipt date in the log. If clerks in the manual queue also wait for goods (holding an invoice until delivery), that wait still counts against them.</p>
<p><b>Consequence:</b> comparisons between the two routes aren't like-for-like. "The manual queue causes most of the missed discount" is partly a result of the clock and not proven by the log.</p>
</div>
<h3>3 · Which clock for which question</h3>
{_table(["", "Clock from goods receipt (WAIT_GR)", "Clock from arrival (ignore WAIT_GR)"], [
        "<td>Measures</td><td>How fast accounts payable is once it can act</td><td>Time from arrival in AP to payable</td>",
        "<td>Fair to AP?</td><td>Yes on the <code>WAITGR</code> route; not on <code>CTRL1</code></td><td>No: supplier and purchasing delays count against AP</td>",
        "<td>Same rule on every route?</td><td>No</td><td>Yes</td>",
        "<td>Money actually lost</td><td>Lower bound</td><td>Closer to it, if payment terms run from the invoice date (and still low, because the invoice date usually comes before arrival)</td>",
        "<td>Use as</td><td><b>Headline for AP performance</b></td><td><b>Upper bound / money at risk</b></td>"])}
<div class="prose"><p><b>Decision:</b> keep the goods-receipt clock as the headline, always show the arrival clock next to it, and don't compare the two routes without this caveat. What the company's payment terms count from (invoice date or goods receipt) should be checked; the log can't show it.</p></div>"""
    return html_, S
