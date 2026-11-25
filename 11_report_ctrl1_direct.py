"""Builds output/report_ctrl1_direct.html: ZJD1 and ZNB1 invoices sent straight to the manual control queue (CTRL1)
at their first Initial validation.

Clock as in 10_report_invoice_clock.py (start at the first WAIT_GR event, else the invoice's first event; stop at the
first posting), applied to both types. Reads output/events.pkl, invoices.csv and cases.csv; reuses the stylesheet of
05_report.py.
"""
import html
import re

import numpy as np
import pandas as pd

from common import (OUT, ROOT, BOT_USERS, POSTING_ACTIVITIES, SLA_SOFT_DAYS, SLA_HARD_DAYS, GR_ACTIVITY, GR_OBJECT_CODE,
                    load_events, load_invoices, load_cases)

TYPES = ["ZJD1", "ZNB1"]
KEY = ["PO_ID", "SAP_PO_No"]
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


def bar(x, top=1.0):
    return f'<div class="bar"><span style="width:{min(x / top, 1) * 100:.1f}%"></span></div>'


# ---------------------------------------------------------------- invoices of both types, with the invoice clock
inv = load_invoices()
inv = inv[inv.po_type.isin(TYPES)].set_index(KEY)
cases = load_cases().set_index("PO_ID")
inv["clock_start"] = inv.goods_receipt.fillna(inv.start)
inv["days"] = (inv.first_post - inv.clock_start).dt.total_seconds() / 86400
inv["n_posted_on_po"] = inv.groupby(level="PO_ID").posted.transform("sum")
inv["inv_value"] = np.where(inv.posted, cases.invoice_value_nok.reindex(inv.index.get_level_values(0)).values
                            / inv.n_posted_on_po.where(inv.n_posted_on_po > 0), 0)
inv["late10"] = inv.posted & (inv.days > SLA_SOFT_DAYS)
inv["late30"] = inv.posted & (inv.days > SLA_HARD_DAYS)
inv["missed"] = DISCOUNT * inv.inv_value * inv.late10

ev = load_events()
ev = ev[ev.SAP_PO_Type_Code.isin(TYPES)].sort_values(["PO_ID", "ts", "Event_ID"])
val = ev[ev.Activity_ID == "Initial validation"].groupby(KEY)
inv["route"] = val.SAP_Next_Processing_Type.first()
inv["val_ts"] = val.ts.first()
ev["nxt"] = ev.groupby(KEY).Activity_ID.shift(-1)
inv["after_val"] = ev[ev.Activity_ID == "Initial validation"].groupby(KEY).nxt.first()
post_ev = ev[ev.Activity_ID.isin(POSTING_ACTIVITIES)].groupby(KEY)
inv["post_act"], inv["poster"] = post_ev.Activity_ID.first(), post_ev.User_ID.first()
inv["deleted"] = ev[ev.Activity_ID == "Invoice document deleted"].groupby(KEY).size().reindex(inv.index).notna()
inv["supplier"] = ev.groupby(KEY).Supplier_ID.first()

C = inv[inv.route == "CTRL1"].copy()              # the invoices this report is about
P = C[C.posted].copy()
P["arr_to_val"] = (P.val_ts - P.start).dt.total_seconds() / 86400
P["val_to_post"] = (P.first_post - P.val_ts).dt.total_seconds() / 86400


def stats(t):
    a, c, p = inv[inv.po_type == t], C[C.po_type == t], P[P.po_type == t]
    ap = a[a.posted]
    return dict(all=len(a), n=len(c), share=len(c) / len(a), posted=len(p), notposted=len(c) - len(p),
                deleted=int(c.deleted.sum() - p.deleted.sum()), has_gr=p.goods_receipt.notna().mean(),
                median=p.days.median(), p75=p.days.quantile(.75), p90=p.days.quantile(.9),
                n10=int(p.late10.sum()), s10=p.late10.mean(), n30=int(p.late30.sum()), s30=p.late30.mean(),
                missed=p.missed.sum(), missed_type=ap.missed.sum(), late_type=int(ap.late10.sum()),
                s10_other=ap[ap.route != "CTRL1"].late10.mean(), pos=c.index.get_level_values(0).nunique(),
                direct=(c.after_val == "Posted manually").mean(), human=(~p.poster.isin(BOT_USERS)).mean(),
                arr_to_val_s=p.arr_to_val.median() * 86400)


R = {t: stats(t) for t in TYPES}
r1, r2 = R["ZJD1"], R["ZNB1"]


def per_type_rows(rows):
    return [f"<td>{lab}</td>" + "".join(f"<td class='n'>{f(R[t])}</td>" for t in TYPES) for lab, f in rows]


ov_html = table(["", code("ZJD1"), code("ZNB1")], per_type_rows([
    ("All invoices of the type", lambda r: num(r["all"])),
    ("<b>Sent straight to CTRL1</b>", lambda r: f"<b>{num(r['n'])}</b> ({pct(r['share'], 0)})"),
    ("on POs", lambda r: num(r["pos"])),
    ("Never posted (left out)", lambda r: num(r["notposted"])),
    ("Posted (the base below)", lambda r: num(r["posted"])),
    ("Have a WAIT_GR step later", lambda r: pct(r["has_gr"])),
    ("Median days to posting", lambda r: num(r["median"], 1)),
    ("75th / 90th percentile, days", lambda r: f"{num(r['p75'], 1)} / {num(r['p90'], 1)}"),
    ("<b>Share &gt; 10 days</b>", lambda r: f"<b>{pct(r['s10'])}</b> ({num(r['n10'])})"),
    ("<b>Share &gt; 30 days</b>", lambda r: f"<b>{pct(r['s30'])}</b> ({num(r['n30'])})"),
    ("Share &gt; 10 days, all other invoices of the type", lambda r: pct(r["s10_other"])),
    ("Missed 2% discount (NOK)", lambda r: mnok(r["missed"])),
    ("Share of the type's missed discount", lambda r: pct(r["missed"] / r["missed_type"], 0)),
    ("Share of the type's late invoices", lambda r: pct(r["n10"] / r["late_type"], 0)),
]))

# What happens right after validation
nxt = C.groupby(["after_val", "po_type"]).size().unstack(fill_value=0)
nxt = nxt.div(nxt.sum()).sort_values("ZNB1", ascending=False)
nxt = pd.concat([nxt.head(5), nxt.iloc[5:].sum().to_frame("Other").T])
nxt_rows = [f"<td>{html.escape(a)}</td>" + "".join(f"<td class='n'>{pct(r[t])}</td>" for t in TYPES) for a, r in nxt.iterrows()]

# Lead-time bands
edges = [-np.inf, 1, 3, 5, SLA_SOFT_DAYS, SLA_HARD_DAYS, np.inf]
labels = ["≤ 1 day", "1–3 days", "3–5 days", "5–10 days", "10–30 days", "&gt; 30 days"]
bands = P.groupby([pd.cut(P.days, edges, labels=labels), "po_type"], observed=False).size().unstack(fill_value=0)
bands = bands / bands.sum()
band_rows = [f"<td>{b}</td>" + "".join(f"<td class='n'>{pct(r[t])}</td><td>{bar(r[t], bands.values.max())}</td>" for t in TYPES)
             for b, r in bands.iterrows()]

# Who posts
post_rows = []
for t in TYPES:
    p = P[P.po_type == t]
    g = p.groupby("poster").agg(n=("days", "size"), median=("days", "median"), late=("late10", "mean"), missed=("missed", "sum"))
    g = g.sort_values("n", ascending=False)
    for i, (u, r) in enumerate(g.head(5).iterrows()):
        tag = ' <span class="pill sys">system</span>' if u in BOT_USERS else ""
        post_rows.append(f"<td>{code(t) if i == 0 else ''}</td><td>{u}{tag}</td><td class='n'>{num(r.n)}</td><td class='n'>{pct(r.n / len(p), 0)}</td>"
                         f"<td class='n'>{num(r['median'], 1)}</td><td class='n'>{pct(r.late, 0)}</td><td class='n'>{mnok(r.missed)}</td>")
    R[t]["top_poster"], R[t]["top_share"], R[t]["n_posters"] = g.index[0], g.n.iloc[0] / len(p), len(g)

# Suppliers
sup_rows = []
for t in TYPES:
    p = P[P.po_type == t]
    g = p.groupby("supplier").agg(n=("days", "size"), late=("late10", "mean"), missed=("missed", "sum")).sort_values("missed", ascending=False)
    R[t]["top5_sup_missed"] = g.missed.head(5).sum() / g.missed.sum()
    for i, (s, r) in enumerate(g.head(5).iterrows()):
        sup_rows.append(f"<td>{code(t) if i == 0 else ''}</td><td>{s}</td><td class='n'>{num(r.n)}</td><td class='n'>{pct(r.late, 0)}</td>"
                        f"<td class='n'>{mnok(r.missed)}</td><td class='n'>{pct(r.missed / R[t]['missed'], 0)}</td>")

# Month of arrival
P["month"] = P.start.dt.to_period("M")
mo = P.groupby(["month", "po_type"]).agg(n=("days", "size"), median=("days", "median"), late=("late10", "mean")).unstack()
mo = mo[(mo["n"] >= 30).all(axis=1)]
mo_rows = [f"<td>{m.strftime('%b %Y')}</td>" + "".join(f"<td class='n'>{num(r[('n', t)])}</td><td class='n'>{num(r[('median', t)], 1)}</td>"
                                                       f"<td class='n'>{pct(r[('late', t)], 0)}</td>" for t in TYPES)
           for m, r in mo.iterrows()]

# Sensitivity: if part of the time was an unseen wait for goods, the clock would start that much later
GR_REF = {}
for t in TYPES:
    w = inv[(inv.po_type == t) & (inv.route == "WAITGR") & inv.posted & inv.goods_receipt.notna()]
    GR_REF[t] = ((w.goods_receipt - w.start).dt.total_seconds() / 86400).quantile([.5, .75])
sens_rows, SENS = [], {t: {} for t in TYPES}
for shift in [0, 1, 3, 5, 7]:
    cells = ""
    for t in TYPES:
        p = P[P.po_type == t]
        late = (p.days - shift) > SLA_SOFT_DAYS
        SENS[t][shift] = late.mean()
        cells += f"<td class='n'>{pct(late.mean())}</td><td class='n'>{mnok((DISCOUNT * p.inv_value * late).sum())}</td>"
    sens_rows.append(f"<td>{'None (as measured)' if shift == 0 else f'{shift} day' + ('s' if shift > 1 else '')}</td>{cells}")

# Smallest assumed goods wait at which ZNB1's queue is no later than the rest of ZNB1
S_EQ = next((s for s, x in SENS["ZNB1"].items() if x <= R["ZNB1"]["s10_other"] * 1.1), max(SENS["ZNB1"]))

# Comparison with the goods-receipt road: median and 90th percentile of each time segment
def mp(x):
    x = x.dropna()
    return f"{x.median():.1f} / {x.quantile(0.9):.1f}" if len(x) else "–"


SEG = {}
for t in TYPES:
    for rt in ["WAITGR", "CTRL1"]:
        a = inv[(inv.po_type == t) & (inv.route == rt) & inv.posted]
        g = a[a.goods_receipt.notna()]
        d = lambda x, y: (y - x).dt.total_seconds() / 86400
        SEG[(t, rt)] = dict(n=len(a), n_gr=len(g), has_gr=len(g) / len(a),
                            arr_gr=d(g.start, g.goods_receipt), gr_post=d(g.goods_receipt, g.first_post),
                            arr_post=d(a.start, a.first_post), clock=a.days, late=a.late10.mean(), late30=a.late30.mean(),
                            late_arr=(d(a.start, a.first_post) > SLA_SOFT_DAYS).mean())
COLS = [(t, rt) for t in TYPES for rt in ["WAITGR", "CTRL1"]]
seg_rows = [f"<td>{lab}</td>" + "".join(f"<td class='n'>{f(SEG[c])}</td>" for c in COLS) for lab, f in [
    ("Posted invoices", lambda s: num(s["n"])),
    ("with a goods-receipt step (WAIT_GR)", lambda s: pct(s["has_gr"], 0)),
    ("Arrival → goods receipt booked", lambda s: mp(s["arr_gr"]) if s["has_gr"] > 0.5 else "not in log"),
    ("Goods receipt booked → posting", lambda s: mp(s["gr_post"]) if s["has_gr"] > 0.5 else "not in log"),
    ("<b>Arrival → posting</b>", lambda s: f"<b>{mp(s['arr_post'])}</b>"),
    ("Share &gt; 10 days, arrival → posting", lambda s: pct(s["late_arr"])),
    ("<b>Invoice clock</b> (goods receipt or arrival → posting)", lambda s: f"<b>{mp(s['clock'])}</b>"),
    ("Share &gt; 10 days, invoice clock", lambda s: pct(s["late"])),
    ("Share &gt; 30 days, invoice clock", lambda s: pct(s["late30"])),
]]
seg_head = ["Days: median / 90th percentile"] + [f"{code(t)} <code>{rt}</code>" for t, rt in COLS]
zg, zc = SEG[("ZNB1", "WAITGR")], SEG[("ZNB1", "CTRL1")]

# What drives the time in the ZNB1 queue: volume vs clerk (human postings, validation -> posting)
Q = P[(P.po_type == "ZNB1") & ~P.poster.isin(BOT_USERS)].copy()
_all = P[~P.poster.isin(BOT_USERS)]
_qi, _qo = np.sort(Q.val_ts.values), np.sort(Q.first_post.values)
Q["backlog"] = np.searchsorted(_qi, Q.val_ts.values, "right") - np.searchsorted(_qo, Q.val_ts.values, "right")
rho_backlog = Q.val_to_post.corr(Q.backlog, method="spearman")
qb = Q.groupby(pd.qcut(Q.backlog, 5, labels=["Smallest fifth", "2nd", "3rd", "4th", "Largest fifth"]), observed=True).agg(
    lo=("backlog", "min"), hi=("backlog", "max"), med=("val_to_post", "median"), p90=("val_to_post", lambda x: x.quantile(.9)),
    late=("days", lambda x: (x > SLA_SOFT_DAYS).mean()))
qb_rows = [f"<td>{k}</td><td class='n'>{r.lo:.0f}–{r.hi:.0f}</td><td class='n'>{r.med:.1f}</td><td class='n'>{r.p90:.1f}</td><td class='n'>{pct(r.late, 0)}</td>"
           for k, r in qb.iterrows()]
wk = Q.groupby(Q.val_ts.dt.to_period("W")).agg(arr=("days", "size"), med=("val_to_post", "median"))
wk = wk[wk.arr >= 20]
rho_week = wk.arr.corr(wk.med, method="spearman")
busy = wk.arr.idxmax()
slow = wk.med.idxmax()
clerk = Q.groupby("poster").agg(n=("days", "size"), med=("val_to_post", "median"), p90=("val_to_post", lambda x: x.quantile(.9)),
                                late=("days", lambda x: (x > SLA_SOFT_DAYS).mean())).sort_values("n", ascending=False).head(5)
clerk_rows = [f"<td>{u}</td><td class='n'>{num(r.n)}</td><td class='n'>{pct(r.n / len(Q), 0)}</td><td class='n'>{r.med:.1f}</td>"
              f"<td class='n'>{r.p90:.1f}</td><td class='n'>{pct(r.late, 0)}</td>" for u, r in clerk.iterrows()]

EXTRA_CSS = """
.bar { background: var(--accent-soft); border-radius: 3px; height: 10px; min-width: 80px; }
.bar span { display: block; height: 100%; background: var(--accent); border-radius: 3px; }
"""

page = f"""<meta charset="utf-8">
<title>Direct to CTRL1</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,500;6..72,600&family=Instrument+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap">
<style>{CSS}{EXTRA_CSS}</style>

<div class="wrap">
<header>
  <div class="eyebrow">Accounts payable · Process mining case · Manual queue</div>
  <h1>Invoices sent straight to the manual queue</h1>
  <p class="lede">ZJD1 and ZNB1 invoices that SAP routes to the manual control queue (<code>CTRL1</code>) at their first <i>Initial validation</i>, skipping the goods-receipt check: how they move, how long they take, and what they cost in missed discount.</p>
  <div class="facts">
    <span><b>{num(r1['n'])}</b> ZJD1 invoices ({pct(r1['share'], 0)} of the type)</span>
    <span><b>{num(r2['n'])}</b> ZNB1 invoices ({pct(r2['share'], 0)})</span>
    <span>limits <b>{SLA_SOFT_DAYS} d</b> and <b>{SLA_HARD_DAYS} d</b></span>
  </div>
</header>

<h2 id="answer">Short answer</h2>
<div class="findings">
  <div class="finding"><div class="k">{pct(r1['s10'])} vs {pct(r2['s10'])}</div><p>of these invoices take longer than 10 days in ZJD1 and ZNB1, against {pct(r1['s10_other'])} and {pct(r2['s10_other'])} of all other invoices of the type.</p></div>
  <div class="finding"><div class="k">{pct(r1['missed'] / r1['missed_type'], 0)} / {pct(r2['missed'] / r2['missed_type'], 0)}</div><p>of the type's missed discount ({mnok(r1['missed'])} and {mnok(r2['missed'])} NOK) comes from these invoices.</p></div>
  <div class="finding"><div class="k">{pct(r1['direct'], 0)} / {pct(r2['direct'], 0)}</div><p>go from validation straight to <i>Posted manually</i>. The lead time is almost all waiting for a clerk.</p></div>
  <div class="finding"><div class="k">Caveat</div><p>Only {pct(r2['has_gr'])} of the ZNB1 invoices have a goods-receipt step, so their clock starts at arrival and any wait for goods is hidden. If each of these invoices had {S_EQ} days of unseen goods wait, ZNB1's queue would be no later than the rest of ZNB1 (<a href="#hidden">section 6</a>).</p></div>
</div>

<h2 id="define"><span class="step">1</span>Which invoices, and how the clock runs</h2>
<ol class="rec">
  <li><b>Selection.</b> One invoice = one <code>PO_ID</code> × <code>SAP_PO_No</code>. An invoice is included if the routing code (<code>SAP_Next_Processing_Type</code>) on its <b>first</b> <i>Initial validation</i> is <code>CTRL1</code>. Invoices that reach the queue later (e.g. after an <code>END_WAIT_GR</code> on the goods-receipt road) are not included.</li>
  <li><b>Clock start.</b> The same rule as in the invoice-clock report: the first <i>{GR_ACTIVITY}</i> event with <code>{GR_OBJECT_CODE}</code>, else the invoice's first event. For these invoices that is almost always the first event, <i>Data management</i> (arrival in AP); validation follows a median {num(r2['arr_to_val_s'])} seconds later (ZNB1). The few with a later <code>{GR_OBJECT_CODE}</code> step ({pct(r1['has_gr'])} in ZJD1, {pct(r2['has_gr'])} in ZNB1) went back through the goods-receipt check after a restart.</li>
  <li><b>Clock stop.</b> The first posting (<i>{"</i>, <i>".join(POSTING_ACTIVITIES)}</i>). Days = stop − start in seconds / 86,400, calendar days.</li>
  <li><b>Base and discount.</b> Only posted invoices count. Late = more than 10 (or 30) days. The missed 2% uses the PO value split evenly over all the PO's posted invoices, the same rule as in the other reports.</li>
</ol>

<h2 id="overview"><span class="step">2</span>ZJD1 vs ZNB1</h2>
{ov_html}
<div class="callout">
<p><b>ZNB1 sends more invoices to the queue and handles them more slowly.</b> {pct(r2['share'], 0)} of ZNB1 invoices go there against {pct(r1['share'], 0)} of ZJD1, and {pct(r2['s10'])} of them are late against {pct(r1['s10'])}. Together they cause {pct(r2['missed'] / r2['missed_type'], 0)} of ZNB1's missed discount.</p>
</div>

<h3>What happens right after validation</h3>
{table(["Next step", "ZJD1", "ZNB1"], nxt_rows)}
<p class="prose">The path is short: in most cases a clerk posts the invoice, and nothing else happens in between. Approval loops, transfers to MM and deletions are the exceptions.</p>

<h3>How long they take</h3>
{table(["Days to posting", "ZJD1", "", "ZNB1", ""], band_rows)}
<p class="prose">ZJD1 spreads out, with {pct(bands.loc['≤ 1 day', 'ZJD1'], 0)} posted within a day. ZNB1 bunches at 5–10 days ({pct(bands.loc['5–10 days', 'ZNB1'], 0)}, median {num(r2['median'], 1)} days), which fits a queue that is cleared about once a week, and {pct(bands.loc['10–30 days', 'ZNB1'], 0)} slip just past the 10-day limit into 10–30 days.</p>

<h2 id="who"><span class="step">3</span>Who posts them</h2>
{table(["Type", "Posted by", "Invoices", "Share", "Median days", "&gt; 10 d", "Missed 2% (NOK)"], post_rows)}
<div class="prose"><p>{pct(r1['human'], 0)} (ZJD1) and {pct(r2['human'], 0)} (ZNB1) of the postings are made by people. In ZNB1, {R['ZNB1']['top_poster']} posts {pct(R['ZNB1']['top_share'], 0)} of these invoices alone; in ZJD1 the busiest clerk, {R['ZJD1']['top_poster']}, posts {pct(R['ZJD1']['top_share'], 0)}. How fast the queue moves depends heavily on a few people (see <code>notes/znb1_why_manual_route_stalls.txt</code> for the ZNB1 queue in detail).</p></div>

<h2 id="suppliers"><span class="step">4</span>Suppliers with the most missed discount</h2>
{table(["Type", "Supplier", "Invoices", "&gt; 10 d", "Missed 2% (NOK)", "Share of type's CTRL1 missed"], sup_rows)}
<p class="prose">The five largest suppliers by missed discount account for {pct(r1['top5_sup_missed'], 0)} (ZJD1) and {pct(r2['top5_sup_missed'], 0)} (ZNB1) of the missed discount on these invoices. Why a supplier's invoices go to <code>CTRL1</code> at all is in <a href="report_znb1_ctrl1.html">Why do ZNB1 invoices go to the manual queue?</a></p>

<h2 id="month"><span class="step">5</span>By month of arrival</h2>
{table(["Month", "ZJD1 invoices", "Median days", "&gt; 10 d", "ZNB1 invoices", "Median days", "&gt; 10 d"], mo_rows)}
<p class="prose">Months with fewer than 30 posted invoices in either type are left out. The slow summer months fit the holiday period; from October the queue moves faster.</p>

<h2 id="compare-gr"><span class="step">6</span>Comparison with the goods-receipt road</h2>
<div class="prose"><p>Each time segment as <b>median / 90th percentile</b> in days, for posted invoices by route at their first validation. <code>WAITGR</code> is the goods-receipt road (goods-receipt check, then usually automatic posting); <code>CTRL1</code> is the manual queue covered by this report.</p></div>
{table(seg_head, seg_rows)}
<div class="callout">
<p><b>Measured from arrival, both ZNB1 roads miss 10 days about equally often</b> ({pct(zg['late_arr'])} vs {pct(zc['late_arr'])}), <b>but in different ways.</b> The goods-receipt road is usually fast with a long tail: {mp(zg['arr_post'])} days. Almost all of that is waiting for the goods receipt ({mp(zg['arr_gr'])} days); once it's booked, posting follows almost at once ({mp(zg['gr_post'])} days). The manual queue is consistently slow with a short tail: {mp(zc['arr_post'])} days, all of it time to posting, as there's no goods-receipt step.</p>
<p><b>With the invoice clock</b> the goods-receipt road drops to {mp(zg['clock'])} days ({pct(zg['late'])} over 10 days), while the manual queue stays at {mp(zc['clock'])} days ({pct(zc['late'])}).</p>
</div>

<h2 id="drivers"><span class="step">7</span>What drives the time in the ZNB1 queue?</h2>
<div class="prose"><p>Time from validation (entering <code>CTRL1</code>) to posting, for the {num(len(Q))} ZNB1 queue invoices posted by a person. If the number of invoices in the queue drove the wait, invoices entering a long queue would wait longer.</p></div>
{table(["Open ZNB1 queue items when the invoice entered", "Range", "Median days", "90th pct", "&gt; 10 d"], qb_rows)}
<div class="prose">
<p><b>Queue size doesn't explain the wait.</b> The rank correlation between queue size at entry and the wait is {rho_backlog:.2f}. Week by week, the number of new queue invoices and the median wait correlate at {rho_week:.2f}. The busiest week ({busy.start_time:%d %b %Y}, {num(wk.loc[busy, 'arr'])} invoices) had a median wait of {wk.loc[busy, 'med']:.1f} days; the slowest week ({slow.start_time:%d %b %Y}) had {num(wk.loc[slow, 'arr'])} invoices and a median of {wk.loc[slow, 'med']:.1f} days.</p>
<p><b>Who posts it does:</b></p>
</div>
{table(["Posted by", "Invoices", "Share", "Median days", "90th pct", "&gt; 10 d"], clerk_rows)}
<p class="prose">The wait is about the same whichever weekday the invoice arrives, which fits a queue worked through on a roughly weekly rhythm, mostly by one person, rather than a queue that grows with volume. Staff availability (summer holidays, see section 5) matters more than the number of invoices.</p>

<h2 id="hidden"><span class="step">8</span>Assumption: invoices come after delivery</h2>
<div class="prose">
<p><b>Assumption used:</b> suppliers send the invoice after delivering the goods. Then a manual-queue invoice has its goods from the start, the clock from arrival is fair, and its whole lead time is accounts-payable time.</p>
<p><b>What speaks against it:</b> on the goods-receipt road the goods receipt is booked a median {num(GR_REF['ZNB1'][.5], 1)} days after the invoice arrives in ZNB1 (90th percentile {num(zg['arr_gr'].quantile(.9), 1)}), and {num(GR_REF['ZJD1'][.5], 1)} days in ZJD1 (90th percentile {num(SEG[('ZJD1', 'WAITGR')]['arr_gr'].quantile(.9), 1)}). So either some invoices do arrive before the goods, or the warehouse books the receipt late. If clerks waited for that too, part of the queue time would be goods wait. Check: how the figures change if each invoice's clock started a fixed number of days later.</p>
</div>
{table(["Assumed unseen goods wait", "ZJD1 &gt; 10 d", "ZJD1 missed", "ZNB1 &gt; 10 d", "ZNB1 missed"], sens_rows)}
<div class="callout"><p><b>Under the assumption, the measured figures stand</b> ({pct(r2['s10'])} of ZNB1 queue invoices over 10 days). The check shows why the assumption matters: most ZNB1 queue invoices are posted 5–10 days after arrival ({pct(bands.loc['5–10 days', 'ZNB1'], 0)}), so {S_EQ} days of goods wait would bring the late share ({pct(SENS['ZNB1'][S_EQ])}) down to the level of all other ZNB1 invoices ({pct(r2['s10_other'])}). Whether clerks hold invoices for missing goods should be confirmed with the AP team.</p></div>

<footer>
<p>Built with Python (conda env <code>datasci</code>) by <code>11_report_ctrl1_direct.py</code> from <code>output/events.pkl</code>, <code>invoices.csv</code> and <code>cases.csv</code>. Clock rules: <a href="report_invoice_clock.html">The invoice clock</a> and <code>docs/assumptions.html</code>.</p>
</footer>
</div>
"""
(OUT / "report_ctrl1_direct.html").write_text(page, encoding="utf-8")
print("wrote", OUT / "report_ctrl1_direct.html", len(page) // 1024, "KB")
for t in TYPES:
    print(t, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in R[t].items()})
print(bands.round(3)); print(mo.round(2)); print(GR_REF)
