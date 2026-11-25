"""Builds output/report_znb1_end_wait_gr.html: ZNB1 invoices that reached the manual queue (CTRL1) via END_WAIT_GR,
with the clock running from END_WAIT_GR to the first posting. Comparison group: ZNB1 invoices sent straight to CTRL1 at
their first validation (clock from arrival).

A focused side analysis: the pipeline clock (GR_CLOCK_PO_TYPES) is not changed. Reads output/events.pkl, invoices.csv and
cases.csv; reuses the stylesheet of 05_report.py.
"""
import html
import re

import numpy as np
import pandas as pd

from common import OUT, ROOT, BOT_USERS, POSTING_ACTIVITIES, SLA_SOFT_DAYS, SLA_HARD_DAYS, GR_ACTIVITY, GR_OBJECT_CODE, \
    load_events, load_invoices, load_cases

T = "ZNB1"
KEY = ["PO_ID", "SAP_PO_No"]
END = "END_WAIT_GR"
DISCOUNT = 0.02
CSS = re.search(r"<style>(.*?)</style>", (ROOT / "05_report.py").read_text(), re.S).group(1)
CSS = CSS.replace("{{", "{").replace("}}", "}")


def pct(x, d=1):
    return "–" if pd.isna(x) else f"{x * 100:.{d}f}%"


def num(x, d=0):
    return "–" if pd.isna(x) else f"{x:,.{d}f}"


def mnok(x):
    return f"{x / 1e6:,.2f}M"


def mp(x):
    x = pd.Series(x).dropna()
    return f"{x.median():.1f} / {x.quantile(0.9):.1f}" if len(x) else "–"


def code(c):
    return f'<span class="code">{c}</span>'


def table(head, rows, cls=""):
    th = "".join(f"<th>{h}</th>" for h in head)
    body = "".join(f"<tr>{r}</tr>" for r in rows)
    return f'<div class="tbl {cls}"><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def bar(x, top=1.0):
    return f'<div class="bar"><span style="width:{min(x / top, 1) * 100:.1f}%"></span></div>'


def days(a, b):
    return (b - a).dt.total_seconds() / 86400


# ---------------------------------------------------------------- data
ev = load_events()
ev = ev[ev.SAP_PO_Type_Code == T].sort_values(["PO_ID", "ts", "Event_ID"])
inv = load_invoices()
inv = inv[inv.po_type == T].set_index(KEY)
cases = load_cases().set_index("PO_ID")

is_end = (ev.Activity_ID == GR_ACTIVITY) & (ev.SAP_Object_Code == END)
inv["end_ts"] = ev[is_end].groupby(KEY).ts.first()
inv["n_end"] = ev[is_end].groupby(KEY).size().reindex(inv.index).fillna(0).astype(int)
val = ev[ev.Activity_ID == "Initial validation"].groupby(KEY)
inv["route"], inv["val_ts"] = val.SAP_Next_Processing_Type.first(), val.ts.first()
post = ev[ev.Activity_ID.isin(POSTING_ACTIVITIES)].groupby(KEY)
inv["post_act"], inv["poster"] = post.Activity_ID.first(), post.User_ID.first()
inv["supplier"] = ev.groupby(KEY).Supplier_ID.first()
# Missed discount: PO value split evenly over the PO's posted invoices (same rule as the pipeline)
inv["n_posted"] = inv.groupby(level="PO_ID").posted.transform("sum")
inv["inv_value"] = np.where(inv.posted, cases.invoice_value_nok.reindex(inv.index.get_level_values(0)).values
                            / inv.n_posted.where(inv.n_posted > 0), 0)

# Selection funnel
f_end = inv[inv.end_ts.notna()]
f_nogr = f_end[f_end.goods_receipt.isna()]
E = f_nogr[f_nogr.posted].copy()                 # the invoices of this report
E["clock"] = days(E.end_ts, E.first_post)
E["arr_end"] = days(E.start, E.end_ts)
E["arr_post"] = days(E.start, E.first_post)
D = inv[(inv.route == "CTRL1") & inv.posted].copy()  # comparison: straight to CTRL1 at validation
D["clock"] = days(D.start, D.first_post)
for X in (E, D):
    X["late10"], X["late30"] = X.clock > SLA_SOFT_DAYS, X.clock > SLA_HARD_DAYS
    X["missed"] = DISCOUNT * X.inv_value * X.late10
E["late10_arr"] = E.arr_post > SLA_SOFT_DAYS
E["late30_arr"] = E.arr_post > SLA_HARD_DAYS
E["missed_arr"] = DISCOUNT * E.inv_value * E.late10_arr

funnel = [
    ("ZNB1 invoices", len(inv)),
    (f"with an <code>{END}</code> event", len(f_end)),
    (f"… of which also a <code>{GR_OBJECT_CODE}</code> event (goods-receipt clock applies, left out)", len(f_end) - len(f_nogr)),
    (f"… no <code>{GR_OBJECT_CODE}</code>", len(f_nogr)),
    ("… never posted (left out)", len(f_nogr) - len(E)),
    ("<b>Posted: the invoices of this report</b>", len(E)),
]
funnel_rows = [f"<td>{lab}</td><td class='n'>{num(n)}</td>" for lab, n in funnel]
route_mix = E.route.value_counts()
n_multi = int((E.n_end > 1).sum())


def summary(X, col="clock", late10="late10", late30="late30", missed="missed"):
    return dict(n=len(X), mp=mp(X[col]), s10=X[late10].mean(), n10=int(X[late10].sum()), s30=X[late30].mean(),
                n30=int(X[late30].sum()), missed=X[missed].sum(), within1=(X[col] < 1).mean())


SE, SA, SD = summary(E), summary(E, "arr_post", "late10_arr", "late30_arr", "missed_arr"), summary(D)
cmp_rows = [f"<td>{lab}</td><td class='n'>{f(SE)}</td><td class='n'>{f(SA)}</td><td class='n'>{f(SD)}</td>" for lab, f in [
    ("Posted invoices", lambda s: num(s["n"])),
    ("<b>Days: median / 90th percentile</b>", lambda s: f"<b>{s['mp']}</b>"),
    ("Posted within 1 day", lambda s: pct(s["within1"], 0)),
    ("<b>&gt; 10 days</b>", lambda s: f"<b>{pct(s['s10'])}</b> ({num(s['n10'])})"),
    ("<b>&gt; 30 days</b>", lambda s: f"<b>{pct(s['s30'])}</b> ({num(s['n30'])})"),
    ("Missed 2% discount (NOK)", lambda s: mnok(s["missed"])),
]]
cmp_head = ["", f"<code>{END}</code> → posting<br><small>this report</small>", f"Same invoices, arrival → posting<br><small>current rule</small>",
            "Straight to CTRL1, arrival → posting<br><small>comparison</small>"]

# Segments of the END_WAIT_GR invoices
seg_rows = [f"<td>{lab}</td><td class='n'>{mp(x)}</td>" for lab, x in [
    ("Arrival → validation", days(E.start, E.val_ts)),
    (f"Validation → <code>{END}</code> (in the goods-receipt check)", days(E.val_ts, E.end_ts)),
    (f"<code>{END}</code> → posting", E.clock),
    ("Arrival → posting", E.arr_post),
]]

# Lead-time bands, both groups
edges = [-np.inf, 1, 3, 5, SLA_SOFT_DAYS, SLA_HARD_DAYS, np.inf]
labels = ["≤ 1 day", "1–3 days", "3–5 days", "5–10 days", "10–30 days", "&gt; 30 days"]
bands = pd.DataFrame({"E": pd.cut(E.clock, edges, labels=labels).value_counts(normalize=True),
                      "D": pd.cut(D.clock, edges, labels=labels).value_counts(normalize=True)}).reindex(labels)
band_rows = [f"<td>{b}</td><td class='n'>{pct(r.E)}</td><td>{bar(r.E, bands.values.max())}</td>"
             f"<td class='n'>{pct(r.D)}</td><td>{bar(r.D, bands.values.max())}</td>" for b, r in bands.iterrows()]

# What happens between END_WAIT_GR and posting
ab = {"Data management": "Data management", "Initial validation": "Initial validation", "Transfer to MM": "Transfer to MM",
      "Sent for approval": "Sent for approval", "Approved MM (accept if price diff)": "Approved (price diff)",
      "Approved MM": "Approved MM", "Approval rejected": "Approval rejected", "Posting in progress": "Posting in progress",
      "Restarted process": "Restarted"}
ee = ev.set_index(KEY).loc[E.index].reset_index().merge(E[["end_ts", "first_post"]].reset_index(), on=KEY)
mid = ee[(ee.ts > ee.end_ts) & (ee.ts < ee.first_post) & ~((ee.Activity_ID == GR_ACTIVITY) & (ee.SAP_Object_Code == END))]
E["between"] = mid.groupby(KEY).Activity_ID.agg(lambda s: " → ".join(ab.get(a, a) for a in s)).reindex(E.index).fillna("")
E["path"] = np.where(E.between == "", "(nothing) → " + E.post_act, E.between + " → " + E.post_act)
pth = E.groupby("path").agg(n=("clock", "size"), med=("clock", "median"), p90=("clock", lambda x: x.quantile(.9)),
                            late=("late10", "sum"), missed=("missed", "sum")).sort_values("n", ascending=False)
top_p = pth.head(7)
other = pth.iloc[7:]
path_rows = [f"<td>{html.escape(p)}</td><td class='n'>{num(r.n)}</td><td class='n'>{pct(r.n / len(E), 0)}</td><td class='n'>{r.med:.1f} / {r.p90:.1f}</td>"
             f"<td class='n'>{num(r.late)}</td><td class='n'>{mnok(r.missed)}</td>" for p, r in top_p.iterrows()]
if len(other):
    path_rows.append(f"<td><i>{len(other)} other paths</i></td><td class='n'>{num(other.n.sum())}</td><td class='n'>{pct(other.n.sum() / len(E), 0)}</td>"
                     f"<td class='n'>–</td><td class='n'>{num(other.late.sum())}</td><td class='n'>{mnok(other.missed.sum())}</td>")
direct_post = (E.between == "")
late_with_steps = E[E.late10 & ~direct_post]

# Who posts
pst = E.groupby("poster").agg(n=("clock", "size"), med=("clock", "median"), p90=("clock", lambda x: x.quantile(.9)),
                              late=("late10", "mean")).sort_values("n", ascending=False)
SYS = ' <span class="pill sys">system</span>'
post_rows = [f"<td>{u}{SYS if u in BOT_USERS else ''}</td><td class='n'>{num(r.n)}</td>"
             f"<td class='n'>{pct(r.n / len(E), 0)}</td><td class='n'>{r.med:.1f} / {r.p90:.1f}</td><td class='n'>{pct(r.late, 0)}</td>"
             for u, r in pst.head(6).iterrows()]
human = (~E.poster.isin(BOT_USERS)).mean()

# Suppliers
sup = E.groupby("supplier").agg(n=("clock", "size"), arr_end=("arr_end", "median"), med=("clock", "median"),
                                late=("late10", "sum"), late_arr=("late10_arr", "sum")).sort_values("n", ascending=False)
sup_rows = [f"<td>{s}</td><td class='n'>{num(r.n)}</td><td class='n'>{r.arr_end:.1f}</td><td class='n'>{r.med:.1f}</td>"
            f"<td class='n'>{num(r.late)}</td><td class='n'>{num(r.late_arr)}</td>" for s, r in sup.head(8).iterrows()]
top8_share = sup.n.head(8).sum() / len(E)

# Month of END_WAIT_GR
mo = E.groupby(E.end_ts.dt.to_period("M")).agg(n=("clock", "size"), med_wait=("arr_end", "median"), med=("clock", "median"),
                                                 late=("late10", "mean"))
mo_rows = [f"<td>{m.strftime('%b %Y')}</td><td class='n'>{num(r.n)}</td><td class='n'>{r.med_wait:.1f}</td><td class='n'>{r.med:.1f}</td>"
           f"<td class='n'>{pct(r.late, 0)}</td>" for m, r in mo.iterrows()]

# What the END_WAIT_GR event looks like
endev = ev[is_end]
end_bot = endev.User_ID.isin(BOT_USERS).mean()
end_users = ", ".join(endev.User_ID.value_counts().index[:2])
wd = endev.ts.dt.dayofweek.value_counts(normalize=True).reindex(range(7), fill_value=0)
hr = endev.ts.dt.hour
office = hr.between(7, 17).mean()
peak_hour = hr.value_counts().idxmax()
next_code = endev.SAP_Next_Processing_Type.value_counts(normalize=True)

# Effect on the ZNB1 headline if this rule were applied (side calculation, pipeline unchanged)
allp = inv[inv.posted].copy()
allp["new"] = np.where(allp.goods_receipt.isna() & allp.end_ts.notna(), days(allp.end_ts, allp.first_post), allp.time_to_post_days)
cur10, new10 = (allp.time_to_post_days > SLA_SOFT_DAYS), (allp.new > SLA_SOFT_DAYS)
cur30, new30 = (allp.time_to_post_days > SLA_HARD_DAYS), (allp.new > SLA_HARD_DAYS)
head_rows = [
    f"<td>Current rule (goods receipt, else arrival)</td><td class='n'>{num(cur10.sum())} ({pct(cur10.mean())})</td><td class='n'>{num(cur30.sum())} ({pct(cur30.mean())})</td><td class='n'>{mnok(DISCOUNT * (allp.inv_value * cur10).sum())}</td>",
    f"<td>With <code>{END}</code> as start for these invoices</td><td class='n'>{num(new10.sum())} ({pct(new10.mean())})</td><td class='n'>{num(new30.sum())} ({pct(new30.mean())})</td><td class='n'>{mnok(DISCOUNT * (allp.inv_value * new10).sum())}</td>",
]

EXTRA_CSS = """
.bar { background: var(--accent-soft); border-radius: 3px; height: 10px; min-width: 80px; }
.bar span { display: block; height: 100%; background: var(--accent); border-radius: 3px; }
"""

page = f"""<meta charset="utf-8">
<title>ZNB1 END_WAIT_GR Clock</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,500;6..72,600&family=Instrument+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap">
<style>{CSS}{EXTRA_CSS}</style>

<div class="wrap">
<header>
  <div class="eyebrow">Accounts payable · Process mining case · ZNB1 manual queue</div>
  <h1>ZNB1: from <code>{END}</code> to posting</h1>
  <p class="lede">{code(T)} invoices that waited in SAP's goods-receipt check until the wait ended <i>without</i> a goods receipt (<code>{END}</code>), which sends them to the manual control queue (<code>CTRL1</code>). Here their clock runs from <code>{END}</code> to the first posting, and they are compared with the invoices sent straight to <code>CTRL1</code> at validation.</p>
  <div class="facts">
    <span><b>{num(len(E))}</b> invoices</span>
    <span><b>{num(E.index.get_level_values(0).nunique())}</b> POs</span>
    <span><b>{num(E.supplier.nunique())}</b> suppliers</span>
    <span>comparison: <b>{num(len(D))}</b> straight to CTRL1</span>
  </div>
</header>

<h2 id="answer">Short answer</h2>
<div class="findings">
  <div class="finding"><div class="k">{SE['mp']} days</div><p>median / 90th percentile from <code>{END}</code> to posting. {pct(SE['within1'], 0)} are posted within a day.</p></div>
  <div class="finding"><div class="k">{pct(SE['s10'])} / {pct(SE['s30'])}</div><p>take longer than 10 / 30 days, against {pct(SD['s10'])} / {pct(SD['s30'])} of the invoices sent straight to <code>CTRL1</code>.</p></div>
  <div class="finding"><div class="k">{pct(SA['s10'])} → {pct(SE['s10'])}</div><p>Counted from arrival, as the current rule does, most of them look late. Almost all of that time was spent in the goods-receipt check ({mp(E.arr_end)} days), not in accounts payable.</p></div>
  <div class="finding"><div class="k">{num(len(late_with_steps))} of {num(SE['n10'])}</div><p>still-late invoices had extra steps after <code>{END}</code>: a restart, <i>Transfer to MM</i> or a price-difference approval.</p></div>
</div>

<h2 id="select"><span class="step">1</span>Which invoices, and how the clock runs</h2>
{table(["Selection", "Invoices"], funnel_rows)}
<ol class="rec">
  <li><b>One invoice</b> = one <code>PO_ID</code> × <code>SAP_PO_No</code>; events ordered by timestamp, then <code>Event_ID</code>.</li>
  <li><b>Included:</b> ZNB1 invoices with a <i>{GR_ACTIVITY}</i> event whose <code>SAP_Object_Code</code> is <code>{END}</code> and no <code>{GR_OBJECT_CODE}</code> event, that were posted. Every <code>{END}</code> event sets the routing code to <code>{next_code.index[0]}</code> ({pct(next_code.iloc[0], 0)}), so these invoices all end up in the manual queue. By route at validation: {", ".join(f"<code>{k}</code> {num(v)}" for k, v in route_mix.items())}.</li>
  <li><b>Clock start:</b> the invoice's first <code>{END}</code> event, the moment the goods-receipt wait ended and accounts payable could act. {num(n_multi)} invoices have more than one; the first one counts.</li>
  <li><b>Clock stop:</b> the first posting (<i>{"</i>, <i>".join(POSTING_ACTIVITIES)}</i>). Days = stop − start in seconds / 86,400, calendar days.</li>
  <li><b>Late</b> = more than 10 (or 30) days. Missed discount = 2% of the PO value split evenly over the PO's posted invoices, for each late invoice.</li>
  <li><b>Comparison group:</b> ZNB1 invoices routed to <code>CTRL1</code> at their first validation, posted, clock from arrival (<i>Data management</i>) to first posting. That report (<a href="report_ctrl1_direct.html">Invoices sent straight to the manual queue</a>) uses the invoice clock, which starts at goods receipt for the 1% that later get one, so its share over 10 days is slightly lower. The {num(route_mix.get('CTRL1', 0))} <code>{END}</code> invoices that were sent to <code>CTRL1</code> at validation and only later went through the goods-receipt check are in both groups.</li>
</ol>

<h2 id="results"><span class="step">2</span>Results</h2>
{table(cmp_head, cmp_rows)}
<div class="callout">
<p><b>From <code>{END}</code>, these invoices are handled faster than the direct <code>CTRL1</code> ones:</b> median / 90th percentile {SE['mp']} vs {SD['mp']} days, {pct(SE['s10'])} vs {pct(SD['s10'])} over 10 days. Once the wait ends, a clerk usually posts the invoice the same day.</p>
<p><b>Counted from arrival, the same invoices look like the worst group</b> ({pct(SA['s10'])} over 10 days, {pct(SA['s30'])} over 30), because the clock then includes the weeks in the goods-receipt check.</p>
</div>
<h3>Where the time goes (median / 90th percentile, days)</h3>
{table(["Segment", "Days"], seg_rows)}
<h3>Days to posting, both groups</h3>
{table(["Days to posting", f"From <code>{END}</code>", "", "Straight to CTRL1, from arrival", ""], band_rows)}

<h2 id="after"><span class="step">3</span>What happens after <code>{END}</code></h2>
{table(["Steps between END_WAIT_GR and posting → posting", "Invoices", "Share", "Days: median / p90", "&gt; 10 d", "Missed 2% (NOK)"], path_rows)}
<p class="prose">{pct(direct_post.mean(), 0)} go straight to posting. The rest pass through a restart (<i>Data management</i> → <i>Initial validation</i> again), <i>Transfer to MM</i> (purchasing has to correct the PO) or a price-difference approval. These detours cause {num(len(late_with_steps))} of the {num(SE['n10'])} invoices that are still late from <code>{END}</code>.</p>

<h2 id="who"><span class="step">4</span>Who posts them</h2>
{table(["Posted by", "Invoices", "Share", "Days: median / p90", "&gt; 10 d"], post_rows)}
<p class="prose">{pct(human, 0)} are posted by people. The work is spread over more clerks than the direct <code>CTRL1</code> queue, where one clerk posts about half of the ZNB1 invoices.</p>

<h2 id="suppliers"><span class="step">5</span>Suppliers</h2>
{table(["Supplier", "Invoices", f"Arrival → {END}, median days", f"{END} → posting, median days", "&gt; 10 d from END_WAIT_GR", "&gt; 10 d from arrival"], sup_rows)}
<p class="prose">The eight largest suppliers account for {pct(top8_share, 0)} of these invoices. The long part is the wait before <code>{END}</code>, so for these suppliers the delay sits with the goods receipt (delivery or booking), not with accounts payable.</p>

<h2 id="month"><span class="step">6</span>By month of <code>{END}</code></h2>
{table(["Month", "Invoices", f"Arrival → {END}, median days", f"{END} → posting, median days", "&gt; 10 d"], mo_rows)}

<h2 id="event"><span class="step">7</span>What the <code>{END}</code> event is</h2>
<div class="prose">
<p>All {num(len(endev))} <code>{END}</code> events in ZNB1 are logged by {end_users} ({pct(end_bot, 0)} by a system user). They fall on weekdays only ({pct(wd.loc[[5, 6]].sum(), 0)} at weekends), {pct(office, 0)} between 07:00 and 17:59, with the most at {peak_hour:02d}:00. That fits either a scheduled job or a clerk releasing the wait in the invoice cockpit under the system's name; the log can't tell which.</p>
<p>Either way, until <code>{END}</code> the invoice was held in SAP's goods-receipt check, not in the manual queue. Starting the clock there treats these invoices the same way as the ones with a <code>{GR_OBJECT_CODE}</code> event: the wait for the goods doesn't count against accounts payable.</p>
</div>

<h2 id="effect"><span class="step">8</span>Effect on the ZNB1 headline</h2>
<div class="prose"><p>Side calculation, not applied in the pipeline: all posted ZNB1 invoices, with only the clock start of these {num(len(E))} invoices changed.</p></div>
{table(["ZNB1 clock rule", "&gt; 10 days", "&gt; 30 days", "Missed 2% discount (NOK)"], head_rows)}

<h2 id="limits"><span class="step">9</span>Limits</h2>
<ul class="tight prose">
  <li><code>{END}</code> is not a goods-receipt date. It marks when the invoice left the goods-receipt check; whether the goods had arrived by then is unknown.</li>
  <li>The comparison group's clock starts at arrival. Under the assumption that suppliers invoice after delivery, that's fair; otherwise it includes some goods wait (see the direct-<code>CTRL1</code> report).</li>
  <li>Small group: {num(len(E))} invoices, so supplier and month figures rest on few cases.</li>
  <li>Invoice values are estimated by splitting the PO value evenly over its posted invoices.</li>
</ul>

<footer>
<p>Built with Python (conda env <code>datasci</code>) by <code>12_report_end_wait_gr.py</code> from <code>output/events.pkl</code>, <code>invoices.csv</code> and <code>cases.csv</code>. Related: <a href="report_ctrl1_direct.html">Invoices sent straight to the manual queue</a>, <a href="report_invoice_clock.html">The invoice clock</a>.</p>
</footer>
</div>
"""
(OUT / "report_znb1_end_wait_gr.html").write_text(page, encoding="utf-8")
print("wrote", OUT / "report_znb1_end_wait_gr.html", len(page) // 1024, "KB")
print("E", SE, "\nA", SA, "\nD", SD)
print(pth.head(8).round(2)); print(pst.head(6).round(2)); print(mo.round(2))
print("route mix", route_mix.to_dict(), "| late with steps", len(late_with_steps), "| office", round(office, 3), "peak", peak_hour, "| next code", next_code.round(3).to_dict())
