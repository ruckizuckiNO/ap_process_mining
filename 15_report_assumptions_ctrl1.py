"""Builds output/report_assumptions_ctrl1.html: the three calculation assumptions behind the missed-discount figures,
documented and tested for the ZNB1 invoices routed to the manual queue (CTRL1) and posted manually.

  1. Clock: start = arrival in AP (first event), or the first WAIT_GR event where there is one; stop = first posting
  2. Invoice amount: the PO value split evenly over the PO's posted invoices
  3. Value field: Case_Invoice_Value_NOK (PO amount at the rate on receipt), not Case_Clearing_value_NOK

Scope: posted ZNB1 invoices whose first Initial validation routed them to CTRL1 and whose first posting is
"Posted manually". Deleted, never-posted and automatically posted invoices are excluded (counted in section 0).
Each assumption gets its evidence and a sensitivity: how much the missed 2% discount changes under the alternatives.
Writes output/assumptions_ctrl1_results.csv. Charts are inline SVG.
Run from the repo root:  conda run -n datasci python -W ignore 15_report_assumptions_ctrl1.py
"""
import html
import re

import numpy as np
import pandas as pd

from common import GR_CLOCK_PO_TYPES, OUT, POSTING_ACTIVITIES, ROOT, SLA_HARD_DAYS, SLA_SOFT_DAYS, load_events, load_invoices

CSS = re.search(r"<style>(.*?)</style>", (ROOT / "05_report.py").read_text(), re.S).group(1)
CSS = CSS.replace("{{", "{").replace("}}", "}")
K = ["PO_ID", "SAP_PO_No"]
DISCOUNT = 0.02
SHIFTS = [-1, 0, 1, 2, 3, 5]  # days the clock start moves earlier (+) or later (-)


def pct(x, d=0):
    return "–" if pd.isna(x) else f"{x * 100:.{d}f}%"


def num(x, d=0):
    return "–" if pd.isna(x) else f"{x:,.{d}f}"


def mnok(x, d=2):
    return f"{x / 1e6:,.{d}f}M"


def code(c):
    return f'<span class="code">{c}</span>'


def table(head, rows):
    th = "".join(f"<th>{h}</th>" for h in head)
    body = "".join(f"<tr>{r}</tr>" for r in rows)
    return f'<div class="tbl"><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


# ---------------------------------------------------------------- data
ev = load_events()
z = ev[ev.SAP_PO_Type_Code == "ZNB1"].sort_values(["PO_ID", "ts", "Event_ID"])
val = z[z.Activity_ID == "Initial validation"].groupby(K).agg(route=("SAP_Next_Processing_Type", "first"), val_ts=("ts", "first"))
fp = z[z.Activity_ID.isin(POSTING_ACTIVITIES)].groupby(K).Activity_ID.first().rename("post_act")
last_act = z.groupby(K).Activity_ID.last().rename("last_act")
first_act = z.groupby(K).Activity_ID.first().rename("first_act")
vals = z.groupby(K).agg(po_value=("Case_Invoice_Value_NOK", "first"), clear_value=("Case_Clearing_value_NOK", "first"))
const_inv = (z.groupby("PO_ID").Case_Invoice_Value_NOK.nunique() <= 1).mean()
const_clear = (z.groupby("PO_ID").Case_Clearing_value_NOK.nunique() <= 1).mean()

inv = load_invoices()
inv = inv[inv.po_type == "ZNB1"].set_index(K).join(val).join(fp).join(last_act).join(first_act).join(vals)
inv["n_posted"] = inv.groupby(level="PO_ID").posted.transform("sum")
inv["n_all"] = inv.groupby(level="PO_ID").posted.transform("size")

# Funnel: CTRL1 invoices -> scope
c = inv[inv.route == "CTRL1"]
never = c[~c.posted]
funnel = [
    ("ZNB1 invoices (PO_ID × SAP_PO_No)", len(inv)),
    ("… routed to CTRL1 at their first Initial validation", len(c)),
    ("&nbsp;&nbsp;− never posted", len(never)),
    ("&nbsp;&nbsp;− posted automatically after the manual check (MM Autopost after manual process)", int((c.post_act == "MM Autopost after manual process").sum())),
    ("&nbsp;&nbsp;− posted automatically (MM Posted Automatically)", int((c.post_act == "MM Posted Automatically").sum())),
]
never_why = never.last_act.value_counts()
S = c[c.posted & (c.post_act == "Posted manually")].copy()
funnel.append(("<b>Scope: posted manually</b>", len(S)))
n_pos = S.index.get_level_values("PO_ID").nunique()

# Amount rules
S["amount"] = S.po_value / S.n_posted                        # used: PO value / posted invoices
S["amount_all"] = S.po_value / S.n_all                       # alt: / all invoices incl. never posted
S["amount_clear"] = S.clear_value / S.n_posted               # alt: clearing value / posted invoices
S["days"] = S.time_to_post_days                              # pipeline clock
S["days_arr"] = (S.first_post - S.start).dt.total_seconds() / 86400
S["late"] = S.days > SLA_SOFT_DAYS
base = DISCOUNT * (S.amount * S.late).sum()
n_late = int(S.late.sum())


def missed(amount_col="amount", days=None):
    d = S.days if days is None else days
    return DISCOUNT * (S[amount_col] * (d > SLA_SOFT_DAYS)).sum()


# ---------------------------------------------------------------- 1. clock
a2v_s = (S.val_ts - S.start).dt.total_seconds()
n_gr = int(S.goods_receipt.notna().sum())
gr_late_arr, gr_late = int((S.days_arr[S.goods_receipt.notna()] > SLA_SOFT_DAYS).sum()), int(S.late[S.goods_receipt.notna()].sum())
missed_arr_only = missed(days=S.days_arr)
shift_rows, shift_res = [], []
for sft in SHIFTS:
    d = S.days + sft
    mm, nl = missed(days=d), int((d > SLA_SOFT_DAYS).sum())
    shift_res.append(dict(shift=sft, late=nl, missed=mm))
    lab = "Used: as in the log" if sft == 0 else (f"{sft} day{'s' if abs(sft) > 1 else ''} earlier" if sft > 0 else f"{-sft} day later")
    shift_rows.append(f"<td>{'<b>' if sft == 0 else ''}{lab}{'</b>' if sft == 0 else ''}</td><td class='n'>{num(nl)} ({pct(nl / len(S))})</td>"
                      f"<td class='n'>{mnok(mm)}</td><td class='n'>{'–' if sft == 0 else f'{(mm - base) / 1e6:+,.2f}M'}</td>")
# How many invoices sit just below the limit (8-10 days): they flip with a small shift
near = int(((S.days > SLA_SOFT_DAYS - 2) & (S.days <= SLA_SOFT_DAYS)).sum())

# ---------------------------------------------------------------- 2. split
po = S.reset_index().groupby("PO_ID").agg(n_posted=("n_posted", "first"), n_all=("n_all", "first"), in_scope=("SAP_PO_No", "size"),
                                         po_value=("po_value", "first"), late=("late", "sum"))
single = po.n_posted == 1
n_single_po, n_single_inv = int(single.sum()), int(S.n_posted.eq(1).sum())
val_single = S.amount[S.n_posted == 1].sum() / S.amount.sum()
bands = pd.cut(po.n_posted, [0, 1, 2, 5, 10, 20, 1000], labels=["1", "2", "3–5", "6–10", "11–20", "> 20"])
band_tab = po.groupby(bands, observed=True).agg(pos=("n_posted", "size"), invoices=("in_scope", "sum"))
inv_by_band = S.groupby(pd.cut(S.n_posted, [0, 1, 2, 5, 10, 20, 1000], labels=list(band_tab.index)), observed=True).agg(
    amount=("amount", "sum"), missed=("late", lambda x: DISCOUNT * (S.loc[x.index, "amount"] * x).sum()))
band_tab = band_tab.join(inv_by_band)
band_rows = [f"<td>{b}</td><td class='n'>{num(r.pos)}</td><td class='n'>{num(r.invoices)}</td><td class='n'>{mnok(r.amount, 0)}</td>"
             f"<td class='n'>{mnok(r.missed)}</td><td class='n'>{pct(r.missed / base)}</td>" for b, r in band_tab.iterrows()]
partly_out = int((po.n_posted > po.in_scope).sum())  # POs whose other posted invoices are outside the scope
# Alternatives
alt_all = missed("amount_all")
alt_full = DISCOUNT * (S.po_value * S.late).sum()                      # whole PO value on every invoice
po_any_late = DISCOUNT * po.po_value[po.late > 0].sum()                 # PO value once, if any invoice late
split_rows = [
    f"<td><b>Used: PO value ÷ posted invoices</b></td><td>Every posted invoice carries an equal share; the shares add up to the PO value.</td><td class='n'><b>{mnok(base)}</b></td><td class='n'>–</td>",
    f"<td>PO value ÷ all invoices (incl. never posted)</td><td>Never-posted invoices keep a share that is never paid.</td><td class='n'>{mnok(alt_all)}</td><td class='n'>{(alt_all - base) / 1e6:+,.2f}M</td>",
    f"<td>PO value counted once per PO if any invoice is late</td><td>The whole PO loses its discount when one invoice is late.</td><td class='n'>{mnok(po_any_late)}</td><td class='n'>{(po_any_late - base) / 1e6:+,.2f}M</td>",
    f"<td>Full PO value on every invoice</td><td>Counts the PO value once per invoice; adds up to far more than was bought.</td><td class='n'>{mnok(alt_full)}</td><td class='n'>{(alt_full - base) / 1e6:+,.2f}M</td>",
]

# ---------------------------------------------------------------- 3. value field
po_c = S.reset_index().drop_duplicates("PO_ID")
ratio = po_c.clear_value / po_c.po_value
r_q = ratio.quantile([0.05, 0.5, 0.95])
same = np.isclose(po_c.clear_value, po_c.po_value, rtol=1e-6).mean()
within2 = ((ratio - 1).abs() <= 0.02).mean()
alt_clear = missed("amount_clear")
sum_po, sum_clear = S.amount.sum(), S.amount_clear.sum()

# ---------------------------------------------------------------- results csv
res = pd.DataFrame([
    dict(assumption="base", variant="used", missed=base),
    *[dict(assumption="clock", variant=f"start {r['shift']:+d} d", missed=r["missed"]) for r in shift_res if r["shift"] != 0],
    dict(assumption="clock", variant="arrival clock for all (ignore 22 WAIT_GR)", missed=missed_arr_only),
    dict(assumption="split", variant="PO value / all invoices", missed=alt_all),
    dict(assumption="split", variant="PO value once if any late", missed=po_any_late),
    dict(assumption="split", variant="full PO value per invoice", missed=alt_full),
    dict(assumption="value", variant="clearing value", missed=alt_clear),
])
res["diff"] = res.missed - base
res.to_csv(OUT / "assumptions_ctrl1_results.csv", index=False)


# ---------------------------------------------------------------- charts (inline SVG)
def days_svg():
    """Histogram of days to posting (0-40 days, 1-day bins) with the 10- and 30-day limits."""
    W, H, L, Rm, Tm, B = 700, 300, 50, 16, 14, 44
    pw, ph = W - L - Rm, H - Tm - B
    edges = np.arange(0, 41)
    cnt, _ = np.histogram(S.days.clip(upper=40.999), bins=np.r_[edges, 41])
    top = cnt.max() * 1.1
    sx = lambda d: L + d / 41 * pw
    sy = lambda v: Tm + (1 - v / top) * ph
    g = []
    for t in np.linspace(0, top, 5)[:-1]:
        g.append(f'<line x1="{L}" x2="{L + pw}" y1="{sy(t):.1f}" y2="{sy(t):.1f}" class="grid"/>'
                 f'<text x="{L - 8}" y="{sy(t) + 4:.1f}" class="tick" text-anchor="end">{t:,.0f}</text>')
    for i, v in enumerate(cnt):
        col = "var(--s1)" if i < SLA_SOFT_DAYS else ("var(--s2)" if i < SLA_HARD_DAYS else "var(--s3)")
        lab = f"{i}–{i + 1} days" if i < 40 else "40+ days"
        g.append(f'<rect x="{sx(i) + 1:.1f}" y="{sy(v):.1f}" width="{pw / 41 - 2:.1f}" height="{sy(0) - sy(v):.1f}" rx="2" fill="{col}">'
                 f'<title>{lab}: {v:,} invoices</title></rect>')
    for d in (SLA_SOFT_DAYS, SLA_HARD_DAYS):
        g.append(f'<line x1="{sx(d):.1f}" x2="{sx(d):.1f}" y1="{Tm}" y2="{Tm + ph}" class="lim"/>'
                 f'<text x="{sx(d) + 4:.1f}" y="{Tm + 12}" class="tick">{d} days</text>')
    for d in range(0, 41, 5):
        g.append(f'<text x="{sx(d):.1f}" y="{H - B + 16}" class="tick" text-anchor="middle">{d}{"+" if d == 40 else ""}</text>')
    g.append(f'<text x="{L + pw / 2}" y="{H - 6}" class="axis" text-anchor="middle">Days from clock start to first posting</text>')
    return f'<div class="chart"><svg viewBox="0 0 {W} {H}" role="img" aria-label="Days to posting">{"".join(g)}</svg></div>'


def ratio_svg():
    """Histogram of clearing value / PO value per PO, 0.90-1.10."""
    W, H, L, Rm, Tm, B = 700, 260, 50, 16, 14, 44
    pw, ph = W - L - Rm, H - Tm - B
    edges = np.linspace(0.90, 1.10, 41)
    cnt, _ = np.histogram(ratio.clip(0.9, 1.0999), bins=edges)
    top = cnt.max() * 1.1
    sx = lambda r: L + (r - 0.90) / 0.20 * pw
    sy = lambda v: Tm + (1 - v / top) * ph
    g = []
    for t in np.linspace(0, top, 5)[:-1]:
        g.append(f'<line x1="{L}" x2="{L + pw}" y1="{sy(t):.1f}" y2="{sy(t):.1f}" class="grid"/>'
                 f'<text x="{L - 8}" y="{sy(t) + 4:.1f}" class="tick" text-anchor="end">{t:,.0f}</text>')
    for a, b, v in zip(edges[:-1], edges[1:], cnt):
        g.append(f'<rect x="{sx(a) + 1:.1f}" y="{sy(v):.1f}" width="{sx(b) - sx(a) - 2:.1f}" height="{sy(0) - sy(v):.1f}" rx="2" fill="var(--s1)">'
                 f'<title>{(a - 1) * 100:+.1f}% to {(b - 1) * 100:+.1f}%: {v:,} POs</title></rect>')
    g.append(f'<line x1="{sx(1):.1f}" x2="{sx(1):.1f}" y1="{Tm}" y2="{Tm + ph}" class="lim"/>')
    for r in np.arange(0.90, 1.101, 0.05):
        g.append(f'<text x="{sx(r):.1f}" y="{H - B + 16}" class="tick" text-anchor="middle">{(r - 1) * 100:+.0f}%</text>')
    g.append(f'<text x="{L + pw / 2}" y="{H - 6}" class="axis" text-anchor="middle">Clearing value vs PO value, per PO (outside ±10% shown at the edges)</text>')
    return f'<div class="chart"><svg viewBox="0 0 {W} {H}" role="img" aria-label="Clearing vs PO value">{"".join(g)}</svg></div>'


# ---------------------------------------------------------------- page
funnel_rows = [f"<td>{lab}</td><td class='n'>{num(n)}</td>" for lab, n in funnel]
never_rows = [f"<td>{html.escape(str(a))}</td><td class='n'>{num(n)}</td>" for a, n in never_why.items()]
s1 = next(r for r in shift_res if r["shift"] == 1)
s3 = next(r for r in shift_res if r["shift"] == 3)

EXTRA_CSS = """
:root { --s1: #2a78d6; --s2: #eb6834; --s3: #a3261b; }
.chart { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 12px 14px; margin: 14px 0; max-width: 760px; }
.chart svg { display: block; width: 100%; height: auto; }
.chart .grid { stroke: var(--line); stroke-width: 1; }
.chart .lim { stroke: var(--ink); stroke-width: 1.2; stroke-dasharray: 4 3; }
.chart .tick { font: 11px var(--f-mono); fill: var(--muted); }
.chart .axis { font: 12px var(--f-body); fill: var(--muted); }
"""

page = f"""<meta charset="utf-8">
<title>Assumptions: ZNB1 Manual Queue</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,500;6..72,600&family=Instrument+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap">
<style>{CSS}{EXTRA_CSS}</style>

<div class="wrap">
<header>
  <div class="eyebrow">Accounts payable · Process mining case · ZNB1 manual queue (CTRL1) · Assumptions</div>
  <h1>How the missed discount is calculated, and how much the assumptions matter</h1>
  <p class="lede">Three choices sit behind every missed-discount figure for the ZNB1 manual queue: when the 10-day clock starts and stops, how a PO's value is spread over its invoices, and which value field is used. This report states each one, shows the evidence in the log, and tests the alternatives. Scope: ZNB1 invoices routed to {code('CTRL1')} and posted manually; deleted, never-posted and automatically posted invoices are left out.</p>
  <div class="facts">
    <span><b>{num(len(S))}</b> invoices</span>
    <span><b>{num(n_pos)}</b> POs</span>
    <span><b>{num(n_late)}</b> posted after {SLA_SOFT_DAYS} days</span>
    <span><b>{mnok(base)}</b> NOK missed discount</span>
  </div>
</header>

<h2 id="answer">Short answer</h2>
<div class="findings">
  <div class="finding"><div class="k">Clock</div><p>Starts at arrival in AP and stops at the first posting. Arrival and validation are {num(a2v_s.median(), 0)} second apart at the median, so the start event doesn't matter. If the real start were 1 day earlier (e.g. the invoice date), the loss would be {mnok(s1['missed'])} ({(s1['missed'] - base) / 1e6:+,.2f}M).</p></div>
  <div class="finding"><div class="k">Even split</div><p>The log has one value per PO, so each posted invoice gets an equal share. {pct(n_single_inv / len(S))} of the invoices are on single-invoice POs where the split is exact; the alternatives give {mnok(alt_all)} to {mnok(po_any_late)}.</p></div>
  <div class="finding"><div class="k">PO value</div><p>{code('Case_Invoice_Value_NOK')} is used, not the clearing value. Per PO they differ by a median of {pct(abs(r_q[0.5] - 1), 1)} ({pct(within2)} within ±2%), and the missed discount with the clearing value is {mnok(alt_clear)} ({(alt_clear - base) / 1e6:+,.2f}M).</p></div>
  <div class="finding"><div class="k">Clock matters most</div><p>Split and value field move the loss only within {mnok(min(alt_all, alt_clear, base))}–{mnok(max(alt_all, alt_clear, base))}. Starting the clock 1–3 days earlier raises it to {mnok(s1['missed'])}–{mnok(s3['missed'])}, because {num(near)} invoices are posted just inside the limit.</p></div>
</div>

<h2 id="scope"><span class="step">0</span>Scope</h2>
{table(["Step", "Invoices"], funnel_rows)}
<div class="prose"><p>The {num(len(never))} never-posted CTRL1 invoices end with the following last step; none of them became payable, so they carry no discount:</p></div>
{table(["Last activity of never-posted CTRL1 invoices", "Invoices"], never_rows)}
<div class="prose"><p>An invoice is one <code>PO_ID</code> × <code>SAP_PO_No</code> cycle. The routing is read at the invoice's <i>first</i> <i>Initial validation</i>; the posting is its <i>first</i> posting event.</p></div>

<h2 id="clock"><span class="step">1</span>The clock</h2>
<div class="prose">
<p><b>Assumption.</b> The 10-day discount clock starts when the invoice arrives in AP (its first event, <i>Data management</i>) and stops at its first posting, when it becomes payable. For {', '.join(GR_CLOCK_PO_TYPES)} the project starts the clock at goods receipt (first <code>WAIT_GR</code> event) where there is one; on the CTRL1 route that applies to only {n_gr} of the {num(len(S))} invoices.</p>
<p><b>Evidence.</b> All {num(int((S.first_act == 'Data management').sum()))} invoices start with <i>Data management</i>. <i>Initial validation</i> follows a median of {num(a2v_s.median(), 0)} second later ({pct((a2v_s <= 60).mean())} within a minute), so starting at validation instead gives the same result. The {n_gr} invoices with a goods-receipt step: {gr_late} late with the GR clock, {gr_late_arr} with the arrival clock; using the arrival clock for all of them gives {mnok(missed_arr_only)} ({(missed_arr_only - base) / 1e3:+,.0f}k).</p>
<p><b>What the log can't show.</b> The log has no invoice date and no payment date. Payment terms usually run from the invoice date, which is before the invoice reaches AP, and payment comes after posting. Both make the real clock longer, so the figures here are a lower bound.</p>
</div>
{days_svg()}
<p class="prose" style="font-size:13px;color:var(--muted)">Days from clock start to first posting, 1-day bins. Blue: within {SLA_SOFT_DAYS} days. Orange: {SLA_SOFT_DAYS}–{SLA_HARD_DAYS} days (discount missed). Red: over {SLA_HARD_DAYS} days. Hover for counts.</p>
<h3>Sensitivity: moving the clock start</h3>
{table(["Clock start", "Invoices posted after 10 days", "Missed discount", "Change"], shift_rows)}
<div class="callout"><p><b>The clock is the sensitive assumption.</b> {num(near)} invoices are posted between {SLA_SOFT_DAYS - 2} and {SLA_SOFT_DAYS} days, just inside the limit, so every day the real clock starts earlier pushes many of them over it: +1 day adds {(s1['missed'] - base) / 1e6:,.2f}M, +3 days adds {(s3['missed'] - base) / 1e6:,.2f}M. The headline uses the log as it is, which is the conservative choice.</p></div>

<h2 id="split"><span class="step">2</span>Splitting the PO value over its invoices</h2>
<div class="prose">
<p><b>Assumption.</b> <code>Case_Invoice_Value_NOK</code> holds the same amount on every row of a PO (true for {pct(const_inv)} of ZNB1 POs), so it is the PO total, not the value of one invoice. Each posted invoice gets an equal share: invoice amount = PO value ÷ number of posted invoices on the PO. Missed discount = 2% × that share for each invoice posted after {SLA_SOFT_DAYS} days.</p>
<p><b>Evidence.</b> On single-invoice POs the split is exact: {num(n_single_po)} of the {num(n_pos)} POs, {pct(n_single_inv / len(S))} of the invoices and {pct(val_single)} of the amount. The rest is an estimate; real instalments may differ in size. Never-posted invoices on a PO get no share, so the shares of the posted ones add up to the PO value. {num(partly_out)} POs also have posted invoices outside the scope (e.g. on the goods-receipt road); those take their own equal share.</p>
</div>
{table(["Posted invoices per PO", "POs", "Invoices in scope", "Amount", "Missed discount", "Share of missed"], band_rows)}
<h3>Alternatives</h3>
{table(["Rule", "Meaning", "Missed discount", "Change"], split_rows)}
<div class="callout"><p><b>The even split is the only rule that keeps the total right.</b> Counting the PO once when any invoice is late ({mnok(po_any_late)}) or the full value on every invoice ({mnok(alt_full)}) overstates the loss; splitting over all invoices including never-posted ones changes little ({(alt_all - base) / 1e6:+,.2f}M). The uncertainty is in <i>which</i> invoices of a multi-invoice PO carry the money, not in the total.</p></div>

<h2 id="value"><span class="step">3</span>PO value, not the cleared value</h2>
<div class="prose">
<p><b>Assumption.</b> The amount is <code>Case_Invoice_Value_NOK</code>: the amount on the invoice at the exchange rate on receipt (course data description). <code>Case_Clearing_value_NOK</code>, the amount paid at the rate on clearing, is not used.</p>
<p><b>Why.</b> The discount is a share of the invoiced amount and is decided when the invoice is processed, before clearing. The clearing value is also one value per PO (constant for {pct(const_clear)} of ZNB1 POs), so it gives no per-invoice amount either. The differences are mostly exchange-rate effects between receipt and payment.</p>
<p><b>Evidence.</b> Per PO in scope: the two are identical for {pct(same)} of the POs and within ±2% for {pct(within2)}; the clearing/PO ratio is {r_q[0.05]:.3f} at P5, {r_q[0.5]:.3f} at the median and {r_q[0.95]:.3f} at P95. Allocated over the invoices in scope: {mnok(sum_po, 0)} NOK with the PO value, {mnok(sum_clear, 0)} with the clearing value ({(sum_clear / sum_po - 1) * 100:+.1f}%).</p>
</div>
{ratio_svg()}
<div class="callout"><p><b>The choice hardly matters.</b> With the clearing value the missed discount is {mnok(alt_clear)} instead of {mnok(base)} ({(alt_clear - base) / 1e3:+,.0f}k, {(alt_clear / base - 1) * 100:+.1f}%).</p></div>

<h2 id="summary"><span class="step">4</span>Summary</h2>
{table(["Assumption", "Used", "Main alternative", "Missed discount (used → alternative)"], [
    f"<td><b>Clock</b></td><td>Arrival in AP (GR for {n_gr} invoices) → first posting</td><td>Start 1 / 3 days earlier (invoice date)</td><td class='n'>{mnok(base)} → {mnok(s1['missed'])} / {mnok(s3['missed'])}</td>",
    f"<td><b>Split</b></td><td>PO value ÷ posted invoices</td><td>÷ all invoices</td><td class='n'>{mnok(base)} → {mnok(alt_all)}</td>",
    f"<td><b>Value field</b></td><td><code>Case_Invoice_Value_NOK</code></td><td><code>Case_Clearing_value_NOK</code></td><td class='n'>{mnok(base)} → {mnok(alt_clear)}</td>",
])}
<ul class="tight prose">
  <li><b>Late interest</b> (12.5% p.a. after {SLA_HARD_DAYS} days, in <code>notes/znb1_ctrl1_financial.py</code>) uses the same clock, split and value field, and assumes payment on the posting day.</li>
  <li>The project-wide assumptions for all of ZNB1, including the goods-receipt clock, are in <code>docs/assumptions.html</code>.</li>
</ul>

<footer>
<p>Built by <code>15_report_assumptions_ctrl1.py</code> from <code>output/events.pkl</code> and <code>invoices.csv</code>. Results: <code>output/assumptions_ctrl1_results.csv</code>.</p>
</footer>
</div>
"""
(OUT / "report_assumptions_ctrl1.html").write_text(page, encoding="utf-8")
print("wrote", OUT / "report_assumptions_ctrl1.html", len(page) // 1024, "KB")
print(res.round(0).to_string())
print("scope", len(S), "POs", n_pos, "late", n_late, "never", len(never), "gr", n_gr, "near", near)
print("single PO", n_single_po, n_single_inv, val_single, "partly out", partly_out)
print("ratio q", r_q.round(4).to_dict(), "same", same, "within2", within2, "sums", sum_po, sum_clear)
print(never_why.to_string())
