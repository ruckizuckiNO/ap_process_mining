"""Builds output/report_queue_fifo.html: a one-page summary of how the ZNB1 manual queue (CTRL1) is worked.

Reads notes/znb1_queue_order_test.csv (run notes/znb1_queue_order_test.py first) for the per-clerk queue-order tests,
and notes/znb1_ctrl1_100pct_suppliers.xlsx (run notes/znb1_ctrl1_100pct_suppliers.py first) for who posts the
invoices of the 100%-CTRL1 suppliers and their missed 2% discount. Charts are inline SVG.
Run from the repo root:  conda run -n datasci python -W ignore 14_report_queue_fifo.py
"""
import html
import re

import pandas as pd

from common import OUT, ROOT, SLA_SOFT_DAYS

CSS = re.search(r"<style>(.*?)</style>", (ROOT / "05_report.py").read_text(), re.S).group(1)
CSS = CSS.replace("{{", "{").replace("}}", "}")
FIFO_CLERK = "User1341"

q = pd.read_csv(ROOT / "notes" / "znb1_queue_order_test.csv").sort_values("postings", ascending=False)
inv = pd.read_excel(ROOT / "notes" / "znb1_ctrl1_100pct_suppliers.xlsx", sheet_name="Invoices")
# Days and missed discount are Excel formulas in that workbook: recompute with the same rule.
days = (pd.to_datetime(inv["First posting"]) - pd.to_datetime(inv["Arrival in AP"])).dt.total_seconds() / 86400
inv["missed"] = 0.02 * inv["Invoice amount (NOK, est.)"] * (days > SLA_SOFT_DAYS)
by_clerk = inv.groupby("Posted by").agg(invoices=("missed", "size"), missed=("missed", "sum"))
m_tot, n_tot = by_clerk.missed.sum(), by_clerk.invoices.sum()
f = q.set_index("user").loc[FIFO_CLERK]
f_inv, f_missed = by_clerk.loc[FIFO_CLERK, "invoices"], by_clerk.loc[FIFO_CLERK, "missed"]
others = q[q.user != FIFO_CLERK]
all_posts = q.postings.sum()


def pct(x, d=0):
    return f"{x * 100:.{d}f}%"


def num(x, d=0):
    return f"{x:,.{d}f}"


def dots_svg(col, title, left_lab, right_lab):
    """Dot plot per clerk on a 0..1 scale with the 'random' line at 0.5."""
    W, L, Rm, rowh, Tm = 640, 96, 20, 24, 30
    pw = W - L - Rm
    H = Tm + rowh * len(q) + 30
    sx = lambda v: L + v * pw
    g = [f'<text x="{L}" y="16" class="ttl">{html.escape(title)}</text>']
    for v in [0, 0.25, 0.5, 0.75, 1]:
        g.append(f'<line x1="{sx(v):.1f}" x2="{sx(v):.1f}" y1="{Tm - 4}" y2="{H - 26}" class="{"mid" if v == 0.5 else "grid"}"/>'
                 f'<text x="{sx(v):.1f}" y="{H - 10}" class="tick" text-anchor="middle">{v:g}</text>')
    g.append(f'<text x="{sx(0.5) + 4:.1f}" y="{Tm + 6}" class="note">random</text>')
    for i, r in enumerate(q.itertuples()):
        y0 = Tm + 8 + i * rowh
        hi = r.user == FIFO_CLERK
        g.append(f'<line x1="{sx(0.5):.1f}" x2="{sx(getattr(r, col)):.1f}" y1="{y0}" y2="{y0}" class="stem{" hi" if hi else ""}"/>'
                 f'<circle cx="{sx(getattr(r, col)):.1f}" cy="{y0}" r="{6 if hi else 4.5}" class="dot{" hi" if hi else ""}">'
                 f'<title>{r.user}: {getattr(r, col):.2f} · {r.postings:,} postings · median wait {r.median_wait_days:.1f} days</title></circle>'
                 f'<text x="{L - 10}" y="{y0 + 4}" class="lab{" b" if hi else ""}" text-anchor="end">{r.user}</text>')
    cap = f'<div class="axcap"><span>← {left_lab}</span><span>{right_lab} →</span></div>'
    return f'<div class="chart"><svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(title)}">{"".join(g)}</svg>{cap}</div>'


rows = [f"<td{' class=hl' if r.user == FIFO_CLERK else ''}><b>{r.user}</b></td><td class='n'>{num(r.postings)}</td>"
        f"<td class='n'>{num(r.median_wait_days, 1)}</td><td class='n'>{num(r.median_waiting_at_posting)}</td>"
        f"<td class='n'><b>{r.age_pct:.2f}</b></td><td class='n'>{r.within_day_tau:.2f}</td><td class='n'>{r.value_pct:.2f}</td>"
        for r in q.itertuples()]
head = ["Clerk", "Postings", "Median wait (days)", "Invoices waiting at posting (median)", "Age rank<br><small>1 = oldest first</small>",
        "Same-day order τ<br><small>1 = arrival order</small>", "Value rank<br><small>1 = largest first</small>"]
table = (f'<div class="tbl"><table><thead><tr>{"".join(f"<th>{h}</th>" for h in head)}</tr></thead>'
         f'<tbody>{"".join(f"<tr>{r}</tr>" for r in rows)}</tbody></table></div>')

EXTRA_CSS = """
:root { --hi: #2a78d6; --lo: #9aa7a1; }
.chart { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 12px 14px; margin: 14px 0; }
.chart svg { display: block; width: 100%; height: auto; }
.chart .grid { stroke: var(--line); stroke-width: 1; }
.chart .mid { stroke: var(--muted); stroke-width: 1; stroke-dasharray: 4 4; }
.chart .stem { stroke: var(--lo); stroke-width: 2; opacity: .5; } .chart .stem.hi { stroke: var(--hi); opacity: .8; }
.chart .dot { fill: var(--lo); stroke: var(--surface); stroke-width: 1.5; } .chart .dot.hi { fill: var(--hi); }
.chart .tick { font: 11px var(--f-mono); fill: var(--muted); }
.chart .note { font: 11px var(--f-body); fill: var(--muted); }
.chart .ttl { font: 600 13px var(--f-body); fill: var(--ink); }
.chart .lab { font: 12px var(--f-mono); fill: var(--ink); } .chart .lab.b { font-weight: 600; fill: var(--hi); }
.axcap { display: flex; justify-content: space-between; font-size: 12px; color: var(--muted); padding: 0 20px 0 96px; }
.stack .chart { max-width: 760px; }
td.hl, td.hl b { color: var(--hi); }
th small { font-weight: 400; text-transform: none; letter-spacing: 0; }
"""

page = f"""<meta charset="utf-8">
<title>ZNB1 Manual Queue: FIFO?</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,500;6..72,600&family=Instrument+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap">
<style>{CSS}{EXTRA_CSS}</style>

<div class="wrap">
<header>
  <div class="eyebrow">Accounts payable · Process mining case · ZNB1 manual queue (CTRL1)</div>
  <h1>One clerk works the queue first-in-first-out, and that is where the discount is lost</h1>
  <p class="lede">Is the ZNB1 manual queue worked first-in-first-out? For {FIFO_CLERK}, yes, almost strictly. For the other clerks, no. Nobody sorts by PO value. {FIFO_CLERK} posts most of the invoices from the always-manual suppliers, and almost all of their missed 2% discount.</p>
  <div class="facts">
    <span><b>{f.age_pct:.2f}</b> age rank for {FIFO_CLERK} (1 = FIFO)</span>
    <span><b>{num(f.median_wait_days, 1)}</b> days median wait</span>
    <span><b>{pct(f_missed / m_tot)}</b> of the missed discount</span>
    <span><b>≈ 0.5</b> value rank for every clerk</span>
  </div>
</header>

<h2 id="answer">Short answer</h2>
<div class="findings">
  <div class="finding"><div class="k">FIFO</div><p>{FIFO_CLERK} posted the oldest waiting invoice {pct(f.age_pct)} of the time, with a median of {num(f.median_waiting_at_posting)} invoices in the queue. Same-day postings follow arrival order (τ = {f.within_day_tau:.2f}).</p></div>
  <div class="finding"><div class="k">Not FIFO</div><p>The other {len(others)} clerks score {others.age_pct.min():.2f}–{others.age_pct.max():.2f}; several pick <i>newer</i> invoices first and keep waits short ({others.median_wait_days.min():.1f}–{others.median_wait_days.max():.1f} days).</p></div>
  <div class="finding"><div class="k">No value priority</div><p>Value rank {q.value_pct.min():.2f}–{q.value_pct.max():.2f}, for most clerks close to 0.5 ({FIFO_CLERK}: {f.value_pct:.2f}). A large PO waits about as long as a small one.</p></div>
  <div class="finding"><div class="k">{num(f_missed / 1e6, 2)}M NOK</div><p>of the {num(m_tot / 1e6, 2)}M missed discount of the 100%-manual suppliers is on invoices {FIFO_CLERK} posted ({pct(f_inv / n_tot)} of the invoices). A strict, slow FIFO queue lets large invoices pass the {SLA_SOFT_DAYS}-day deadline.</p></div>
</div>

<h2 id="evidence"><span class="step">1</span>The evidence</h2>
<div class="stack">
{dots_svg("age_pct", "Age rank: does the clerk post the oldest invoice?", "newest first", "oldest first (FIFO)")}
{dots_svg("value_pct", "Value rank: does the clerk post the largest PO first?", "smallest first", "largest first")}
</div>
<p class="prose" style="font-size:13px;color:var(--muted)">One dot per clerk with at least 100 manual ZNB1 postings, sorted by postings. 0.5 = no order (random). Hover a dot for the numbers.</p>
{table}

<h2 id="method"><span class="step">2</span>How it was tested</h2>
<div class="prose">
<p><b>Queue.</b> Per clerk: the ZNB1 invoices that clerk eventually posted manually, from the moment each entered <code>CTRL1</code> to its posting ({num(all_posts)} postings by {len(q)} clerks).</p>
<p><b>Age rank.</b> At every posting: the share of the clerk's other waiting invoices that arrived <i>later</i> than the one posted, averaged over all postings. 1 = always the oldest (FIFO), 0.5 = random, 0 = always the newest.</p>
<p><b>Same-day order.</b> Kendall τ between posting order and arrival order among one day's postings. <b>Value rank.</b> As the age rank, but by PO value.</p>
<p><b>Why it isn't chance.</b> With a median of {num(f.median_waiting_at_posting)} invoices waiting at each of {num(f.postings)} postings, picking the oldest {pct(f.age_pct)} of the time can't happen by accident.</p>
</div>

<h2 id="action"><span class="step">3</span>What it means</h2>
<ul class="tight prose">
  <li><b>Sort the queue by deadline and value.</b> {FIFO_CLERK}'s queue takes {num(f.median_wait_days, 1)} days at the median, close to the {SLA_SOFT_DAYS}-day limit, and ignores value. Working it by discount deadline × amount would protect the large invoices without automating anything.</li>
  <li><b>Spread the load.</b> The other clerks clear their queues in days. The FIFO queue is a capacity bottleneck on one person, tied to the suppliers assigned to them.</li>
</ul>
<div class="callout"><p><b>Caveats.</b> The log has no queue view, sort order or assignment data, so the queue is rebuilt from the invoices each clerk <i>posted</i>. The pattern is clear behaviour, not a documented rule. Assignment appears to follow the supplier, so each clerk's queue has a different supplier mix.</p></div>

<footer>
<p>Built by <code>14_report_queue_fifo.py</code> from <code>notes/znb1_queue_order_test.csv</code> (<code>notes/znb1_queue_order_test.py</code>) and <code>notes/znb1_ctrl1_100pct_suppliers.xlsx</code> (invoices posted straight from the queue, arrival clock, missed discount = 2% of invoices posted after more than {SLA_SOFT_DAYS} days).</p>
</footer>
</div>
"""
(OUT / "report_queue_fifo.html").write_text(page, encoding="utf-8")
print("wrote", OUT / "report_queue_fifo.html", len(page) // 1024, "KB")
print(FIFO_CLERK, f_inv, n_tot, f_missed, m_tot)
