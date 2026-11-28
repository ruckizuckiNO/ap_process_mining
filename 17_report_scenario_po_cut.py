"""Builds output/report_scenario_po_cut.html: target scenario for the 35 ZNB1 always-manual suppliers.

Scenario (set by the user, 2026-11-28):
  1. POs with PO value <= 1.75M NOK are automated. As in report_po_cut_gbr.html, about 5% of their invoice proxies are
     still posted after > 10 days (gradient-boosting estimate trained on ZJD1, taken from output/po_cut_gbr_results.csv).
  2. POs above 1.75M NOK stay manual, with targets:
     a) 1.75M < PO value <= 5M: every invoice proxy posted within 10 days (no missed discount),
     b) every invoice proxy above the cut posted within 30 days (no late interest),
     c) PO value > 5M: of the invoice proxies posted after > 10 days today, 5% are still late (95% recovered).
Reading of c) (confirmed by the user 2026-11-28): the basis is today's late > 5M invoice proxies; still missed =
5% of them x 2% x the 75th percentile of their amounts (user's choice 2026-11-28, instead of the average).
Range: the 5% are the smallest / the largest of today's late ones.
Sensitivity: c) as 5% of all > 5M invoice proxies; c) at 0% / 10%; segment 1 with the model's 90% interval and the
alternative models.
Scope, clock and amounts as in 13_report_po_cut_gbr.py: the 35 suppliers with 100% of invoice proxies routed to CTRL1 at
their first Initial validation, all their posted invoice proxies; pipeline clock (goods receipt if WAIT_GR, else
arrival) to first posting; amount = PO value split evenly over the PO's posted invoice proxies.
Run 13_report_po_cut_gbr.py first.  Run from the repo root:  conda run -n datasci python -W ignore 17_report_scenario_po_cut.py
"""
import html
import re

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (FIG, OUT, POSTING_ACTIVITIES, ROOT, SLA_HARD_DAYS, SLA_SOFT_DAYS, load_cases, load_events,
                    load_invoices, use_invoice_proxy_labels)

use_invoice_proxy_labels()  # chart text: invoice -> invoice proxy

CSS = re.search(r"<style>(.*?)</style>", (ROOT / "05_report.py").read_text(), re.S).group(1)
CSS = CSS.replace("{{", "{").replace("}}", "}")
K = ["PO_ID", "SAP_PO_No"]
DISCOUNT, INTEREST = 0.02, 0.125
CUT, MID = 1.75e6, 5e6
LATE_BIG = 0.05          # c) share of today's late (> 10 days) > 5M invoice proxies that stay late
AMOUNT_Q = 0.75          # c) amount per still-late invoice proxy = this quantile of today's late > 5M amounts
SEGMENTS = ["auto", "mid", "big"]
SEG_NAME = {"auto": "≤ 1.75M: automated", "mid": "1.75–5M: manual, all within 10 days",
            "big": "> 5M: manual, 5% of today's late stay late"}
SEG_SHORT = {"auto": "PO ≤ 1.75M", "mid": "1.75M < PO ≤ 5M", "big": "PO > 5M"}

# ---------------------------------------------------------------- data (selection as in 13_report_po_cut_gbr.py)
ev = load_events()
ev = ev[ev.SAP_PO_Type_Code == "ZNB1"].sort_values(["PO_ID", "ts", "Event_ID"])
ev["nxt"] = ev.groupby(K).Activity_ID.shift(-1)
iv = ev[ev.Activity_ID == "Initial validation"].groupby(K)
route, nxt = iv.SAP_Next_Processing_Type.first(), iv.nxt.first()
poster = ev[ev.Activity_ID.isin(POSTING_ACTIVITIES)].groupby(K).User_ID.first()

Z = load_invoices()
Z = Z[Z.po_type == "ZNB1"].set_index(K)
Z["route"], Z["supplier"], Z["poster"] = route, ev.groupby(K).Supplier_ID.first(), poster
Z["po_value"] = load_cases().set_index("PO_ID").invoice_value_nok.reindex(Z.index.get_level_values(0)).values
share = Z[Z.route.notna()].groupby("supplier").route.agg(lambda x: (x == "CTRL1").mean())
s100 = share.index[share == 1]
direct = Z.posted & nxt.reindex(Z.index).isin(POSTING_ACTIVITIES)
s100 = s100.intersection(Z.loc[direct, "supplier"].unique())
Z["n_posted"] = Z.groupby(level="PO_ID").posted.transform("sum")
Z["amount"] = np.where(Z.posted, Z.po_value / Z.n_posted.where(Z.n_posted > 0), 0)
Z = Z[Z.supplier.isin(s100) & Z.posted].copy()
d = Z.time_to_post_days
Z["late10"], Z["late30"] = d > SLA_SOFT_DAYS, d > SLA_HARD_DAYS
Z["missed"] = DISCOUNT * Z.amount * Z.late10
Z["interest"] = Z.amount * INTEREST * (d - SLA_HARD_DAYS).clip(lower=0) / 365
Z["seg"] = np.select([Z.po_value <= CUT, Z.po_value <= MID], ["auto", "mid"], "big")

gbr = pd.read_csv(OUT / "po_cut_gbr_results.csv").set_index("cut").loc[CUT]
assert len(Z[Z.seg == "auto"]) == int(gbr.invoices), "scope differs from 13_report_po_cut_gbr.py: rerun 13"
assert abs(Z.loc[Z.seg == "auto", "missed"].sum() - gbr.missed_today) < 1

# ---------------------------------------------------------------- today vs scenario per segment
rows = {}
for s in SEGMENTS:
    x = Z[Z.seg == s]
    r = dict(proxies=len(x), pos=x.index.get_level_values("PO_ID").nunique(), suppliers=x.supplier.nunique(),
             amount=x.amount.sum(), median_days=x.time_to_post_days.median(),
             late_today=int(x.late10.sum()), late_share_today=x.late10.mean(), late30_today=int(x.late30.sum()),
             missed_today=x.missed.sum(), interest_today=x.interest.sum(),
             user1341=(x.poster == "User1341").mean())
    if s == "auto":
        r.update(late_scn=gbr.exp_late, late_share_scn=gbr.exp_late_share, missed_scn=gbr.still_missed,
                 missed_scn_lo=gbr.still_lo, missed_scn_hi=gbr.still_hi)
    elif s == "mid":
        r.update(late_scn=0.0, late_share_scn=0.0, missed_scn=0.0, missed_scn_lo=0.0, missed_scn_hi=0.0)
    else:
        late_amt = np.sort(x.loc[x.late10, "amount"].to_numpy())       # today's late > 5M invoice proxies
        k = int(round(LATE_BIG * len(late_amt)))
        r.update(late_scn=LATE_BIG * r["late_today"], late_share_scn=LATE_BIG * r["late_today"] / r["proxies"],
                 missed_scn=LATE_BIG * r["late_today"] * DISCOUNT * np.quantile(late_amt, AMOUNT_Q),  # 75th pct amount
                 amount_q=np.quantile(late_amt, AMOUNT_Q), amount_mean=late_amt.mean(), amount_median=np.median(late_amt),
                 missed_scn_lo=DISCOUNT * late_amt[:k].sum(),                # if the 5% were the smallest
                 missed_scn_hi=DISCOUNT * late_amt[-k:].sum())               # if the 5% were the largest
    r["interest_scn"] = 0.0 if s != "auto" else np.nan   # b) applies above the cut; not modelled below
    r["recovered"] = r["missed_today"] - r["missed_scn"]
    rows[s] = r
seg = pd.DataFrame(rows).T
tot_today, tot_scn = seg.missed_today.sum(), seg.missed_scn.sum()
tot_lo, tot_hi = seg.missed_scn_lo.sum(), seg.missed_scn_hi.sum()
interest_today_above = seg.loc[["mid", "big"], "interest_today"].sum()
interest_today_auto = seg.loc["auto", "interest_today"]
n_all, amount_all = len(Z), Z.amount.sum()

# Sensitivities
big = Z[Z.seg == "big"]
sens = [
    ("Main: c) 5% of today's late > 5M invoice proxies stay late, at the 75th-percentile amount", tot_scn),
    ("c) same, at their average amount", tot_scn - seg.loc["big", "missed_scn"] + LATE_BIG * seg.loc["big", "missed_today"]),
    ("c) same, at their median amount", tot_scn - seg.loc["big", "missed_scn"]
     + LATE_BIG * seg.loc["big", "late_today"] * DISCOUNT * seg.loc["big", "amount_median"]),
    ("c) the 5% are the smallest of today's late ones", tot_scn - seg.loc["big", "missed_scn"] + seg.loc["big", "missed_scn_lo"]),
    ("c) the 5% are the largest of today's late ones", tot_scn - seg.loc["big", "missed_scn"] + seg.loc["big", "missed_scn_hi"]),
    ("c) 10% of today's late > 5M stay late (75th-percentile amount)", tot_scn - seg.loc["big", "missed_scn"]
     + 0.10 * seg.loc["big", "late_today"] * DISCOUNT * seg.loc["big", "amount_q"]),
    ("c) 0%: all > 5M within 10 days", tot_scn - seg.loc["big", "missed_scn"]),
    ("c) other reading: 5% of all > 5M invoice proxies late",
     tot_scn - seg.loc["big", "missed_scn"] + LATE_BIG * DISCOUNT * seg.loc["big", "amount"]),
    ("Segment ≤ 1.75M: model 90% interval, low", tot_scn - seg.loc["auto", "missed_scn"] + gbr.still_lo),
    ("Segment ≤ 1.75M: model 90% interval, high", tot_scn - seg.loc["auto", "missed_scn"] + gbr.still_hi),
    ("Segment ≤ 1.75M: arrival-clock model", tot_scn - seg.loc["auto", "missed_scn"] + gbr.still_missed_arr),
    ("Segment ≤ 1.75M: ZJD1 WAITGR road only", tot_scn - seg.loc["auto", "missed_scn"] + gbr.still_missed_wgr),
]
out = seg.copy()
out.index = [SEG_SHORT[s] for s in out.index]
out.loc["Total"] = out.sum(numeric_only=True)
out.loc["Total", ["median_days", "late_share_today", "user1341", "late_share_scn", "amount_q", "amount_mean", "amount_median"]] = np.nan
out.to_csv(OUT / "scenario_po_cut_results.csv")

# Today's lateness profile of the manual segments (how far they are from the 10-day target)
prof = {}
for s in ["mid", "big"]:
    x = Z[Z.seg == s].time_to_post_days
    prof[s] = dict(le10=(x <= 10).mean(), d10_15=((x > 10) & (x <= 15)).mean(), d15_30=((x > 15) & (x <= 30)).mean(),
                   gt30=(x > 30).mean(), p90=x.quantile(0.9), median=x.median())

# ---------------------------------------------------------------- chart: waterfall today -> scenario
BLUE, ORANGE, GREEN, GREY = "#2a78d6", "#eb6834", "#2c7a4b", "#9aa5a0"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"]})
steps = [("Missed today", tot_today, "total")] + \
        [(f"{SEG_SHORT[s]}\nrecovered", -seg.loc[s, "recovered"], "delta") for s in SEGMENTS] + \
        [("Still missed\nin the scenario", tot_scn, "total")]
fig, ax = plt.subplots(figsize=(11, 5.6), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
level = 0.0
for i, (lab, v, kind) in enumerate(steps):
    if kind == "total":
        bottom, h, col = 0, v, (ORANGE if i == 0 else BLUE)
        level = v
        txt = f"{v / 1e6:.2f}M"
    else:
        bottom, h, col = level + v, -v, GREEN
        level += v
        txt = f"−{-v / 1e6:.2f}M"
    ax.bar(i, h / 1e6, bottom=bottom / 1e6, width=0.62, color=col, zorder=2)
    ax.text(i, (bottom + h) / 1e6 + tot_today / 1e6 * 0.015, txt, ha="center", va="bottom", fontsize=10.5, color=INK)
    if i < len(steps) - 1:
        ax.plot([i + 0.31, i + 0.69], [level / 1e6] * 2, color=INK2, lw=0.8, zorder=3)
ax.set_xticks(range(len(steps)), [s[0] for s in steps], fontsize=9.5, color=INK)
ax.set_ylabel("Missed 2% cash discount (M NOK)", color=INK2, fontsize=10)
ax.set_ylim(0, tot_today / 1e6 * 1.15)
ax.grid(axis="y", color=GRID, zorder=0)
ax.tick_params(axis="both", length=0, colors=INK2)
for s_ in ax.spines.values():
    s_.set_visible(False)
fig.suptitle(f"35 always-manual ZNB1 suppliers: missed discount {tot_today / 1e6:.2f}M to {tot_scn / 1e6:.2f}M NOK "
             f"in the target scenario ({1 - tot_scn / tot_today:.0%} recovered)", x=0.02, ha="left", y=0.97,
             fontsize=13, color=INK)
fig.text(0.02, 0.012, "≤ 1.75M: automated, still-late share from the ZJD1 gradient-boosting model (report_po_cut_gbr.html). "
         "1.75–5M: all invoices posted within 10 days. > 5M: 5% of today's late invoices stay late.\n"
         "Posted invoices of the 35 suppliers; clock goods receipt (WAIT_GR) or arrival to first posting; amount = PO value "
         "split evenly over the PO's posted invoices.", fontsize=7.5, color=INK2, ha="left")
fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.2)
fig.savefig(FIG / "scenario_po_cut_waterfall.png", facecolor=SURFACE)
plt.close(fig)


# ---------------------------------------------------------------- page
def pct(x, dg=0):
    return "–" if pd.isna(x) else f"{x * 100:.{dg}f}%"


def num(x, dg=0):
    return "–" if pd.isna(x) else f"{x:,.{dg}f}"


def mnok(x, dg=2):
    return f"{x / 1e6:,.{dg}f}M"


def table(head, rows_, cls=""):
    th = "".join(f"<th>{h}</th>" for h in head)
    body = "".join(f"<tr>{r}</tr>" for r in rows_)
    return f'<div class="tbl {cls}"><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


rule_rows = [
    f"<td><b>PO ≤ 1.75M NOK</b></td><td>Automated (handled like normal orders, ZJD1)</td>"
    f"<td>About 5% still posted after 10 days: the model estimate from the PO-cut report, {pct(gbr.exp_late_share, 1)} "
    f"(90% interval {pct(gbr.exp_late_share_lo, 1)}–{pct(gbr.exp_late_share_hi, 1)})</td>",
    "<td><b>1.75M &lt; PO ≤ 5M</b></td><td>Stays manual</td><td>a) every invoice proxy posted within 10 days, "
    "b) within 30 days</td>",
    f"<td><b>PO &gt; 5M</b></td><td>Stays manual</td><td>c) of the {num(seg.loc['big', 'late_today'])} invoice proxies "
    "posted after 10 days today, 5% are still late (95% recovered); b) all within 30 days</td>",
]
seg_rows = []
for s in SEGMENTS:
    r = seg.loc[s]
    rng = (f"<br><small>{mnok(r.missed_scn_lo)}–{mnok(r.missed_scn_hi)}</small>" if s != "mid" else "")
    seg_rows.append(
        f"<td>{html.escape(SEG_SHORT[s])}</td><td class='n'>{num(r.proxies)}</td><td class='n'>{num(r.pos)}</td>"
        f"<td class='n'>{mnok(r.amount, 0)}</td><td class='n'>{num(r.late_today)} ({pct(r.late_share_today)})</td>"
        f"<td class='n'>{mnok(r.missed_today)}</td><td class='n'>{num(r.late_scn)} ({pct(r.late_share_scn, 1)})</td>"
        f"<td class='n'>{mnok(r.missed_scn)}{rng}</td><td class='n'><b>{mnok(r.recovered)}</b></td>")
seg_rows.append(
    f"<td><b>Total</b></td><td class='n'><b>{num(seg.proxies.sum())}</b></td><td class='n'><b>{num(seg.pos.sum())}</b></td>"
    f"<td class='n'><b>{mnok(seg.amount.sum(), 0)}</b></td><td class='n'><b>{num(seg.late_today.sum())} "
    f"({pct(seg.late_today.sum() / seg.proxies.sum())})</b></td><td class='n'><b>{mnok(tot_today)}</b></td>"
    f"<td class='n'><b>{num(seg.late_scn.sum())} ({pct(seg.late_scn.sum() / seg.proxies.sum(), 1)})</b></td>"
    f"<td class='n'><b>{mnok(tot_scn)}</b><br><small>{mnok(tot_lo)}–{mnok(tot_hi)}</small></td>"
    f"<td class='n'><b>{mnok(tot_today - tot_scn)}</b></td>")
prof_rows = [f"<td>{html.escape(SEG_SHORT[s])}</td><td class='n'>{num(seg.loc[s, 'proxies'])}</td><td class='n'>{p['median']:.1f}</td>"
             f"<td class='n'>{p['p90']:.1f}</td><td class='n'>{pct(p['le10'])}</td><td class='n'>{pct(p['d10_15'])}</td>"
             f"<td class='n'>{pct(p['d15_30'])}</td><td class='n'>{pct(p['gt30'], 1)}</td>" for s, p in prof.items()]
sens_rows = [f"<td>{lab}</td><td class='n'>{mnok(v)}</td><td class='n'>{mnok(tot_today - v)}</td>"
             f"<td class='n'>{pct(1 - v / tot_today)}</td>" for lab, v in sens]
mid, bg, au = seg.loc["mid"], seg.loc["big"], seg.loc["auto"]
manual_n = int(mid.proxies + bg.proxies)

page = f"""<meta charset="utf-8">
<title>Target Scenario: PO-Value Cut</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,500;6..72,600&family=Instrument+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap">
<style>{CSS}</style>

<div class="wrap">
<header>
  <div class="eyebrow">Accounts payable · Process mining case · ZNB1 always-manual suppliers · Target scenario</div>
  <h1>Target Scenario: Automate Small POs, Speed Up the Rest</h1>
  <p class="lede">What the 35 ZNB1 suppliers whose invoice proxies always go to the manual queue would still lose in cash discount if POs up to 1.75M NOK were automated and the larger POs met posting targets.</p>
  <div class="facts">
    <span><b>{num(n_all)}</b> posted invoice proxies</span>
    <span><b>{num(Z.index.get_level_values('PO_ID').nunique())}</b> POs</span>
    <span><b>{mnok(amount_all, 0)}</b> NOK</span>
    <span>missed today <b>{mnok(tot_today)}</b></span>
    <span>in the scenario <b>{mnok(tot_scn)}</b></span>
  </div>
</header>

<ul class="toc">
  <li><a href="#answer">Short answer</a></li>
  <li><a href="#rules">1 · Scenario</a></li>
  <li><a href="#result">2 · Result</a></li>
  <li><a href="#gap">3 · How far from the targets</a></li>
  <li><a href="#sens">4 · Sensitivity</a></li>
  <li><a href="#method">5 · Method and assumptions</a></li>
</ul>

<h2 id="answer">Short answer</h2>
<div class="findings">
  <div class="finding"><div class="k">{mnok(tot_today)} → {mnok(tot_scn)}</div><p>NOK missed 2% discount, today vs. target scenario: <b>{mnok(tot_today - tot_scn)}</b> ({pct(1 - tot_scn / tot_today)}) recovered.</p></div>
  <div class="finding"><div class="k">{mnok(bg.recovered)}</div><p>of that comes from the {num(bg.pos)} POs above 5M NOK ({num(bg.proxies)} invoice proxies). The large POs carry most of the money, so the posting target on them matters most.</p></div>
  <div class="finding"><div class="k">{mnok(au.recovered)}</div><p>from automating the {num(au.pos)} POs up to 1.75M NOK: {pct(au.pos / seg.pos.sum())} of the POs, but only {pct(au.missed_today / tot_today)} of today's missed discount.</p></div>
  <div class="finding"><div class="k">{num(manual_n)}</div><p>invoice proxies stay manual ({pct(manual_n / n_all)}). Today {pct(1 - (mid.late_today + bg.late_today) / manual_n)} of them are posted within 10 days, at a median of {Z[Z.seg != 'auto'].time_to_post_days.median():.1f} days: the targets need a faster queue, not a new process.</p></div>
</div>

<h2 id="rules"><span class="step">1</span>The scenario</h2>
<div class="prose"><p>The PO value decides how an invoice proxy is handled. The rules apply to all posted invoice proxies of the 35 suppliers whose invoice proxies are 100% routed to the manual queue (<code>CTRL1</code>) at validation, the same scope as the PO-cut report.</p></div>
{table(["PO value", "Handling", "Target / assumption"], rule_rows)}
<div class="callout"><p><b>b) adds nothing today.</b> No invoice proxy above the cut is posted after 30 days today, so the 30-day target costs nothing extra; it guards against late interest ({num(interest_today_above)} NOK today above the cut). Below the cut {num(au.late30_today)} invoice proxies take longer than 30 days today ({num(interest_today_auto)} NOK interest); the model does not estimate the 30-day share after automation.</p></div>

<h2 id="result"><span class="step">2</span>Result: today vs. scenario</h2>
{table(["Segment", "Invoice proxies", "POs", "Amount (NOK)", "> 10 days today", "Missed today", "> 10 days scenario", "Missed scenario", "Recovered"], seg_rows)}
<figure class="wide"><div class="plate"><img src="figures/scenario_po_cut_waterfall.png" alt="Waterfall of missed discount from today to the scenario" loading="lazy"></div>
<figcaption>From today's missed discount to the scenario: what each segment recovers. Ranges under "Missed scenario": ≤ 1.75M = the model's 90% interval; &gt; 5M = the 5% smallest to the 5% largest of today's late invoice proxies.</figcaption></figure>
<div class="prose">
<p><b>Reading the table.</b> Automation helps the many small POs, but the money sits in the few large ones: the {num(bg.pos)} POs above 5M NOK hold {pct(bg.amount / amount_all)} of the amount and {pct(bg.missed_today / tot_today)} of today's missed discount. Even after the scenario they keep {pct(bg.missed_scn / tot_scn)} of what is still lost ({mnok(bg.missed_scn)} of {mnok(tot_scn)}): only {num(bg.late_scn, 0)} invoice proxies stay late there, but each carries a large amount.</p>
</div>

<h2 id="gap"><span class="step">3</span>How far today's manual queue is from the targets</h2>
<div class="prose"><p>The two manual segments are posted like this today (days from clock start to first posting):</p></div>
{table(["Segment", "Invoice proxies", "Median days", "90% within (days)", "≤ 10 days", "10–15 days", "15–30 days", "> 30 days"], prof_rows)}
<div class="prose">
<p>Most late invoice proxies miss the limit by only a few days: the queue is worked first-in-first-out at a median of about 8 days (see the FIFO report). Two levers reach the targets without automation:</p>
<ul>
  <li><b>Order the queue by deadline × amount</b> instead of arrival, so large invoice proxies are posted before day 10.</li>
  <li><b>Cut the routine wait</b> from about 8 days to well under 10, e.g. by sharing the queue of the main clerk ({pct(Z[Z.seg != 'auto'].poster.eq('User1341').mean())} of the manual invoice proxies above the cut are posted by User1341).</li>
</ul>
</div>

<h2 id="sens"><span class="step">4</span>Sensitivity</h2>
<div class="prose"><p>Target c) is based on the invoice proxies on POs above 5M that are posted after 10 days <i>today</i>: 5% of them stay late. Which ones stay late matters, because the amounts differ a lot: the main result values each invoice proxy that stays late at the 75th percentile of their amounts ({mnok(seg.loc['big', 'amount_q'])} NOK; average {mnok(seg.loc['big', 'amount_mean'])}, median {mnok(seg.loc['big', 'amount_median'])}), a cautious choice above the typical amount. The range takes the 5% smallest or largest. The row "other reading" shows the alternative interpretation (5% of <i>all</i> invoice proxies above 5M late) for comparison.</p></div>
{table(["Variant", "Still missed", "Recovered", "Recovered share"], sens_rows)}
<div class="prose"><p>The largest uncertainty is <i>which</i> of today's late invoice proxies above 5M stay late: {mnok(seg.loc['big', 'missed_scn_lo'])} to {mnok(seg.loc['big', 'missed_scn_hi'])} for that segment. The automation model moves the total by at most {mnok(abs(gbr.still_missed_arr - gbr.still_missed_wgr))} across its variants.</p></div>

<h2 id="method"><span class="step">5</span>Method and assumptions</h2>
<dl class="gloss">
  <dt>Scope</dt><dd>The 35 ZNB1 suppliers with 100% of invoice proxies routed to <code>CTRL1</code> at their first <i>Initial validation</i> (and at least one posted straight from the queue); all their posted invoice proxies, {num(n_all)} on {num(Z.index.get_level_values('PO_ID').nunique())} POs. Deleted and never-posted invoice proxies are left out.</dd>
  <dt>Today</dt><dd>Clock from goods receipt (first <code>WAIT_GR</code>) or, without one, arrival in AP, to the first posting. Missed discount = 2% of the amount of every invoice proxy posted after more than 10 days. Late interest = 12.5% p.a. from day 30, payment assumed on the posting day.</dd>
  <dt>Amount</dt><dd>PO value (<code>Case_Invoice_Value_NOK</code>) split evenly over the PO's posted invoice proxies; the log has no amount per invoice.</dd>
  <dt>≤ 1.75M</dt><dd>Still-late share from the monotone gradient-boosting model trained on posted ZJD1 invoice proxies (PO value as input), with a bootstrap 90% interval; taken unchanged from <code>po_cut_gbr_results.csv</code>.</dd>
  <dt>1.75–5M</dt><dd>Target a): 0% after 10 days, so nothing missed.</dd>
  <dt>&gt; 5M</dt><dd>Target c): basis = the {num(seg.loc['big', 'late_today'])} invoice proxies on POs above 5M posted after 10 days today ({mnok(seg.loc['big', 'missed_today'])} missed discount). 5% of them stay late ({num(seg.loc['big', 'late_scn'], 1)} invoice proxies), each valued at the 75th percentile of today's late amounts ({mnok(seg.loc['big', 'amount_q'])} NOK): still missed = {num(seg.loc['big', 'late_scn'], 1)} × 2% × {mnok(seg.loc['big', 'amount_q'])} = {mnok(seg.loc['big', 'missed_scn'])}. Which ones stay late is not modelled; range {mnok(seg.loc['big', 'missed_scn_lo'])} (the 5% smallest) to {mnok(seg.loc['big', 'missed_scn_hi'])} (the 5% largest).</dd>
  <dt>Not included</dt><dd>Capacity: whether the clerks can reach a) and c) is not modelled. Payment terms are assumed 2%/10 days for every supplier (not in the log). The log ends at posting, not payment.</dd>
</dl>

<footer>
<p>Built by <code>17_report_scenario_po_cut.py</code> from <code>events.pkl</code>, <code>invoices.csv</code>, <code>cases.csv</code> and <code>po_cut_gbr_results.csv</code> (run <code>13_report_po_cut_gbr.py</code> first). Results in <code>output/scenario_po_cut_results.csv</code>.</p>
</footer>
</div>
"""
(OUT / "report_scenario_po_cut.html").write_text(page, encoding="utf-8")
print("wrote", OUT / "report_scenario_po_cut.html")
pd.set_option("display.width", 250)
print(seg.round(3).to_string())
print(f"today {tot_today:,.0f}  scenario {tot_scn:,.0f} ({tot_lo:,.0f}-{tot_hi:,.0f})  recovered {tot_today - tot_scn:,.0f}")
for lab, v in sens:
    print(f"{lab:70s} {v / 1e6:6.2f}M")
print(prof)
