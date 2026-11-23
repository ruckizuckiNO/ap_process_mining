"""Builds output/report_znb1_ctrl1.html: what the log can and can't say about why ZNB1 invoices go to the manual queue CTRL1.

Unit: one ZNB1 invoice (PO_ID x SAP_PO_No), routing code at its first Initial validation, restricted to CTRL1 vs WAITGR.
Reads output/events.pkl; reuses the stylesheet of 05_report.py. The other reports are not touched.
"""
import html
import re

import numpy as np
import pandas as pd
from common import OUT, ROOT
from ctrl1_data import T, build
CSS = re.search(r"<style>(.*?)</style>", (ROOT / "05_report.py").read_text(), re.S).group(1)
CSS = CSS.replace("{{", "{").replace("}}", "}")


def pct(x, d=0):
    return "–" if pd.isna(x) else f"{x * 100:.{d}f}%"


def num(x, d=0):
    return "–" if pd.isna(x) else f"{x:,.{d}f}"


def code(c):
    return f'<span class="code">{c}</span>'


def table(head, rows, cls=""):
    th = "".join(f"<th>{h}</th>" for h in head)
    body = "".join(f"<tr>{r}</tr>" for r in rows)
    return f'<div class="tbl {cls}"><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def bar(x):
    return f'<div class="bar"><span style="width:{x * 100:.1f}%"></span></div>'


def share_rows(g, label_fmt=str):
    """Rows of (group, invoices, CTRL1 share, bar) from a groupby result with columns size/mean."""
    return [f"<td>{label_fmt(k)}</td><td class='n'>{num(r['size'])}</td><td class='n'>{pct(r['mean'])}</td><td>{bar(r['mean'])}</td>"
            for k, r in g.iterrows()]


HEAD = ["", "Invoices", "To CTRL1", ""]

# ---------------------------------------------------------------- data
val, dm, v = build()
all_codes = val.SAP_Next_Processing_Type.value_counts()
n, base = len(v), v.ctrl.mean()
n_ctrl = int(v.ctrl.sum())

# What the SAP fields at validation contain
fields = []
for f, src in [("SAP_Processing_Type", val), ("User_ID", val), ("SAP_Object_Code", val), ("SAP_Processing_Type", dm)]:
    vals = src[f].dropna().unique()
    shown = "empty on every event" if len(vals) == 0 else ", ".join(f"<code>{html.escape(str(x))}</code>" for x in vals[:3])
    fields.append(f"<td>{'Data management' if src is dm else 'Initial validation'}</td><td><code>{f}</code></td><td>{shown}</td>")

# Supplier
s = v.groupby("Supplier_ID").ctrl.agg(["size", "mean"])
s["cls"] = pd.cut(s["mean"], [-0.01, 0.05, 0.95, 1.0], labels=["Almost never CTRL1 (≤ 5%)", "Mixed", "Almost always CTRL1 (> 95%)"])
cls = s.groupby("cls", observed=True).agg(suppliers=("size", "count"), invoices=("size", "sum"))
cls["ctrl"] = v.groupby(v.Supplier_ID.map(s.cls), observed=True).ctrl.sum()
always = cls.loc["Almost always CTRL1 (> 95%)"]
cls_rows = [f"<td>{k}</td><td class='n'>{num(r.suppliers)}</td><td class='n'>{num(r.invoices)}</td><td class='n'>{pct(r.invoices / n)}</td>"
            f"<td class='n'>{num(r.ctrl)}</td><td class='n'>{pct(r.ctrl / n_ctrl)}</td>" for k, r in cls.iterrows()]
big = s[s["size"] >= 200].sort_values("size", ascending=False)
sup_rows = [f"<td>{k}</td><td class='n'>{num(r['size'])}</td><td class='n'>{pct(r['mean'])}</td><td>{bar(r['mean'])}</td><td>{r.cls}</td>"
            for k, r in big.iterrows()]
always_big = list(s[(s.cls == "Almost always CTRL1 (> 95%)")].sort_values("size", ascending=False).index[:5])

# Same PO
po = v.groupby("PO_ID").ctrl.agg(["size", "mean"])
multi = po[po["size"] > 1]
po_same = ((multi["mean"] == 0) | (multi["mean"] == 1)).mean()
first = v.groupby("PO_ID").ctrl.transform("first")
later = v[v.inv_pos > 1]
later_same = (later.ctrl == first[later.index]).mean()
after_ctrl, after_wait = later[first[later.index]].ctrl.mean(), later[~first[later.index]].ctrl.mean()

# Value
bands = [0, 1e5, 5e5, 1e6, 2e6, 5e6, 1e7, np.inf]
blabel = ["< 100k", "100k–500k", "500k–1M", "1M–2M", "2M–5M", "5M–10M", "> 10M"]
vb = v.groupby(pd.cut(v.value, bands, labels=blabel), observed=True).ctrl.agg(["size", "mean"])
mixed = v[v.Supplier_ID.isin(s.index[s.cls == "Mixed"])].copy()
mixed = mixed[mixed.groupby("Supplier_ID").Supplier_ID.transform("size") >= 8]
mixed["vq"] = mixed.groupby("Supplier_ID").value.transform(lambda x: pd.qcut(x.rank(method="first"), 4, labels=False))
wq = mixed.groupby("vq").ctrl.agg(["size", "mean"])
wq.index = ["Lowest quarter", "2nd quarter", "3rd quarter", "Highest quarter"]

# Invoice order, amount difference, timing
pos = v.groupby(v.inv_pos.clip(upper=5)).ctrl.agg(["size", "mean"])
pos.index = ["1st invoice", "2nd", "3rd", "4th", "5th or later"]
v["diff_cls"] = np.select([v["diff"].abs() < 1e-9, v["diff"] < 0], ["Equal", "Clearing lower"], "Clearing higher")
dfc = v.groupby("diff_cls").ctrl.agg(["size", "mean"]).loc[["Equal", "Clearing lower", "Clearing higher"]]
wd = v.groupby(v.ts.dt.dayofweek).ctrl.agg(["size", "mean"])
wd.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][: len(wd)]
mo = v.groupby(v.ts.dt.to_period("M")).ctrl.agg(["size", "mean"])
mo = mo[mo["size"] >= 50]
mo.index = [p.strftime("%b %Y") for p in mo.index]

# Predictability: results of the model report (09_report_ctrl1_model.py, run it first)
MR = pd.read_csv(OUT / "ctrl1_model_results.csv", index_col="model")
auc_all, acc_all = MR.loc["Gradient boosting, all fields", ["auc", "acc"]]
auc_ns = MR.loc["Gradient boosting, no supplier", "auc"]
auc_so = MR.loc["Gradient boosting, supplier only", "auc"]

EXTRA_CSS = """
.bar { background: var(--accent-soft); border-radius: 3px; height: 10px; min-width: 100px; }
.bar span { display: block; height: 100%; background: var(--accent); border-radius: 3px; }
.grid2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 420px), 1fr)); gap: 0 20px; }
.grid2 > * { min-width: 0; }
"""

page = f"""<meta charset="utf-8">
<title>ZNB1 Manual Queue Drivers</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,500;6..72,600&family=Instrument+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap">
<style>{CSS}{EXTRA_CSS}</style>

<div class="wrap">
<header>
  <div class="eyebrow">Accounts payable · Process mining case · ZNB1 routing</div>
  <h1>Why do ZNB1 invoices go to the manual queue?</h1>
  <p class="lede">At <i>Initial validation</i>, SAP sends each invoice either to the goods-receipt check (<code>WAITGR</code>, the road to automatic posting) or to the manual control queue (<code>CTRL1</code>). This report checks what the event log can tell us about that decision for manual orders ({code(T)}).</p>
  <div class="facts">
    <span><b>{num(n)}</b> ZNB1 invoices</span>
    <span><b>{pct(base)}</b> to CTRL1</span>
    <span><b>{num(v.Supplier_ID.nunique())}</b> suppliers</span>
    <span><b>{num(v.PO_ID.nunique())}</b> POs</span>
  </div>
</header>

<h2 id="answer">Short answer</h2>
<div class="findings">
  <div class="finding"><div class="k">No reason code</div><p>The log doesn't record why an invoice goes to <code>CTRL1</code>. The SAP fields at validation are identical for both roads, so the rule itself can't be read from the data.</p></div>
  <div class="finding"><div class="k">{pct(always.ctrl / n_ctrl)}</div><p>of <code>CTRL1</code> invoices come from {num(always.suppliers)} suppliers whose invoices almost always go there. The supplier is the strongest pattern by far.</p></div>
  <div class="finding"><div class="k">{pct(later_same)}</div><p>of later invoices on a PO take the same road as the PO's first invoice. The decision sticks to the PO.</p></div>
  <div class="finding"><div class="k">AUC {auc_all:.2f}</div><p>A model using only fields from the log predicts the road correctly for {pct(acc_all)} of invoices. The decision is systematic, not random, and points to a supplier or PO setting.</p></div>
</div>

<h2 id="scope"><span class="step">Scope</span>What was analysed</h2>
<div class="prose">
<p>One row per ZNB1 invoice (<code>PO_ID</code> × <code>SAP_PO_No</code>), with the routing code (<code>SAP_Next_Processing_Type</code>) of its first <i>Initial validation</i>. Only invoices sent to <code>CTRL1</code> or <code>WAITGR</code> are compared. The few routed via Warehouse Management (<code>ZWMGRW</code>, {num(all_codes.get('ZWMGRW', 0))}) or to deletion (<code>DELE</code>, {num(all_codes.get('DELE', 0))}) are left out.</p>
<p>The value is <code>Case_Invoice_Value_NOK</code>, which is the PO total, not the value of the single invoice. Shares below are the share of invoices sent to <code>CTRL1</code>; the overall average is {pct(base)}.</p>
</div>

<h2 id="missing"><span class="step">1</span>What the log doesn't contain</h2>
{table(["Event", "Field", "Values on ZNB1 invoices"], fields)}
<div class="prose"><p>Validation is always done by the same system user with the same processing type, and the object code is empty. There's no validation message, block reason, tolerance or variance code. Those are the fields that would say why an invoice was stopped, and they live in SAP (e.g. the invoice's blocking reasons and messages), not in this extract.</p></div>

<h2 id="supplier"><span class="step">2</span>The supplier decides most of it</h2>
<div class="prose"><p>Suppliers fall into three groups: those whose invoices almost always go to <code>CTRL1</code>, those that almost never do, and a mixed group.</p></div>
{table(["Supplier group", "Suppliers", "Invoices", "Share of invoices", "Of which to CTRL1", "Share of all CTRL1"], cls_rows)}
<h4>Suppliers with at least 200 invoices</h4>
{table(["Supplier", "Invoices", "To CTRL1", "", "Group"], sup_rows)}
<div class="callout"><p><b>Same PO type, same process, opposite roads.</b> Supplier361 sends 100% of its invoices to the manual queue, while Supplier115 and Supplier44, also large ZNB1 suppliers, almost never do. Something about the supplier, or the way its POs are set up, makes the difference.</p></div>

<h2 id="po"><span class="step">3</span>The decision sticks to the PO</h2>
<div class="prose">
<p>Of the {num(len(multi))} POs with more than one invoice, {pct(po_same)} send all their invoices down the same road. Looking at each later invoice against the PO's first one:</p>
<ul class="tight">
  <li>First invoice went to <code>CTRL1</code>: {pct(after_ctrl)} of the later invoices go there too.</li>
  <li>First invoice went to <code>WAITGR</code>: only {pct(after_wait)} of the later invoices go to <code>CTRL1</code>.</li>
</ul>
<p>So the trigger is something fixed on the PO or the supplier, not something that varies from invoice to invoice.</p>
</div>

<h2 id="value"><span class="step">4</span>PO value: a gradual slope, no threshold</h2>
<div class="grid2">
<div><h4>All invoices, by PO value (NOK)</h4>{table(HEAD, share_rows(vb))}</div>
<div><h4>Within the same supplier (mixed group), by value quarter</h4>{table(HEAD, share_rows(wq))}</div>
</div>
<div class="prose"><p>Large POs go to <code>CTRL1</code> more often, from {pct(vb['mean'].iloc[0])} below 100k NOK to {pct(vb['mean'].iloc[-1])} above 10M. But the share rises step by step, with no cut-off that would point to a fixed amount limit. Within the same supplier the effect is small ({pct(wq['mean'].iloc[0])} → {pct(wq['mean'].iloc[-1])}), so most of the value pattern comes from which suppliers have the large POs. A value limit could still exist, but this log can't isolate it, because the value is only known per PO.</p></div>

<h2 id="other"><span class="step">5</span>Other candidates</h2>
<div class="grid2">
<div><h4>Order of the invoice on its PO</h4>{table(HEAD, share_rows(pos))}</div>
<div><h4>Clearing amount vs invoice amount</h4>{table(HEAD, share_rows(dfc))}</div>
</div>
<div class="prose">
<p><b>Later invoices</b> on a PO go to <code>CTRL1</code> more often than the first one. That could fit partial deliveries whose quantities no longer match the PO, but it's also partly the supplier effect: POs with many invoices come from the "always <code>CTRL1</code>" suppliers more often.</p>
<p><b>Amount differences don't explain it.</b> If price differences sent invoices to the queue, invoices whose clearing amount differs from the invoice amount should go there more often. The opposite is true. The differences are mostly exchange-rate effects.</p>
</div>
<div class="grid2">
<div><h4>Weekday of validation</h4>{table(HEAD, share_rows(wd))}</div>
<div><h4>Month of validation (≥ 50 invoices)</h4>{table(HEAD, share_rows(mo))}</div>
</div>
<p class="prose">Timing varies but shows no consistent pattern. The share falls through the autumn, which matches the change in the case population at the end of the period more than a rule change.</p>

<h2 id="model"><span class="step">6</span>How predictable is the road?</h2>
<div class="prose"><p>A machine-learning model trained on the fields above predicts the road for {pct(acc_all)} of invoices on POs it hasn't seen (AUC {auc_all:.2f}, where 0.5 is a coin flip). The supplier alone reaches {auc_so:.2f}; all other fields without the supplier only {auc_ns:.2f}. The decision follows fixed attributes, so it's very likely a configured rule, not a random or clerk-by-clerk choice. Setup, validation, field importance and robustness checks are in the separate report <a href="report_znb1_ctrl1_model.html">How predictable is the manual queue?</a></p></div>

<h2 id="next"><span class="step">Interpretation</span>Likely reasons and what to check</h2>
<div class="prose">
<p>The pattern points to a setting on the supplier or on the PO, not to a check on each invoice. These are typical SAP setups that would produce it. They are hypotheses that the log can't confirm:</p>
</div>
<ol class="rec">
  <li><b>A flag on the supplier's master record</b>, e.g. a payment block or a "check invoices manually" setting that sends all of that supplier's invoices to a clerk.</li>
  <li><b>POs not set up for automatic matching against the goods receipt</b>, e.g. manual one-off orders without the goods-receipt-based invoice verification flag. The invoice then can't wait for the goods receipt and goes straight to the queue.</li>
  <li><b>Price or quantity tolerances</b> that hand-entered manual orders keep exceeding, especially for some suppliers and on later partial invoices.</li>
</ol>
<div class="callout">
<p><b>Next step:</b> ask the AP team or SAP support what is set differently for the "always <code>CTRL1</code>" suppliers ({", ".join(always_big)}) compared with Supplier115 and Supplier44. The blocking reasons and messages on a sample of their invoices would give the actual rule.</p>
</div>

<footer>
<p>Built with Python (conda env <code>datasci</code>) by <code>08_report_ctrl1_drivers.py</code> from <code>output/events.pkl</code> via <code>ctrl1_data.py</code> (run <code>01_prepare.py</code> first). Model figures come from <code>output/ctrl1_model_results.csv</code>, written by <code>09_report_ctrl1_model.py</code>.</p>
</footer>
</div>
"""
(OUT / "report_znb1_ctrl1.html").write_text(page, encoding="utf-8")
print("wrote", OUT / "report_znb1_ctrl1.html", len(page) // 1024, "KB")
print(cls)
