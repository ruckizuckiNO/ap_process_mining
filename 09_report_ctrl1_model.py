"""Builds output/report_znb1_ctrl1_model.html: the machine-learning check of how predictable the CTRL1 routing of ZNB1 is.

Also writes output/ctrl1_model_results.csv, which 08_report_ctrl1_drivers.py reads for its summary. Run this script first.
Charts are inline SVG (no extra files).
"""
import html
import re

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import OUT, ROOT
from ctrl1_data import T, FEATURES, build, features

CSS = re.search(r"<style>(.*?)</style>", (ROOT / "05_report.py").read_text(), re.S).group(1)
CSS = CSS.replace("{{", "{").replace("}}", "}")
SEED = 0


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


# ---------------------------------------------------------------- data and models
_, _, v = build()
X, y = features(v), v.ctrl.to_numpy()
groups_po, groups_sup = v.PO_ID.to_numpy(), v.Supplier_ID.to_numpy()
n, base = len(v), y.mean()

ALL = list(FEATURES)
NO_SUP = [f for f in ALL if f != "supplier"]
# Fields known when the invoice is validated: amount_diff uses the clearing amount (after payment) and n_inv counts
# invoices that arrive later, so both leak future information
AT_VAL = [f for f in ALL if f not in ("amount_diff", "n_inv")]


def hgb(feats):
    return HistGradientBoostingClassifier(max_iter=200, categorical_features=[f == "supplier" for f in feats], random_state=SEED)


def logit(feats):
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


MODELS = {  # name: (estimator factory, features, short description)
    "Logistic regression, no supplier": (logit, NO_SUP, "Linear baseline on value, amount difference, invoice order and timing."),
    "Gradient boosting, no supplier": (hgb, NO_SUP, "Same fields, non-linear."),
    "Gradient boosting, supplier only": (hgb, ["supplier"], "Learns one CTRL1 rate per supplier."),
    "Gradient boosting, all fields": (hgb, ALL, "Main model: all eight fields."),
    "Gradient boosting, known at validation": (hgb, AT_VAL, "Drops the two fields that use information from after validation."),
}


def oof(make, feats, groups, n_splits=5):
    """Out-of-fold probabilities with folds grouped so that a group never appears in train and test at once."""
    p = np.zeros(n)
    for tr, te in GroupKFold(n_splits).split(X, y, groups):
        m = make(feats).fit(X.iloc[tr][feats], y[tr])
        p[te] = m.predict_proba(X.iloc[te][feats])[:, 1]
    return p


def scores(p, yy=None):
    yy = y if yy is None else yy
    pred = p > 0.5
    return dict(auc=roc_auc_score(yy, p), acc=(pred == yy).mean(), prec=precision_score(yy, pred),
                rec=recall_score(yy, pred), f1=f1_score(yy, pred))


P = {k: oof(make, feats, groups_po) for k, (make, feats, _) in MODELS.items()}
R = {k: scores(p) for k, p in P.items()}
main = "Gradient boosting, all fields"
cm = confusion_matrix(y, P[main] > 0.5)

# Unseen suppliers: folds grouped by supplier (the supplier field can't help here)
P_sup = {k: oof(MODELS[k][0], MODELS[k][1], groups_sup) for k in ["Gradient boosting, no supplier", main]}
R_sup = {k: scores(p) for k, p in P_sup.items()}

# Time split: train on invoices validated before 1 Oct 2025, test on the rest
cut = pd.Timestamp("2025-10-01")
tr, te = (v.ts < cut).to_numpy(), (v.ts >= cut).to_numpy()
R_time = {}
for k in ["Gradient boosting, no supplier", "Gradient boosting, supplier only", main]:
    make, feats, _ = MODELS[k]
    m = make(feats).fit(X[tr][feats], y[tr])
    R_time[k] = scores(m.predict_proba(X[te][feats])[:, 1], y[te])

# Permutation importance of the main model, averaged over the PO-grouped folds (drop in AUC when a field is shuffled)
imp = []
for tr_i, te_i in GroupKFold(5).split(X, y, groups_po):
    m = hgb(ALL).fit(X.iloc[tr_i], y[tr_i])
    r = permutation_importance(m, X.iloc[te_i], y[te_i], scoring="roc_auc", n_repeats=5, random_state=SEED)
    imp.append(r.importances_mean)
IMP = pd.Series(np.mean(imp, axis=0), index=ALL).sort_values(ascending=False)

# Drop-one ablation: AUC of the main model without each field
ABL = pd.Series({f: roc_auc_score(y, oof(hgb, [g for g in ALL if g != f], groups_po)) for f in ALL}).sort_values()

pd.DataFrame([{"model": k, "features": "+".join(MODELS[k][1]), **R[k]} for k in MODELS]).to_csv(OUT / "ctrl1_model_results.csv", index=False)

# ---------------------------------------------------------------- charts (inline SVG)
SERIES = {"Gradient boosting, no supplier": "var(--s1)", "Gradient boosting, supplier only": "var(--s2)", main: "var(--s3)"}


def roc_svg():
    W, H, L, Rm, Tm, B = 560, 400, 52, 16, 14, 46
    pw, ph = W - L - Rm, H - Tm - B
    sx = lambda f: L + f * pw
    sy = lambda t: Tm + (1 - t) * ph
    g = []
    for t in np.linspace(0, 1, 6):
        g.append(f'<line x1="{sx(0)}" x2="{sx(1)}" y1="{sy(t):.1f}" y2="{sy(t):.1f}" class="grid"/>'
                 f'<text x="{L - 8}" y="{sy(t) + 4:.1f}" class="tick" text-anchor="end">{t:.1f}</text>'
                 f'<text x="{sx(t):.1f}" y="{H - B + 18}" class="tick" text-anchor="middle">{t:.1f}</text>')
    g.append(f'<line x1="{sx(0)}" y1="{sy(0)}" x2="{sx(1)}" y2="{sy(1)}" class="diag"/>'
             f'<text x="{sx(0.62):.1f}" y="{sy(0.55):.1f}" class="tick" transform="rotate(-36 {sx(0.62):.1f} {sy(0.55):.1f})">coin flip (AUC 0.5)</text>')
    for k, col in SERIES.items():
        fpr, tpr, thr = roc_curve(y, P[k])
        idx = np.unique(np.r_[np.linspace(0, len(fpr) - 1, 300).astype(int), len(fpr) - 1])
        pts = " ".join(f"{sx(fpr[i]):.1f},{sy(tpr[i]):.1f}" for i in idx)
        g.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2" stroke-linejoin="round">'
                 f'<title>{html.escape(k)}: AUC {R[k]["auc"]:.3f}</title></polyline>')
        j = np.argmin(np.abs(thr - 0.5))
        g.append(f'<circle cx="{sx(fpr[j]):.1f}" cy="{sy(tpr[j]):.1f}" r="5" fill="{col}" stroke="var(--surface)" stroke-width="2">'
                 f'<title>{html.escape(k)} at threshold 0.5: {tpr[j] * 100:.0f}% of CTRL1 invoices caught, '
                 f'{fpr[j] * 100:.0f}% of WAITGR invoices wrongly flagged</title></circle>')
    g.append(f'<text x="{sx(0.5)}" y="{H - 6}" class="axis" text-anchor="middle">False-positive rate (WAITGR invoices flagged as CTRL1)</text>'
             f'<text x="14" y="{sy(0.5)}" class="axis" text-anchor="middle" transform="rotate(-90 14 {sy(0.5)})">True-positive rate (CTRL1 caught)</text>')
    legend = "".join(f'<span><i style="background:{c}"></i>{html.escape(k.replace("Gradient boosting, ", ""))} · AUC {R[k]["auc"]:.2f}</span>'
                     for k, c in SERIES.items())
    return (f'<div class="chart"><div class="legend">{legend}</div>'
            f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="ROC curves of three models">{"".join(g)}</svg></div>')


def bars_svg(s, fmt, label, unit_max=None):
    rowh, L, W = 30, 250, 640
    top = unit_max or max(s.max(), 1e-9)
    H = rowh * len(s) + 8
    g = []
    for i, (f, val_) in enumerate(s.items()):
        y0 = 4 + i * rowh
        w = max(val_, 0) / top * (W - L - 70)
        g.append(f'<text x="{L - 10}" y="{y0 + 18}" class="lab" text-anchor="end">{html.escape(FEATURES[f])}</text>'
                 f'<rect x="{L}" y="{y0 + 6}" width="{w:.1f}" height="16" rx="4" fill="var(--s1)"><title>{html.escape(FEATURES[f])}: {fmt(val_)}</title></rect>'
                 f'<text x="{L + w + 8:.1f}" y="{y0 + 18}" class="val">{fmt(val_)}</text>')
    return f'<div class="chart"><svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(label)}">{"".join(g)}</svg></div>'


# ---------------------------------------------------------------- tables
model_rows = [f"<td><b>{html.escape(k)}</b><br><small>{MODELS[k][2]}</small></td><td class='n'>{R[k]['auc']:.3f}</td>"
              f"<td class='n'>{pct(R[k]['acc'], 1)}</td><td class='n'>{pct(R[k]['prec'])}</td><td class='n'>{pct(R[k]['rec'])}</td>"
              f"<td class='n'>{R[k]['f1']:.2f}</td>" for k in MODELS]
model_rows.insert(0, f"<td><b>Always guess WAITGR</b><br><small>Majority class, no model.</small></td><td class='n'>0.500</td>"
                     f"<td class='n'>{pct(1 - base, 1)}</td><td class='n'>–</td><td class='n'>0%</td><td class='n'>–</td>")
feat_rows = [f"<td><code>{f}</code></td><td>{FEATURES[f]}</td><td>{'yes' if f in AT_VAL else '<b>no</b>'}</td>" for f in ALL]
val_rows = []
for k in ["Gradient boosting, no supplier", "Gradient boosting, supplier only", main]:
    val_rows.append(f"<td>{html.escape(k)}</td><td class='n'>{R[k]['auc']:.3f}</td>"
                    f"<td class='n'>{R_sup[k]['auc']:.3f}</td>" if k in R_sup else
                    f"<td>{html.escape(k)}</td><td class='n'>{R[k]['auc']:.3f}</td><td class='n'>–</td>")
    val_rows[-1] += f"<td class='n'>{R_time[k]['auc']:.3f}</td>"
abl_rows = [f"<td>{FEATURES[f]}</td><td class='n'>{a:.3f}</td><td class='n'>{a - R[main]['auc']:+.3f}</td>" for f, a in ABL.items()]
tn, fp, fn, tp = cm.ravel()
_rate = v.groupby("Supplier_ID").ctrl.transform("mean")
_mixed = ((_rate > 0.05) & (_rate <= 0.95)).to_numpy()
_err = (P[main] > 0.5) != y
err_mixed, inv_mixed = _err[_mixed].sum() / _err.sum(), _mixed.mean()
best_unseen = max(R_sup.values(), key=lambda r: r["auc"])["auc"]
cm_html = table(["", "Predicted WAITGR", "Predicted CTRL1"], [
    f"<td><b>Actual WAITGR</b></td><td class='n ok'>{num(tn)}</td><td class='n warn'>{num(fp)}</td>",
    f"<td><b>Actual CTRL1</b></td><td class='n warn'>{num(fn)}</td><td class='n ok'>{num(tp)}</td>"])

r_all, r_ns, r_so, r_at = R[main], R["Gradient boosting, no supplier"], R["Gradient boosting, supplier only"], R["Gradient boosting, known at validation"]
top2 = list(IMP.index[:2])

EXTRA_CSS = """
:root { --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; }
.chart { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 12px 14px; margin: 14px 0; max-width: 720px; }
.chart svg { display: block; width: 100%; height: auto; }
.chart .grid { stroke: var(--line); stroke-width: 1; }
.chart .diag { stroke: var(--muted); stroke-width: 1; stroke-dasharray: 4 4; }
.chart .tick { font: 11px var(--f-mono); fill: var(--muted); }
.chart .axis { font: 12px var(--f-body); fill: var(--muted); }
.chart .lab { font: 13px var(--f-body); fill: var(--ink); }
.chart .val { font: 12px var(--f-mono); fill: var(--ink); }
.legend { display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: 13px; margin-bottom: 6px; }
.legend i { display: inline-block; width: 14px; height: 3px; border-radius: 2px; margin-right: 6px; vertical-align: middle; }
"""

page = f"""<meta charset="utf-8">
<title>ZNB1 Routing Model</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,500;6..72,600&family=Instrument+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap">
<style>{CSS}{EXTRA_CSS}</style>

<div class="wrap">
<header>
  <div class="eyebrow">Accounts payable · Process mining case · ZNB1 routing · Machine learning</div>
  <h1>How predictable is the manual queue?</h1>
  <p class="lede">A machine-learning check of the routing at <i>Initial validation</i> for manual orders ({code(T)}): can the fields in the event log predict whether an invoice goes to the manual control queue (<code>CTRL1</code>) or to the goods-receipt check (<code>WAITGR</code>)? The descriptive findings are in the separate report <a href="report_znb1_ctrl1.html">Why do ZNB1 invoices go to the manual queue?</a></p>
  <div class="facts">
    <span><b>{num(n)}</b> invoices</span>
    <span><b>{pct(base)}</b> CTRL1</span>
    <span><b>{len(ALL)}</b> fields</span>
    <span><b>5-fold</b> CV grouped by PO</span>
  </div>
</header>

<h2 id="answer">Short answer</h2>
<div class="findings">
  <div class="finding"><div class="k">AUC {r_all['auc']:.2f}</div><p>The main model predicts the road correctly for {pct(r_all['acc'])} of invoices on POs it has never seen, against {pct(1 - base)} for always guessing <code>WAITGR</code>.</p></div>
  <div class="finding"><div class="k">AUC {r_so['auc']:.2f}</div><p>with the supplier alone. Without the supplier, all other fields together reach only {r_ns['auc']:.2f}. The supplier carries most of the signal.</p></div>
  <div class="finding"><div class="k">AUC {best_unseen:.2f}</div><p>at best for suppliers the model has never seen. The rule can't be inferred from value or timing; it has to be learned supplier by supplier.</p></div>
  <div class="finding"><div class="k">Not a cause</div><p>A good prediction shows the decision is systematic. It doesn't show why. The actual rule needs the SAP configuration.</p></div>
</div>

<h2 id="setup"><span class="step">1</span>Setup</h2>
<div class="prose">
<p><b>Question.</b> If the routing is a configured rule, the fields that drive it should predict it well. If it's random or depends on things outside the log, they won't. The model is a measuring tool for that question, not something to deploy.</p>
<p><b>Unit and target.</b> One row per ZNB1 invoice (<code>PO_ID</code> × <code>SAP_PO_No</code>). Target = 1 if the invoice's first <i>Initial validation</i> routes it to <code>CTRL1</code>, 0 if to <code>WAITGR</code>. The {pct(base)} share of CTRL1 is balanced enough to use plain accuracy next to AUC.</p>
</div>
<h3>Fields</h3>
{table(["Field", "Meaning", "Known at validation?"], feat_rows)}
<div class="prose">
<p>The SAP fields on the validation event itself (processing type, user, object code) are the same for every invoice, so they carry no information and are left out. <code>amount_diff</code> uses the clearing amount, which is only known after payment, and <code>n_inv</code> counts invoices that arrive later. Both are kept to measure the pattern, and a separate model without them shows that they don't matter (section 2). <code>log_value</code> is the PO total, which can also include later invoices.</p>
<p><b>Models.</b> A logistic regression as a linear baseline, and scikit-learn's <code>HistGradientBoostingClassifier</code> (200 iterations, supplier as a categorical field) for non-linear patterns and interactions.</p>
<p><b>Validation.</b> 5-fold cross-validation with folds <b>grouped by PO</b>. Invoices on the same PO almost always take the same road, so letting one PO's invoices sit in both training and test data would overstate the result. All numbers below are out-of-fold: each invoice is predicted by a model that never saw its PO.</p>
<p><b>Metrics.</b> AUC (0.5 = coin flip, 1 = perfect ranking). Accuracy, precision (of the invoices predicted CTRL1, how many really went there), recall (of the CTRL1 invoices, how many were caught) and F1, all at a 0.5 threshold.</p>
</div>

<h2 id="results"><span class="step">2</span>Results</h2>
{table(["Model", "AUC", "Accuracy", "Precision", "Recall", "F1"], model_rows)}
{roc_svg()}
<p class="prose" style="font-size:13px;color:var(--muted)">ROC curves of three gradient-boosting models (out-of-fold). Dots mark the 0.5 threshold. Hover a line or dot for its numbers.</p>
<div class="callout">
<p><b>The supplier does most of the work.</b> The supplier alone gets AUC {r_so['auc']:.2f}; adding all other fields only lifts it to {r_all['auc']:.2f}. Without the supplier, the logistic regression reaches {R['Logistic regression, no supplier']['auc']:.2f} and gradient boosting {r_ns['auc']:.2f}.</p>
<p><b>No leakage effect.</b> With only the fields known at validation the model still reaches AUC {r_at['auc']:.2f} (vs {r_all['auc']:.2f}), so the two fields that use later information don't drive the result.</p>
</div>
<h3>Errors of the main model</h3>
{cm_html}
<p class="prose">At the 0.5 threshold the model catches {pct(tp / (tp + fn))} of the CTRL1 invoices and wrongly flags {pct(fp / (fp + tn))} of the WAITGR invoices. {pct(err_mixed)} of the errors come from the "mixed" suppliers (5–95% CTRL1), whose invoices go both ways, although they have only {pct(inv_mixed)} of the invoices.</p>

<h2 id="importance"><span class="step">3</span>Which fields matter?</h2>
<div class="prose"><p><b>Permutation importance:</b> how much the AUC of the main model drops when one field is shuffled, averaged over the five folds. A large drop means the model relies on that field.</p></div>
{bars_svg(IMP, lambda x: f"−{x:.3f}", "Permutation importance (drop in AUC)")}
<div class="prose"><p><b>Drop-one check:</b> the AUC when the model is retrained without a field. This catches fields that matter but overlap with others, which permutation importance can understate.</p></div>
{table(["Field left out", "AUC without it", "Change vs all fields"], abl_rows)}
<div class="callout"><p><b>The supplier dominates.</b> Shuffling it costs {IMP['supplier']:.2f} AUC; every other field costs less than {IMP.drop('supplier').max() + 0.005:.2f}. Retraining without the supplier costs {ABL['supplier'] - r_all['auc']:+.3f}, without any other single field at most {ABL.drop('supplier').min() - r_all['auc']:+.3f}.</p></div>

<h2 id="robust"><span class="step">4</span>Robustness</h2>
<div class="prose">
<p><b>Unseen suppliers.</b> Folds grouped by supplier, so every test supplier is new to the model. <b>Time split.</b> Trained on invoices validated before 1 October 2025 ({num(tr.sum())}), tested on the later ones ({num(te.sum())}).</p>
</div>
{table(["Model", "AUC: grouped by PO", "AUC: unseen suppliers", "AUC: time split"], val_rows)}
<div class="callout">
<p><b>For a new supplier the routing can't be predicted well:</b> AUC {R_sup['Gradient boosting, no supplier']['auc']:.2f} without the supplier field and {R_sup[main]['auc']:.2f} with it (an unknown supplier code only confuses the model). Value, invoice order and timing don't add up to a general rule. That fits a setting stored per supplier or per PO, not a rule on the invoice amount.</p>
<p><b>The pattern holds over time</b> (AUC {R_time[main]['auc']:.2f} on the later months), so the supplier effect is stable, not a one-off.</p>
</div>

<h2 id="limits"><span class="step">5</span>What this does and doesn't show</h2>
<ul class="tight prose">
  <li><b>Prediction isn't explanation.</b> The model shows that the routing follows the supplier. It doesn't show which setting causes it. Candidates (supplier master flags, POs not set up for matching against the goods receipt, tolerance limits) are in the drivers report.</li>
  <li><b>The value is per PO</b>, not per invoice, so a per-invoice amount limit can't be tested properly.</li>
  <li><b>The population changes</b> towards the end of the period (fewer, slower cases from October). The time split checks this but can't rule it out.</li>
  <li><b>One model family, default settings.</b> No hyperparameter tuning; the point is the size of the signal, not the best possible score.</li>
</ul>

<footer>
<p>Built with Python (conda env <code>datasci</code>, scikit-learn) by <code>09_report_ctrl1_model.py</code> from <code>output/events.pkl</code>; the dataset comes from <code>ctrl1_data.py</code>, shared with the drivers report. Results table: <code>output/ctrl1_model_results.csv</code>. Random seed {SEED}.</p>
</footer>
</div>
"""
(OUT / "report_znb1_ctrl1_model.html").write_text(page, encoding="utf-8")
print("wrote", OUT / "report_znb1_ctrl1_model.html", len(page) // 1024, "KB")
for k in MODELS:
    print(f"{k:42} AUC {R[k]['auc']:.3f} acc {R[k]['acc']:.3f}")
print("unseen suppliers:", {k: round(r['auc'], 3) for k, r in R_sup.items()})
print("time split:", {k: round(r['auc'], 3) for k, r in R_time.items()})
print("importance:", IMP.round(3).to_dict())
print("ablation:", ABL.round(3).to_dict())
