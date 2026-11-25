"""Builds output/report_po_cut_gbr.html: how many invoices would still be late (> 10 days) if the ZNB1 100%-manual
suppliers' POs up to a PO-value cut (1.5 / 1.75 / 2.0 / 2.25M NOK) were handled like normal orders (ZJD1).

Method: a gradient-boosting regression (scikit-learn HistGradientBoostingRegressor, squared loss on the 0/1 target
"posted > 10 days", so the prediction is the expected late share; monotone: the late share may only rise with the PO
value) is trained on all posted ZJD1 invoices with the PO value as the only input. An unconstrained
GradientBoostingRegressor is kept as a comparison. It is then applied to the ZNB1 invoices below each cut to estimate how many would still be
late and how much of the 2% discount would still be missed. 5-fold CV grouped by PO, calibration check, bootstrap
over POs for intervals, and two sensitivities (arrival clock; ZJD1 invoices on the automatic WAITGR road only).

Writes output/po_cut_gbr_results.csv. Charts are inline SVG. Needs scikit-learn.
Run from the repo root:  conda run -n datasci python -W ignore 13_report_po_cut_gbr.py
"""
import html
import re

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from common import OUT, POSTING_ACTIVITIES, ROOT, SLA_SOFT_DAYS, load_cases, load_events, load_invoices

CSS = re.search(r"<style>(.*?)</style>", (ROOT / "05_report.py").read_text(), re.S).group(1)
CSS = CSS.replace("{{", "{").replace("}}", "}")
SEED = 0
K = ["PO_ID", "SAP_PO_No"]
DISCOUNT = 0.02
CUTS = [1.5e6, 1.75e6, 2.0e6, 2.25e6]
N_BOOT = 100
PARAMS = dict(max_iter=300, learning_rate=0.03, max_depth=3, min_samples_leaf=300)  # main model, monotone increasing
FREE = dict(n_estimators=500, learning_rate=0.02, max_depth=3, min_samples_leaf=200, subsample=0.8)  # comparison


def main_model():
    return HistGradientBoostingRegressor(loss="squared_error", monotonic_cst=[1], random_state=SEED, **PARAMS)


def free_model():
    return GradientBoostingRegressor(loss="squared_error", random_state=SEED, **FREE)


MAIN = "Main: monotone, 300 trees, depth 3, leaf ≥ 300"
FREE_LAB = "Unconstrained: 500 trees, depth 3, leaf ≥ 200"
TUNING = [  # (label, factory) compared in the CV table
    ("Very simple: 200 stumps", lambda: GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=1, min_samples_leaf=200, subsample=0.8, random_state=SEED)),
    (FREE_LAB, free_model),
    (MAIN, main_model),
    ("Overfit: 1,000 trees, depth 3, leaf ≥ 20", lambda: GradientBoostingRegressor(n_estimators=1000, learning_rate=0.1, max_depth=3, min_samples_leaf=20, subsample=0.8, random_state=SEED)),
]


def pct(x, d=0):
    return "–" if pd.isna(x) else f"{x * 100:.{d}f}%"


def num(x, d=0):
    return "–" if pd.isna(x) else f"{x:,.{d}f}"


def mnok(x, d=2):
    return f"{x / 1e6:,.{d}f}M"


def code(c):
    return f'<span class="code">{c}</span>'


def table(head, rows, cls=""):
    th = "".join(f"<th>{h}</th>" for h in head)
    body = "".join(f"<tr>{r}</tr>" for r in rows)
    return f'<div class="tbl {cls}"><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def cut_label(c):
    return f"{c / 1e6:g}M NOK"


# ---------------------------------------------------------------- data
ev = load_events()
ev = ev[ev.SAP_PO_Type_Code.isin(["ZJD1", "ZNB1"])].sort_values(["PO_ID", "ts", "Event_ID"])
ev["nxt"] = ev.groupby(K).Activity_ID.shift(-1)
iv = ev[ev.Activity_ID == "Initial validation"].groupby(K)
route, nxt = iv.SAP_Next_Processing_Type.first(), iv.nxt.first()
supplier = ev.groupby(K).Supplier_ID.first()
po_value = load_cases().set_index("PO_ID").invoice_value_nok

inv = load_invoices()
inv = inv[inv.po_type.isin(["ZJD1", "ZNB1"])].set_index(K)
inv["route"], inv["supplier"] = route, supplier
inv["po_value"] = po_value.reindex(inv.index.get_level_values(0)).values
# Clock as in the ZNB1 pipeline: from the first WAIT_GR event (goods receipt) where there is one, else arrival.
inv["days_gr"] = (inv.first_post - inv.goods_receipt.fillna(inv.start)).dt.total_seconds() / 86400
inv["days_arr"] = (inv.first_post - inv.start).dt.total_seconds() / 86400

# Training data: posted ZJD1 invoices with a positive PO value
J = inv[(inv.po_type == "ZJD1") & inv.posted & (inv.po_value > 0)].reset_index()
X = np.log10(J.po_value.to_numpy()).reshape(-1, 1)
y = (J.days_gr > SLA_SOFT_DAYS).astype(float).to_numpy()
y_arr = (J.days_arr > SLA_SOFT_DAYS).astype(float).to_numpy()
groups = J.PO_ID.to_numpy()
n_zero = int(((inv.po_type == "ZJD1") & inv.posted & (inv.po_value <= 0)).sum())
waitgr = (J.route == "WAITGR").to_numpy()

# Application data: ZNB1 100%-manual suppliers, all posted invoices (same scope as notes/znb1_ctrl1_100pct_exposure.py)
Z = inv[inv.po_type == "ZNB1"].copy()
share = Z[Z.route.notna()].groupby("supplier").route.agg(lambda x: (x == "CTRL1").mean())
s100 = share.index[share == 1]
direct = Z.posted & nxt.reindex(Z.index).isin(POSTING_ACTIVITIES)
s100 = s100.intersection(Z.loc[direct, "supplier"].unique())
Z["n_posted"] = Z.groupby(level="PO_ID").posted.transform("sum")
Z["amount"] = np.where(Z.posted, Z.po_value / Z.n_posted.where(Z.n_posted > 0), 0)
Z = Z[Z.supplier.isin(s100) & Z.posted].copy()
Z["late"] = Z.time_to_post_days > SLA_SOFT_DAYS  # today: pipeline clock (arrival; 2 invoices have a WAIT_GR step)
Z["missed"] = DISCOUNT * Z.amount * Z.late
Z_total = Z.missed.sum()
Z_pos = Z.index.get_level_values("PO_ID").nunique()


# ---------------------------------------------------------------- models
def oof(make, target, mask=None):
    """Out-of-fold predictions, folds grouped by PO so a PO's invoices are never in train and test at once."""
    idx = np.arange(len(target)) if mask is None else np.flatnonzero(mask)
    p = np.full(len(target), np.nan)
    for tr, te in GroupKFold(5).split(idx, target[idx], groups[idx]):
        m = make().fit(X[idx[tr]], target[idx[tr]])
        p[idx[te]] = np.clip(m.predict(X[idx[te]]), 0, 1)
    return p


def score(p, target, mask=None):
    mk = ~np.isnan(p) if mask is None else mask
    bc = brier_score_loss(target[mk], np.full(mk.sum(), target[mk].mean()))
    b = brier_score_loss(target[mk], p[mk])
    return dict(auc=roc_auc_score(target[mk], p[mk]), brier=b, skill=1 - b / bc, brier_const=bc)


# Decile baseline: observed late share per PO-value decile (out-of-fold, same folds)
def oof_deciles(target):
    p = np.zeros(len(target))
    for tr, te in GroupKFold(5).split(X, target, groups):
        edges = np.unique(np.quantile(X[tr, 0], np.linspace(0, 1, 11)))
        btr = np.clip(np.searchsorted(edges, X[tr, 0], side="right") - 1, 0, len(edges) - 2)
        bte = np.clip(np.searchsorted(edges, X[te, 0], side="right") - 1, 0, len(edges) - 2)
        rate = pd.Series(target[tr]).groupby(btr).mean()
        p[te] = rate.reindex(bte).fillna(target[tr].mean()).to_numpy()
    return p


CV = {lab: score(oof(mk, y), y) for lab, mk in TUNING}
CV_DEC = score(oof_deciles(y), y)
P_OOF = oof(main_model, y)
main_cv, free_cv = CV[MAIN], CV[FREE_LAB]

# Calibration: out-of-fold prediction deciles vs observed
cal = pd.DataFrame({"p": P_OOF, "y": y})
cal["bin"] = pd.qcut(cal.p.rank(method="first"), 10, labels=False)
cal = cal.groupby("bin").agg(n=("y", "size"), pred=("p", "mean"), obs=("y", "mean"))
cal_gap = (cal.pred - cal.obs).abs().max()
cal_over, cal_under = int((cal.pred > cal.obs).sum()), int((cal.pred < cal.obs).sum())

# Final models on all data (main + sensitivities)
M_MAIN = main_model().fit(X, y)
M_FREE = free_model().fit(X, y)
M_ARR = main_model().fit(X, y_arr)
M_WGR = main_model().fit(X[waitgr], y[waitgr])
CV_ARR = score(oof(main_model, y_arr), y_arr)
P_WGR = oof(main_model, y, waitgr)
CV_WGR = score(P_WGR, y, waitgr)

GRID = np.linspace(4, 8, 161)  # log10 PO value, 10k .. 100M NOK
curve = np.clip(M_MAIN.predict(GRID.reshape(-1, 1)), 0, 1)
curve_free = np.clip(M_FREE.predict(GRID.reshape(-1, 1)), 0, 1)

# Bootstrap over ZJD1 POs: refit, predict the curve and the ZNB1 invoices
zx = np.log10(Z.po_value.to_numpy()).reshape(-1, 1)
rng = np.random.default_rng(SEED)
po_ids = np.unique(groups)
rows_by_po = pd.Series(np.arange(len(groups))).groupby(groups).apply(np.array)
boot_curve, boot_z = [], []
for _ in range(N_BOOT):
    pick = rng.choice(po_ids, len(po_ids), replace=True)
    idx = np.concatenate(rows_by_po.loc[pick].to_numpy())
    m = main_model().fit(X[idx], y[idx])
    boot_curve.append(np.clip(m.predict(GRID.reshape(-1, 1)), 0, 1))
    boot_z.append(np.clip(m.predict(zx), 0, 1))
boot_curve, boot_z = np.array(boot_curve), np.array(boot_z)
band_lo, band_hi = np.percentile(boot_curve, [5, 95], axis=0)

Z["p_main"] = np.clip(M_MAIN.predict(zx), 0, 1)
Z["p_arr"] = np.clip(M_ARR.predict(zx), 0, 1)
Z["p_wgr"] = np.clip(M_WGR.predict(zx), 0, 1)
Z["p_free"] = np.clip(M_FREE.predict(zx), 0, 1)

# ---------------------------------------------------------------- per cut
res = []
for c in CUTS:
    b = (Z.po_value <= c).to_numpy()
    zb = Z[b]
    still = lambda p: (p * DISCOUNT * zb.amount).sum()
    boot_share = boot_z[:, b].mean(axis=1)
    boot_still = (boot_z[:, b] * DISCOUNT * zb.amount.to_numpy()).sum(axis=1)
    res.append(dict(
        cut=c, pos=zb.index.get_level_values("PO_ID").nunique(), invoices=len(zb), amount=zb.amount.sum(),
        late_today=int(zb.late.sum()), late_today_share=zb.late.mean(), missed_today=zb.missed.sum(),
        exp_late=zb.p_main.sum(), exp_late_share=zb.p_main.mean(),
        exp_late_share_lo=np.percentile(boot_share, 5), exp_late_share_hi=np.percentile(boot_share, 95),
        still_missed=still(zb.p_main), still_lo=np.percentile(boot_still, 5), still_hi=np.percentile(boot_still, 95),
        exp_late_share_arr=zb.p_arr.mean(), still_missed_arr=still(zb.p_arr),
        exp_late_share_wgr=zb.p_wgr.mean(), still_missed_wgr=still(zb.p_wgr),
        exp_late_share_free=zb.p_free.mean(), still_missed_free=still(zb.p_free),
        above=Z_total - zb.missed.sum(),
    ))
R = pd.DataFrame(res)
R["recovered"] = R.missed_today - R.still_missed
R["recovered_arr"] = R.missed_today - R.still_missed_arr
R["recovered_wgr"] = R.missed_today - R.still_missed_wgr
R["recovered_free"] = R.missed_today - R.still_missed_free
R["after"] = R.above + R.still_missed
R.to_csv(OUT / "po_cut_gbr_results.csv", index=False)

# Observed ZJD1 late share per cut band (for the text)
J["late"] = y
obs_below = {c: J.loc[J.po_value <= c, "late"].mean() for c in CUTS}


# ---------------------------------------------------------------- charts (inline SVG)
def curve_svg():
    W, H, L, Rm, Tm, B = 680, 380, 56, 18, 16, 48
    pw, ph = W - L - Rm, H - Tm - B
    x0, x1, ymax = GRID[0], GRID[-1], 0.30
    sx = lambda v: L + (v - x0) / (x1 - x0) * pw
    sy = lambda t: Tm + (1 - min(t, ymax) / ymax) * ph
    g = []
    # cut zone
    g.append(f'<rect x="{sx(np.log10(CUTS[0])):.1f}" y="{Tm}" width="{sx(np.log10(CUTS[-1])) - sx(np.log10(CUTS[0])):.1f}" '
             f'height="{ph}" class="zone"><title>Cut range 1.5–2.25M NOK</title></rect>')
    for t in np.arange(0, ymax + 1e-9, 0.05):
        g.append(f'<line x1="{L}" x2="{L + pw}" y1="{sy(t):.1f}" y2="{sy(t):.1f}" class="grid"/>'
                 f'<text x="{L - 8}" y="{sy(t) + 4:.1f}" class="tick" text-anchor="end">{t * 100:.0f}%</text>')
    for e, lab in [(4, "10k"), (5, "100k"), (6, "1M"), (7, "10M"), (8, "100M")]:
        g.append(f'<line x1="{sx(e):.1f}" x2="{sx(e):.1f}" y1="{Tm}" y2="{Tm + ph}" class="grid"/>'
                 f'<text x="{sx(e):.1f}" y="{H - B + 18}" class="tick" text-anchor="middle">{lab}</text>')
    # bootstrap band
    pts = [f"{sx(v):.1f},{sy(t):.1f}" for v, t in zip(GRID, band_hi)] + \
          [f"{sx(v):.1f},{sy(t):.1f}" for v, t in zip(GRID[::-1], band_lo[::-1])]
    g.append(f'<polygon points="{" ".join(pts)}" class="band"><title>90% bootstrap band ({N_BOOT} refits on resampled POs)</title></polygon>')
    # observed: 20 quantile bins of ZJD1 PO value
    q = pd.qcut(X[:, 0], 20, duplicates="drop")
    ob = pd.DataFrame({"x": X[:, 0], "y": y}).groupby(q, observed=True).agg(x=("x", "median"), y=("y", "mean"), n=("y", "size"))
    for r in ob.itertuples():
        g.append(f'<circle cx="{sx(r.x):.1f}" cy="{sy(r.y):.1f}" r="4.5" class="obs"><title>Observed ZJD1: PO value ≈ '
                 f'{10 ** r.x / 1e6:,.2f}M NOK · {r.y * 100:.1f}% late · {r.n:,} invoices</title></circle>')
    line = " ".join(f"{sx(v):.1f},{sy(t):.1f}" for v, t in zip(GRID, curve))
    line_free = " ".join(f"{sx(v):.1f},{sy(t):.1f}" for v, t in zip(GRID, curve_free))
    g.append(f'<polyline points="{line_free}" fill="none" stroke="var(--muted)" stroke-width="1.2" stroke-linejoin="round" opacity=".7">'
             f'<title>Unconstrained gradient boosting (comparison)</title></polyline>')
    g.append(f'<polyline points="{line}" fill="none" stroke="var(--s1)" stroke-width="2.5" stroke-linejoin="round">'
             f'<title>Monotone gradient-boosting estimate of the late share (main model)</title></polyline>')
    for c in CUTS:
        t = float(np.clip(M_MAIN.predict([[np.log10(c)]]), 0, 1)[0])
        g.append(f'<line x1="{sx(np.log10(c)):.1f}" x2="{sx(np.log10(c)):.1f}" y1="{Tm}" y2="{Tm + ph}" class="cut"/>'
                 f'<circle cx="{sx(np.log10(c)):.1f}" cy="{sy(t):.1f}" r="3.5" fill="var(--ink)"><title>Cut {cut_label(c)}: '
                 f'model late share at this PO value {t * 100:.1f}%</title></circle>')
    g.append(f'<text x="{sx(np.log10(CUTS[0])) + 4:.1f}" y="{Tm + 14}" class="tick">cuts</text>')
    g.append(f'<text x="{L + pw / 2}" y="{H - 8}" class="axis" text-anchor="middle">PO value (NOK, log scale)</text>'
             f'<text x="14" y="{Tm + ph / 2}" class="axis" text-anchor="middle" transform="rotate(-90 14 {Tm + ph / 2})">'
             f'Share of invoices posted &gt; {SLA_SOFT_DAYS} days</text>')
    legend = ('<span><i style="background:var(--s1)"></i>Gradient boosting, monotone (main)</span>'
              '<span><i style="background:var(--muted)"></i>Unconstrained (comparison)</span>'
              '<span><i class="dot" style="background:var(--s2)"></i>Observed ZJD1 (20 equal-size bins)</span>'
              '<span><i class="sq" style="background:var(--band)"></i>90% bootstrap band</span>')
    return (f'<div class="chart"><div class="legend">{legend}</div>'
            f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Late share versus PO value for ZJD1">{"".join(g)}</svg></div>')


def cal_svg():
    W, H, L, Rm, Tm, B = 420, 340, 52, 16, 14, 46
    pw, ph = W - L - Rm, H - Tm - B
    top = np.ceil(max(cal.pred.max(), cal.obs.max()) * 1.1 * 20) / 20  # round up to 5%
    sx = lambda t: L + t / top * pw
    sy = lambda t: Tm + (1 - t / top) * ph
    g = []
    for t in np.arange(0, top + 1e-9, 0.05):
        g.append(f'<line x1="{L}" x2="{L + pw}" y1="{sy(t):.1f}" y2="{sy(t):.1f}" class="grid"/>'
                 f'<text x="{L - 8}" y="{sy(t) + 4:.1f}" class="tick" text-anchor="end">{t * 100:.0f}%</text>'
                 f'<text x="{sx(t):.1f}" y="{H - B + 18}" class="tick" text-anchor="middle">{t * 100:.0f}%</text>')
    g.append(f'<line x1="{sx(0)}" y1="{sy(0)}" x2="{sx(top):.1f}" y2="{sy(top):.1f}" class="diag"/>')
    for r in cal.itertuples():
        g.append(f'<circle cx="{sx(r.pred):.1f}" cy="{sy(r.obs):.1f}" r="5" class="obs"><title>Predicted {r.pred * 100:.1f}% · '
                 f'observed {r.obs * 100:.1f}% · {r.n:,} invoices</title></circle>')
    g.append(f'<text x="{L + pw / 2}" y="{H - 6}" class="axis" text-anchor="middle">Predicted late share (out-of-fold)</text>'
             f'<text x="14" y="{Tm + ph / 2}" class="axis" text-anchor="middle" transform="rotate(-90 14 {Tm + ph / 2})">Observed late share</text>')
    return (f'<div class="chart narrow"><svg viewBox="0 0 {W} {H}" role="img" aria-label="Calibration plot">{"".join(g)}</svg></div>')


def stack_svg():
    W, L, Rm, rowh = 720, 120, 20, 46
    H = rowh * len(R) + 40
    pw = W - L - Rm
    sx = lambda v: v / Z_total * pw
    g = []
    for i, r in enumerate(R.itertuples()):
        y0 = 8 + i * rowh
        segs = [(r.recovered, "var(--s1)", "Recovered: discount claimed after the change", "w"),
                (r.still_missed, "var(--s2)", "Still missed below the cut (model estimate)", "w"),
                (r.above, "var(--grey)", "Above the cut: stays manual, unchanged", "k")]
        x = L
        for v, col, lab, tc in segs:
            w = sx(v)
            g.append(f'<rect x="{x:.1f}" y="{y0}" width="{max(w - 2, 0):.1f}" height="28" rx="3" fill="{col}">'
                     f'<title>{cut_label(r.cut)} · {lab}: {mnok(v)} NOK ({v / Z_total * 100:.1f}%)</title></rect>')
            if w > 44:
                g.append(f'<text x="{x + w / 2:.1f}" y="{y0 + 18.5}" class="seg {tc}" text-anchor="middle">{mnok(v)}</text>')
            x += w
        g.append(f'<text x="{L - 10}" y="{y0 + 19}" class="lab" text-anchor="end">Cut {cut_label(r.cut)}</text>')
    for t in np.arange(0, Z_total / 1e6 + 0.01, 2):
        xx = L + t * 1e6 / Z_total * pw
        g.append(f'<text x="{xx:.1f}" y="{H - 8}" class="tick" text-anchor="middle">{t:.0f}M</text>')
    legend = ('<span><i class="sq" style="background:var(--s1)"></i>Recovered (claimed after the change)</span>'
              '<span><i class="sq" style="background:var(--s2)"></i>Still missed below the cut (model)</span>'
              '<span><i class="sq" style="background:var(--grey)"></i>Above the cut (stays manual)</span>')
    return (f'<div class="chart"><div class="legend">{legend}</div>'
            f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Missed discount split per cut">{"".join(g)}</svg></div>')


# ---------------------------------------------------------------- tables
tun_rows = [f"<td>Always predict the average ({pct(y.mean(), 1)})</td><td class='n'>0.500</td>"
            f"<td class='n'>{main_cv['brier_const']:.4f}</td><td class='n'>0.0%</td>",
            f"<td>Late share per PO-value decile (lookup table)</td><td class='n'>{CV_DEC['auc']:.3f}</td>"
            f"<td class='n'>{CV_DEC['brier']:.4f}</td><td class='n'>{pct(CV_DEC['skill'], 1)}</td>"]
tun_rows += [f"<td>{'<b>' if lab == MAIN else ''}Gradient boosting · {html.escape(lab)}{'</b>' if lab == MAIN else ''}</td>"
             f"<td class='n'>{CV[lab]['auc']:.3f}</td><td class='n'>{CV[lab]['brier']:.4f}</td><td class='n'>{pct(CV[lab]['skill'], 1)}</td>"
             for lab, _ in TUNING]
cal_rows = [f"<td class='n'>{i + 1}</td><td class='n'>{num(r.n)}</td><td class='n'>{pct(r.pred, 1)}</td><td class='n'>{pct(r.obs, 1)}</td>"
            for i, r in enumerate(cal.itertuples())]
cut_rows = [f"<td><b>{cut_label(r.cut)}</b></td><td class='n'>{num(r.pos)}</td><td class='n'>{num(r.invoices)}</td>"
            f"<td class='n'>{num(r.late_today)} ({pct(r.late_today_share)})</td><td class='n'>{mnok(r.missed_today)}</td>"
            f"<td class='n'><b>{num(r.exp_late)} ({pct(r.exp_late_share, 1)})</b><br><small>{pct(r.exp_late_share_lo, 1)}–{pct(r.exp_late_share_hi, 1)}</small></td>"
            f"<td class='n'>{mnok(r.still_missed)}<br><small>{mnok(r.still_lo)}–{mnok(r.still_hi)}</small></td>"
            f"<td class='n'><b>{mnok(r.recovered)}</b></td>" for r in R.itertuples()]
tot_rows = [f"<td><b>{cut_label(r.cut)}</b></td><td class='n'>{mnok(Z_total)}</td><td class='n'>{mnok(r.recovered)} ({pct(r.recovered / Z_total)})</td>"
            f"<td class='n'>{mnok(r.still_missed)}</td><td class='n'>{mnok(r.above)}</td><td class='n'><b>{mnok(r.after)}</b> ({pct(r.after / Z_total)})</td>"
            for r in R.itertuples()]
sens_rows = [f"<td><b>{cut_label(r.cut)}</b></td>"
             f"<td class='n'>{pct(r.exp_late_share, 1)} · {mnok(r.recovered)}</td>"
             f"<td class='n'>{pct(r.exp_late_share_wgr, 1)} · {mnok(r.recovered_wgr)}</td>"
             f"<td class='n'>{pct(r.exp_late_share_free, 1)} · {mnok(r.recovered_free)}</td>"
             f"<td class='n'>{pct(r.exp_late_share_arr, 1)} · {mnok(r.recovered_arr)}</td>" for r in R.itertuples()]

r15, r225 = R.iloc[0], R.iloc[-1]


def span(a, b, d=0):
    """'x–y' for a range, or just 'x' when both ends round to the same value."""
    return pct(a, d) if pct(a, d) == pct(b, d) else f"{pct(a, d)}–{pct(b, d)}"
lift = curve[np.searchsorted(GRID, np.log10(2.25e6))] / curve[np.searchsorted(GRID, np.log10(2e5))]

EXTRA_CSS = """
:root { --s1: #2a78d6; --s2: #eb6834; --grey: #c9c8c2; --band: rgba(42,120,214,.16); }
.chart { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 12px 14px; margin: 14px 0; max-width: 760px; }
.chart.narrow { max-width: 460px; }
.chart svg { display: block; width: 100%; height: auto; }
.chart .grid { stroke: var(--line); stroke-width: 1; }
.chart .diag { stroke: var(--muted); stroke-width: 1; stroke-dasharray: 4 4; }
.chart .cut { stroke: var(--ink); stroke-width: 1; stroke-dasharray: 3 3; opacity: .55; }
.chart .zone { fill: var(--line); opacity: .45; }
.chart .band { fill: var(--band); stroke: none; }
.chart .obs { fill: var(--s2); stroke: var(--surface); stroke-width: 1.5; }
.chart .tick { font: 11px var(--f-mono); fill: var(--muted); }
.chart .axis { font: 12px var(--f-body); fill: var(--muted); }
.chart .lab { font: 13px var(--f-body); fill: var(--ink); }
.chart .seg { font: 600 12px var(--f-mono); }
.chart .seg.w { fill: #fff; } .chart .seg.k { fill: var(--ink); }
.legend { display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: 13px; margin-bottom: 6px; }
.legend i { display: inline-block; width: 14px; height: 3px; border-radius: 2px; margin-right: 6px; vertical-align: middle; }
.legend i.dot { width: 9px; height: 9px; border-radius: 50%; }
.legend i.sq { width: 12px; height: 12px; border-radius: 3px; }
td small { color: var(--muted); font-size: 11px; }
"""

page = f"""<meta charset="utf-8">
<title>PO-Value Cut: Late-Share Model</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,500;6..72,600&family=Instrument+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap">
<style>{CSS}{EXTRA_CSS}</style>

<div class="wrap">
<header>
  <div class="eyebrow">Accounts payable · Process mining case · ZNB1 100%-manual suppliers · Machine learning</div>
  <h1>If small POs ran like normal orders, how many would still be late?</h1>
  <p class="lede">The PO-value cut assumes that every invoice on a ZNB1 PO at or below the cut is handled automatically and posted within {SLA_SOFT_DAYS} days. Normal orders ({code('ZJD1')}) show that this is too optimistic: some of their invoices are late too, more often on larger POs. A gradient-boosting regression trained on ZJD1 estimates the share that would still be late at each cut, and how much of the 2% discount would really be recovered.</p>
  <div class="facts">
    <span><b>{num(len(J))}</b> ZJD1 invoices to learn from</span>
    <span><b>{pct(y.mean(), 1)}</b> of them late</span>
    <span><b>{num(len(Z))}</b> ZNB1 invoices to apply it to</span>
    <span><b>{mnok(Z_total)}</b> NOK missed today</span>
  </div>
</header>

<h2 id="answer">Short answer</h2>
<div class="findings">
  <div class="finding"><div class="k">{pct(r15.exp_late_share, 1)}–{pct(r225.exp_late_share, 1)}</div><p>of the invoices below the cut would still be posted after {SLA_SOFT_DAYS} days if they ran like ZJD1 (1.5M to 2.25M cut), against {span(r15.late_today_share, r225.late_today_share)} today.</p></div>
  <div class="finding"><div class="k">{mnok(r15.recovered)}–{mnok(r225.recovered)}</div><p>NOK of discount recovered, instead of the {mnok(r15.missed_today)}–{mnok(r225.missed_today)} the simple cut assumes. About {pct(r15.still_missed / r15.missed_today)}–{pct(r225.still_missed / r225.missed_today)} of it would still be lost.</p></div>
  <div class="finding"><div class="k">{pct(r225.after / Z_total)}</div><p>of today's {mnok(Z_total)} NOK would still be missed even at the 2.25M cut. The money sits on the large POs above the cut, which stay manual.</p></div>
  <div class="finding"><div class="k">AUC {main_cv['auc']:.2f}</div><p>PO value alone ranks lateness only weakly, but the predicted shares match the observed ones on average, and the expected <i>share</i> is what the estimate needs.</p></div>
</div>

<h2 id="setup"><span class="step">1</span>Setup</h2>
<div class="prose">
<p><b>Question.</b> For each cut (1.5, 1.75, 2.0 and 2.25M NOK), how many of the ZNB1 invoices at or below the cut would still be posted after more than {SLA_SOFT_DAYS} days if they were processed like normal orders, and how much of the 2% discount would they still miss?</p>
<p><b>Learning data.</b> All {num(len(J))} posted {code('ZJD1')} invoices (one row per <code>PO_ID</code> × <code>SAP_PO_No</code>) on {num(len(np.unique(groups)))} POs, July 2025 to March 2026. ZJD1 is the normal-order process with the same goods-receipt road that the cut would open for ZNB1. All ZJD1 invoices count, including the {pct(1 - waitgr.mean())} that ZJD1 itself sends to the manual queue: that is how normal orders behave today. {n_zero} invoices with a PO value of 0 are left out.</p>
<p><b>Target.</b> 1 if the invoice was first posted more than {SLA_SOFT_DAYS} days after the clock start, else 0. The clock is the ZNB1 pipeline rule: it starts at goods receipt (the first <code>WAIT_GR</code> event) where there is one, else at arrival. An automated ZNB1 invoice would go through that wait and fall under the same rule. The arrival clock is a sensitivity in section 4.</p>
<p><b>Input.</b> Only the PO value (<code>Case_Invoice_Value_NOK</code>, log10), as the cut is a rule on the PO value.</p>
<p><b>Model.</b> Gradient-boosting regression (scikit-learn <code>HistGradientBoostingRegressor</code>) with squared loss on the 0/1 target. With that loss the prediction is the expected late share at a given PO value, i.e. a probability, which is exactly what the estimate needs. Settings: {PARAMS['max_iter']} trees, depth {PARAMS['max_depth']}, learning rate {PARAMS['learning_rate']}, at least {PARAMS['min_samples_leaf']} invoices per leaf. Predictions are clipped to 0–100%.</p>
<p><b>Monotone constraint.</b> The model may only let the late share rise (or stay flat) as the PO value grows. Without it, gradient boosting follows the noise: the unconstrained version (grey line in section 2) jumps between about 3% and 25% for neighbouring PO values. The constraint fits the data (larger POs are checked more and wait longer) and costs almost nothing in accuracy (section 2).</p>
<p><b>Validation.</b> 5-fold cross-validation with folds <b>grouped by PO</b>, so that invoices on the same PO are never in training and test at once. Uncertainty: {N_BOOT} refits on ZJD1 POs drawn with replacement (bootstrap), giving 90% intervals.</p>
<p><b>Application.</b> The {num(len(Z))} posted invoices on {num(Z_pos)} POs of the {len(s100)} ZNB1 suppliers that go 100% to the manual queue (deleted and never-posted invoices left out), as in the exposure and cut charts. Today they are posted from the manual queue, measured with the pipeline clock (from arrival, as almost none has a goods-receipt wait), and miss {mnok(Z_total)} NOK. For each invoice below a cut the model gives its probability of still being late; still-missed discount = Σ probability × 2% × invoice amount (PO value split evenly over the PO's posted invoices).</p>
</div>

<h2 id="model"><span class="step">2</span>What ZJD1 shows</h2>
{curve_svg()}
<p class="prose" style="font-size:13px;color:var(--muted)">Late share of ZJD1 invoices by PO value. Dots: observed share in 20 equal-size bins. Blue line: monotone gradient-boosting estimate (main model) with its 90% bootstrap band. Grey line: unconstrained gradient boosting. Grey zone and dashed lines: the four cuts. Hover for numbers.</p>
<div class="callout">
<p><b>Lateness rises with PO value.</b> Around 200k NOK about {pct(curve[np.searchsorted(GRID, np.log10(2e5))], 1)} of ZJD1 invoices are late; at 2.25M the estimate is {pct(curve[np.searchsorted(GRID, np.log10(2.25e6))], 1)}, about {lift:.1f}× as much. Below the cuts the observed ZJD1 late share is {pct(obs_below[CUTS[0]], 1)} (≤ 1.5M) to {pct(obs_below[CUTS[-1]], 1)} (≤ 2.25M).</p>
</div>
<h3>Cross-validation</h3>
{table(["Model", "AUC", "Brier score", "Brier skill vs average"], tun_rows)}
<div class="prose">
<p><b>Brier score</b> = mean squared error of the predicted probability (lower is better). <b>Brier skill</b> = improvement over always predicting the average late share. <b>AUC</b> = how well the model ranks late above on-time invoices (0.5 = coin flip).</p>
<p>The monotone main model beats the average and the decile lookup table (Brier {main_cv['brier']:.4f} vs {CV_DEC['brier']:.4f}). The unconstrained model is slightly better (Brier {free_cv['brier']:.4f}, AUC {free_cv['auc']:.2f}), a difference of {pct((main_cv['brier'] - free_cv['brier']) / free_cv['brier'], 1)} in the error, but its curve is not believable, so the smooth monotone model is used. The skill is small ({pct(main_cv['skill'], 1)}) and the AUC modest ({main_cv['auc']:.2f}): PO value explains only part of why one invoice is late and another isn't. The overfit setting is <i>worse</i> than predicting the average, which shows why the leaves are kept large.</p>
</div>
<h3>Calibration</h3>
<div class="prose"><p>For a share estimate the predicted probabilities have to be right on average, not just well ranked. Out-of-fold predictions, grouped in ten equal-size bins, against what actually happened:</p></div>
{cal_svg()}
{table(["Bin", "Invoices", "Predicted late", "Observed late"], cal_rows)}
<p class="prose">Overall the model predicts {pct(P_OOF.mean(), 2)} late against {pct(y.mean(), 2)} observed. Per bin the gap is at most {pct(cal_gap, 1)} points, and it goes both ways ({cal_over} bins over, {cal_under} under), so there is no systematic bias. The calibration is reasonable rather than exact: the largest miss is the top bin (the largest POs, above the cuts), where the model overstates lateness a little. Summed over many invoices, the shares give a usable expected number of late invoices; for a single invoice they are rough.</p>

<h2 id="cuts"><span class="step">3</span>Results per cut</h2>
{table(["Cut", "POs ≤ cut", "Invoices ≤ cut", "Late today", "Missed today", "Expected late like ZJD1<br><small>90% interval</small>", "Still missed (model)<br><small>90% interval</small>", "Recovered"], cut_rows)}
{stack_svg()}
<p class="prose" style="font-size:13px;color:var(--muted)">Today's {mnok(Z_total)} NOK missed discount of the 100%-manual suppliers, split per cut. Hover for numbers.</p>
{table(["Cut", "Missed today", "Recovered", "Still missed ≤ cut", "Above cut (unchanged)", "Missed after the change"], tot_rows)}
<div class="callout">
<p><b>The model trims the simple estimate by {pct(r15.still_missed / r15.missed_today)}–{pct(r225.still_missed / r225.missed_today)}.</b> At the 1.5M cut, {mnok(r15.missed_today)} NOK is missed on POs below the cut today. If they ran like ZJD1, about {pct(r15.exp_late_share, 1)} of those invoices would still be late, so {mnok(r15.still_missed)} would still be missed and {mnok(r15.recovered)} recovered. At 2.25M: {mnok(r225.recovered)} recovered, {mnok(r225.still_missed)} still missed.</p>
<p><b>The cut alone moves little money.</b> Even at 2.25M, {mnok(r225.after)} NOK ({pct(r225.after / Z_total)}) of today's loss remains, almost all of it on large POs above the cut ({mnok(r225.above)}). Raising the cut from 1.5M to 2.25M adds only {mnok(r225.recovered - r15.recovered)} NOK of recovered discount.</p>
</div>

<h2 id="robust"><span class="step">4</span>Sensitivity</h2>
<div class="prose">
<p><b>Automatic road only.</b> The model retrained on the {num(waitgr.sum())} ZJD1 invoices routed to <code>WAITGR</code> (out-of-fold AUC {CV_WGR['auc']:.2f}). This is the road an automated ZNB1 invoice would take; the main model also includes ZJD1's own manual share.</p>
<p><b>Unconstrained model.</b> The plain <code>GradientBoostingRegressor</code> without the monotone constraint (grey line in section 2). It fits the data a little better in cross-validation but swings up and down with the noise.</p>
<p><b>Arrival clock.</b> The model retrained with the clock from arrival in AP for every invoice (AUC {CV_ARR['auc']:.2f}). This counts the wait for the goods as late time too, so it is an upper bound on lateness, in line with the project rule to show the arrival clock next to the goods-receipt headline.</p>
</div>
{table(["Cut", "Main (all ZJD1, GR clock)<br><small>late share · recovered</small>", "ZJD1 WAITGR road only<br><small>late share · recovered</small>", "Unconstrained model<br><small>late share · recovered</small>", "Arrival clock<br><small>late share · recovered</small>"], sens_rows)}
<div class="callout"><p><b>The clock matters more than the model.</b> On the automatic road the share still late is lower and the recovery higher. With the arrival clock, waiting for the goods counts too, so far more invoices would still be late and the recovery drops to {mnok(r15.recovered_arr)}–{mnok(r225.recovered_arr)} NOK. The goods-receipt clock is the project's headline; the arrival clock is the pessimistic case.</p></div>

<h2 id="limits"><span class="step">5</span>What this does and doesn't show</h2>
<ul class="tight prose">
  <li><b>ZJD1 as a stand-in.</b> The estimate assumes that a ZNB1 invoice below the cut would behave like a ZJD1 invoice with the same PO value. Different suppliers, goods and order types may behave differently.</li>
  <li><b>One input.</b> Only PO value is used, as the cut is a PO-value rule. The weak AUC shows that most of the variation in lateness comes from other things (supplier, goods receipt timing, workload), so single-invoice predictions are rough; the summed shares are what is reliable.</li>
  <li><b>Goods-receipt proxy.</b> The log has no GR date; the first <code>WAIT_GR</code> event stands in for it, as in the rest of the project. For the ZNB1 invoices it is assumed that the clock would start there once they ran through the goods-receipt check.</li>
  <li><b>Value per PO, not per invoice.</b> The cut and the model use the PO total; the discount uses the PO value split evenly over its posted invoices.</li>
  <li><b>The large POs are untouched.</b> The model says nothing about speeding up the manual queue for POs above the cut, where most of the money is.</li>
</ul>

<footer>
<p>Built with Python (conda env <code>datasci</code>, scikit-learn) by <code>13_report_po_cut_gbr.py</code> from <code>output/events.pkl</code>, <code>invoices.csv</code> and <code>cases.csv</code>. Results table: <code>output/po_cut_gbr_results.csv</code>. Random seed {SEED}, {N_BOOT} bootstrap refits.</p>
</footer>
</div>
"""
(OUT / "report_po_cut_gbr.html").write_text(page, encoding="utf-8")
print("wrote", OUT / "report_po_cut_gbr.html", len(page) // 1024, "KB")
print({k: {m: round(v, 4) for m, v in d.items()} for k, d in CV.items()}, "deciles", CV_DEC)
print("cal gap", cal_gap, "WGR", CV_WGR, "ARR", CV_ARR)
pd.set_option("display.width", 250)
print(R.round(4).T.to_string())
