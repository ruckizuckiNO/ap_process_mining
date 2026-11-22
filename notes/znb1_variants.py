"""Variant analysis for ZNB1: which activity paths do invoices follow, and which paths cost the 2% discount?

Variant = the invoice's (PO_ID x SAP_PO_No) activity sequence, events ordered by ts then Event_ID, repeated
consecutive activities collapsed. Every ZNB1 invoice counts for the variant volumes. Lateness and missed discount use
the headline scope (ZNB1 POs without never-posted invoices, GR clock, invoice value = PO value / posted invoices,
late = posted > 10 days, missed discount = 2% of late invoice value). This is the scope of the always-manual notes
(9.8M NOK for those 25 suppliers), so the total is 13.8M NOK; the 15.9M headline in 04_prioritise.py keeps all POs.
Variants are grouped into families (see family()); the long tail is shown per family.
-> notes/znb1_variants.csv (per variant), notes/znb1_variant_families.csv, notes/znb1_variant_drivers.csv,
   notes/figures/znb1_variants_pareto.png, notes/figures/znb1_variants_cost.png,
   notes/znb1_variants_manual_post.csv + notes/figures/znb1_variants_manual_post.png (variants with a manual posting),
   notes/znb1_variants_top3_days.csv + notes/figures/znb1_variants_top3_days.png (top 3 by days band)
Run from the repo root:  PYTHONPATH=. conda run -n datasci python -W ignore notes/znb1_variants.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import POSTING_ACTIVITIES, SLA_HARD_DAYS, SLA_SOFT_DAYS, load_cases, load_events, load_invoices

NOTES = Path(__file__).resolve().parent
FIG = NOTES / "figures"
K = ["PO_ID", "SAP_PO_No"]
TOP = 12

BLUE, ORANGE, MUTED = "#2a78d6", "#eb6834", "#c9c8c2"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"

SHORT = {
    "Data management": "Data mgmt", "Initial validation": "Validation", "Wait for goods receipt": "Wait GR",
    "Posted manually": "Manual post", "MM Posted Automatically": "Auto post", "Invoice document deleted": "Deleted",
    "Posting in progress": "In progress", "MM Autopost after manual process": "Auto post after fix",
    "Sent for approval": "Approval", "Approved MM (accept if price diff)": "Approved (price)",
    "Approved MM": "Approved", "Approval rejected": "Rejected", "Get goods receipt from WM": "GR from WM",
    "Transfer to MM": "To MM", "Restarted process": "Restart", "Auto posting failed": "Auto post failed",
}
REWORK = {"Restarted process", "Transfer to MM", "Get goods receipt from WM", "Transfered to new Sequence number",
          "Auto posting failed", "Transfer to FI", "FI for Manual processing after initial check",
          "MM for Manual processing after initial check"}
APPROVAL = {"Sent for approval", "Approved MM (accept if price diff)", "Approved MM", "Approval rejected", "Approved",
            "Asked for Comment"}
FAMILIES = ["Wait GR, auto post", "Wait GR, manual post", "Direct manual post", "Auto post after manual step",
            "Approval loop", "Rework (WM / restart / transfer)", "Deleted", "Not posted"]


def family(acts):
    s = set(acts)
    if "Invoice document deleted" in s:
        return "Deleted"
    if not s & set(POSTING_ACTIVITIES):
        return "Not posted"
    if s & REWORK or acts.count("Initial validation") > 1:
        return "Rework (WM / restart / transfer)"
    if s & APPROVAL:
        return "Approval loop"
    if "MM Autopost after manual process" in s or "Posting in progress" in s:
        return "Auto post after manual step"
    if "MM Posted Automatically" in s:
        return "Wait GR, auto post"
    if "Wait for goods receipt" in s:
        return "Wait GR, manual post"
    return "Direct manual post"


# --- variants per invoice ---
ev = load_events()
z = ev[ev.SAP_PO_Type_Code == "ZNB1"].sort_values(["ts", "Event_ID"])
z = z[z.Activity_ID != z.groupby(K).Activity_ID.shift()]  # collapse self-loops
acts = z.groupby(K).Activity_ID.agg(list)
t = pd.DataFrame({"acts": acts})
t["variant"] = t.acts.map(" > ".join)
t["family"] = t.acts.map(family)
t["supplier"] = z.groupby(K).Supplier_ID.first()
post = z[z.Activity_ID.isin(POSTING_ACTIVITIES)].groupby(K).first()
t["poster"] = post.User_ID.reindex(t.index)

# --- headline discount scope ---
inv = load_invoices().set_index(K)
bad_pos = inv[(inv.po_type == "ZNB1") & ~inv.posted].index.get_level_values("PO_ID").unique()
t["in_scope"] = ~t.index.get_level_values("PO_ID").isin(bad_pos)
t["posted"] = inv.posted.reindex(t.index).fillna(False).astype(bool)
t["days"] = inv.time_to_post_days.reindex(t.index)
t["late"] = t.in_scope & t.posted & (t.days > SLA_SOFT_DAYS)
po_value = load_cases().set_index("PO_ID").invoice_value_nok
t["po_value"] = po_value.reindex(t.index.get_level_values("PO_ID")).values
n_posted = (t.posted & t.in_scope).groupby(level="PO_ID").transform("sum").clip(lower=1)
t["inv_value"] = t.po_value / n_posted * (t.posted & t.in_scope)
t["missed_discount"] = 0.02 * t.inv_value * t.late
headline = t.missed_discount.sum()

manual = pd.read_excel(NOTES / "znb1_always_manual_suppliers.xlsx", sheet_name="Always manual", header=3).Supplier
manual = set(manual[manual.astype(str).str.startswith("Supplier")])
t["always_manual_supplier"] = t.supplier.isin(manual)
t["by_user1341"] = t.poster == "User1341"


def summarise(g):
    s = g.in_scope & g.posted
    return pd.Series({
        "invoices": len(g),
        "invoices_in_discount_scope": int(s.sum()),
        "late_invoices": int(g.late.sum()),
        "late_pct": 100 * g.late.sum() / s.sum() if s.sum() else np.nan,
        "median_days_to_post": g.days[s].median(),
        "invoice_value_nok": g.inv_value.sum(),
        "missed_discount_nok": g.missed_discount.sum(),
        "suppliers": g.supplier.nunique(),
        "always_manual_supplier_pct": 100 * g.always_manual_supplier.mean(),
        "posted_by_user1341_pct": 100 * g.by_user1341.sum() / g.posted.sum() if g.posted.sum() else np.nan,
    })


var = t.groupby("variant").apply(summarise).sort_values("invoices", ascending=False)
var.insert(0, "family", t.groupby("variant").family.first().reindex(var.index))
var.insert(1, "steps", var.index.str.count(" > ") + 1)
var["share_pct"] = 100 * var.invoices / len(t)
var["cum_share_pct"] = var.share_pct.cumsum()
var["missed_discount_share_pct"] = 100 * var.missed_discount_nok / headline
var.insert(0, "rank", range(1, len(var) + 1))
var.insert(1, "label", [" > ".join(SHORT.get(a, a) for a in v.split(" > ")) for v in var.index])
var.to_csv(NOTES / "znb1_variants.csv")

fam = t.groupby("family").apply(summarise).reindex(FAMILIES)
fam.insert(1, "variants", t.groupby("family").variant.nunique().reindex(FAMILIES))
fam["share_pct"] = 100 * fam.invoices / len(t)
fam["missed_discount_share_pct"] = 100 * fam.missed_discount_nok / headline
fam.to_csv(NOTES / "znb1_variant_families.csv")

# Drivers of the costly direct manual path: suppliers and posting clerks
d = t[t.family == "Direct manual post"]
drv = pd.concat({
    "supplier": d.groupby("supplier").missed_discount.sum(),
    "posting_clerk": d.groupby("poster").missed_discount.sum(),
}, names=["by", "id"]).rename("missed_discount_nok").reset_index()
drv["share_of_path_pct"] = 100 * drv.missed_discount_nok / d.missed_discount.sum()
drv = drv.sort_values(["by", "missed_discount_nok"], ascending=[True, False]).groupby("by").head(10)
drv.to_csv(NOTES / "znb1_variant_drivers.csv", index=False)

# --- charts ---
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"]})


def style(ax, grid="y"):
    ax.set_facecolor(SURFACE)
    ax.grid(axis=grid, color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(axis="both", length=0, colors=INK2, labelsize=8.5)
    for s in ax.spines.values():
        s.set_visible(False)


# 1) Pareto of variants by invoice count (one % axis)
top = var.head(TOP)
other = var.iloc[TOP:]
shares = list(top.share_pct) + [other.share_pct.sum()]
cum = np.cumsum(shares)
fig, ax = plt.subplots(figsize=(13.5, 6.6), dpi=200)
fig.patch.set_facecolor(SURFACE)
style(ax)
x = np.arange(len(shares))
cols = [BLUE if c <= 80 or i == 0 or cum[i - 1] < 80 else MUTED for i, c in enumerate(cum)]
cols[-1] = MUTED
ax.bar(x, shares, width=0.72, color=cols, edgecolor=SURFACE, linewidth=2, zorder=2)
ax.plot(x, cum, color=INK, linewidth=2, marker="o", markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.5,
        zorder=3)
ax.axhline(80, color=INK2, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
ax.text(len(x) - 0.6, 81.5, "80%", color=INK2, fontsize=8.5, ha="right")
for i, s in enumerate(shares):
    if s > 10:
        ax.text(i, s - 2.5, f"{s:.0f}%", ha="center", va="top", fontsize=9, color="white")
    else:
        ax.text(i, s + 1.2, f"{s:.0f}%" if s >= 1 else f"{s:.1f}%", ha="center", fontsize=8.5, color=INK)
for i in (2, 9):
    ax.text(i + 0.12, cum[i] + 2.5, f"{cum[i]:.0f}%", fontsize=8.5, color=INK, ha="left")
labels = [f"#{r}  {l.replace(' > ', chr(10))}" for r, l in zip(top["rank"], top.label)]
labels.append(f"{len(other)} other\nvariants")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=7.2, color=INK2)
ax.set_ylim(0, 105)
ax.set_ylabel("Share of ZNB1 invoices (%)", color=INK2, fontsize=9.5)
n_80 = int((var.cum_share_pct < 80).sum()) + 1
fig.suptitle(f"{len(var)} ZNB1 variants, but {n_80} paths cover 80% of invoices", x=0.06, ha="left", y=0.97,
             fontsize=15, fontweight="bold", color=INK)
fig.text(0.06, 0.915, f"{len(t):,} invoices · bars = share per variant (blue = vital few up to 80%) · "
         "line = cumulative share", fontsize=9.5, color=INK2, ha="left", va="top")
fig.text(0.06, 0.01, "Variant = activity sequence per invoice (PO_ID x SAP_PO_No), ordered by timestamp then Event_ID, "
         "repeated steps collapsed.", fontsize=7.5, color=INK2, ha="left")
fig.subplots_adjust(left=0.06, right=0.98, top=0.85, bottom=0.25)
fig.savefig(FIG / "znb1_variants_pareto.png", facecolor=SURFACE)
plt.close(fig)

# 2) Cost per family: three small-multiple panels sharing the rows
f2 = fam[fam.invoices > 0]
y = np.arange(len(f2))[::-1]
fig, axes = plt.subplots(1, 3, figsize=(13, 5.6), dpi=200, sharey=True, gridspec_kw={"wspace": 0.12})
fig.patch.set_facecolor(SURFACE)
panels = [
    ("Invoices", f2.invoices, lambda v: f"{v:,.0f}", MUTED),
    ("Late (>10 days), % of posted in scope", f2.late_pct, lambda v: "–" if np.isnan(v) else f"{v:.0f}%", ORANGE),
    ("Missed 2% discount (M NOK)", f2.missed_discount_nok / 1e6, lambda v: f"{v:.2f}M", BLUE),
]
for ax, (title, vals, fm, col) in zip(axes, panels):
    style(ax, grid="x")
    v = vals.fillna(0).values
    ax.barh(y, v, height=0.62, color=col, edgecolor=SURFACE, linewidth=2, zorder=2)
    for yy, vv, raw in zip(y, v, vals.values):
        ax.text(vv + v.max() * 0.02, yy, fm(raw), va="center", fontsize=8.5, color=INK)
    ax.set_xlim(0, v.max() * 1.3)
    ax.set_title(title, loc="left", fontsize=10, color=INK, fontweight="bold")
    ax.set_xticks([])
axes[0].set_yticks(y)
axes[0].set_yticklabels([f"{n}  ({int(k)} variant{'s' if k > 1 else ''})" for n, k in zip(f2.index, f2.variants)], fontsize=9, color=INK)
dm = fam.loc["Direct manual post"]
fig.suptitle(f"The direct manual path carries {dm.missed_discount_share_pct:.0f}% of the missed discount",
             x=0.02, ha="left", y=0.97, fontsize=15, fontweight="bold", color=INK)
fig.text(0.02, 0.905, f"ZNB1 variants grouped into families · {headline / 1e6:.1f}M NOK missed discount in total · "
         f"direct manual: {dm.always_manual_supplier_pct:.0f}% of invoices from the 25 always-manual suppliers, "
         f"{dm.posted_by_user1341_pct:.0f}% posted by User1341", fontsize=9.5, color=INK2, ha="left", va="top")
fig.text(0.02, 0.01, "Late and discount: ZNB1 POs without never-posted invoices, GR clock, invoice value = PO value / "
         "posted invoices (13.8M NOK total; the 15.9M headline also keeps POs with never-posted invoices). Deleted and not-posted invoices have no discount.", fontsize=7.5,
         color=INK2, ha="left")
fig.subplots_adjust(left=0.2, right=0.98, top=0.8, bottom=0.08)
fig.savefig(FIG / "znb1_variants_cost.png", facecolor=SURFACE)
plt.close(fig)

# 3) Only variants that contain a manual posting: same three panels, one row per variant
TOP_M = 10
mv = var[var.index.str.contains("Posted manually")]
mtop = mv.head(TOP_M).copy()
mrest = mv.iloc[TOP_M:]
r = t[t.variant.isin(mrest.index)]
mtop.loc["__other__"] = summarise(r)
mtop.loc["__other__", "label"] = f"{len(mrest)} other manual-post variants"
y = np.arange(len(mtop))[::-1]
fig, axes = plt.subplots(1, 3, figsize=(14, 7.4), dpi=200, sharey=True, gridspec_kw={"wspace": 0.1})
fig.patch.set_facecolor(SURFACE)
panels = [
    ("Invoices", mtop.invoices.astype(float), lambda v: f"{v:,.0f}", MUTED),
    ("Late (>10 days), % of posted in scope", mtop.late_pct.astype(float), lambda v: "–" if np.isnan(v) else f"{v:.0f}%", ORANGE),
    ("Missed 2% discount (M NOK)", mtop.missed_discount_nok.astype(float) / 1e6, lambda v: f"{v:.2f}M", BLUE),
]
for ax, (title, vals, fm, col) in zip(axes, panels):
    style(ax, grid="x")
    v = vals.fillna(0).values
    ax.barh(y, v, height=0.62, color=col, edgecolor=SURFACE, linewidth=2, zorder=2)
    for yy, vv, raw in zip(y, v, vals.values):
        ax.text(vv + v.max() * 0.02, yy, fm(raw), va="center", fontsize=8.5, color=INK)
    ax.set_xlim(0, v.max() * 1.3)
    ax.set_title(title, loc="left", fontsize=10, color=INK, fontweight="bold")
    ax.set_xticks([])
axes[0].set_yticks(y)
def wrap(l, width=48):
    lines, cur = [], ""
    for step in l.split(" > "):
        nxt = f"{cur} > {step}" if cur else step
        if len(nxt) > width and cur:
            lines.append(cur + " >")
            cur = step
        else:
            cur = nxt
    return "\n".join(lines + [cur])


axes[0].set_yticklabels([l if i == "__other__" else wrap(f"#{int(rk)}  {l}")
                         for i, l, rk in zip(mtop.index, mtop.label, mtop["rank"].fillna(0))], fontsize=8.5, color=INK)
m_inv, m_miss = mv.invoices.sum(), mv.missed_discount_nok.sum()
fig.suptitle(f"{len(mv)} ZNB1 variants include a manual posting; one path carries "
             f"{mv.missed_discount_nok.iloc[0] / m_miss:.0%} of their missed discount",
             x=0.02, ha="left", y=0.97, fontsize=15, fontweight="bold", color=INK)
fig.text(0.02, 0.915, f"Variants that contain 'Posted manually' · {m_inv:,.0f} invoices ({100 * m_inv / len(t):.0f}% of ZNB1) "
         f"· {m_miss / 1e6:.1f}M of {headline / 1e6:.1f}M NOK missed discount · # = rank among all {len(var)} variants",
         fontsize=9.5, color=INK2, ha="left", va="top")
fig.text(0.02, 0.01, "Late and discount: ZNB1 POs without never-posted invoices, GR clock, invoice value = PO value / "
         "posted invoices. Variant = activity sequence per invoice, repeated steps collapsed.", fontsize=7.5,
         color=INK2, ha="left")
fig.subplots_adjust(left=0.27, right=0.98, top=0.84, bottom=0.06)
fig.savefig(FIG / "znb1_variants_manual_post.png", facecolor=SURFACE)
plt.close(fig)
mv.to_csv(NOTES / "znb1_variants_manual_post.csv")

# 4) Top 3 variants: invoices and invoice value by days to posting (<=10, 10-30, >30)
BANDS = [f"≤{SLA_SOFT_DAYS} days", f"{SLA_SOFT_DAYS}–{SLA_HARD_DAYS} days", f">{SLA_HARD_DAYS} days"]
BAND_COLORS = [BLUE, ORANGE, "#a8360f"]
top3 = var.head(3)
b = t[t.variant.isin(top3.index) & t.in_scope & t.posted].copy()
b["band"] = pd.cut(b.days, [-np.inf, SLA_SOFT_DAYS, SLA_HARD_DAYS, np.inf], labels=BANDS)
cnt = b.pivot_table(index="variant", columns="band", values="days", aggfunc="size", observed=False).reindex(top3.index)
val = b.pivot_table(index="variant", columns="band", values="inv_value", aggfunc="sum", observed=False).reindex(top3.index)
out = pd.concat({"invoices": cnt, "invoice_value_nok": val}, axis=1)
out.columns = [f"{a} {c}" for a, c in out.columns]
out.insert(0, "label", top3.label)
out.insert(1, "all_invoices_in_variant", top3.invoices)
out["missed_discount_nok"] = top3.missed_discount_nok
out.to_csv(NOTES / "znb1_variants_top3_days.csv")

y = np.arange(3)[::-1]
fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), dpi=200, sharey=True, gridspec_kw={"wspace": 0.08})
fig.patch.set_facecolor(SURFACE)
for ax, data, title, fm in [
        (axes[0], cnt, "Number of invoices", lambda v: f"{v:,.0f}"),
        (axes[1], val / 1e6, "Invoice value (M NOK)", lambda v: f"{v:,.0f}M")]:
    style(ax, grid="x")
    tot = data.sum(axis=1).values
    left = np.zeros(3)
    for band, col in zip(BANDS, BAND_COLORS):
        v = data[band].values
        ax.barh(y, v, left=left, height=0.58, color=col, edgecolor=SURFACE, linewidth=2, zorder=2, label=band)
        for yy, l, vv, tt in zip(y, left, v, tot):
            if vv / tt >= 0.07:
                ax.text(l + vv / 2, yy, f"{fm(vv)}\n{vv / tt:.0%}", ha="center", va="center", fontsize=8.5,
                        color="white", zorder=3)
        left += v
    for yy, tt, row in zip(y, tot, data.itertuples(index=False)):
        small = [f"{fm(x)} ({x / tt:.1%}) {bd}" for x, bd in zip(row, BANDS) if 0 < x / tt < 0.07]
        ax.text(tt + tot.max() * 0.015, yy, f"Total {fm(tt)}" + ("\n" + "\n".join(small) if small else ""),
                va="center", fontsize=8.5, color=INK)
    ax.set_xlim(0, tot.max() * 1.32)
    ax.set_xticks([])
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, fontweight="bold")
axes[0].set_yticks(y)
axes[0].set_yticklabels([wrap(f"#{r}  {l}", 30) for r, l in zip(top3["rank"], top3.label)], fontsize=9.5, color=INK)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, [f"Posted {x} after clock start" if i == 0 else f"Posted {x}" for i, x in enumerate(l)],
           loc="upper left", bbox_to_anchor=(0.175, 0.875), ncol=3, frameon=False, fontsize=9, labelcolor=INK2,
           handlelength=1.6)
over30 = cnt[BANDS[2]].sum()
fig.suptitle(f"Path #1 is late in the 10–30 day window; path #3 has most of the {over30} invoices over 30 days",
             x=0.02, ha="left", y=0.97, fontsize=15, fontweight="bold", color=INK)
fig.text(0.02, 0.915, f"Days from clock start (goods receipt) to first posting · {len(b):,} posted invoices in scope · "
         "# = rank in the variant Pareto", fontsize=9.5, color=INK2, ha="left", va="top")
fig.text(0.02, 0.01, "Scope: ZNB1 POs without never-posted invoices (so counts are a little below the Pareto totals). "
         "Invoice value = PO value / posted invoices. ≤10 days earns the 2% discount.", fontsize=7.5, color=INK2,
         ha="left")
fig.subplots_adjust(left=0.175, right=0.98, top=0.78, bottom=0.07)
fig.savefig(FIG / "znb1_variants_top3_days.png", facecolor=SURFACE)
plt.close(fig)
print(out.round(0).to_string())

pd.set_option("display.width", 250)
print(f"invoices {len(t)}  variants {len(var)}  missed {headline:,.0f}  always-manual {t.missed_discount[t.always_manual_supplier].sum():,.0f}")
print(var[["label", "family", "invoices", "share_pct", "cum_share_pct", "late_pct", "median_days_to_post",
           "missed_discount_nok", "missed_discount_share_pct"]].head(15).round(1).to_string(index=False))
print(fam.round(1).to_string())
print(drv.round(1).to_string(index=False))
