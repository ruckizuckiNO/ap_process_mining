"""Potential of automating the ZNB1 always-manual suppliers up to a PO-value threshold (default 1.5M NOK).

Scenario: every invoice of the 25 always-manual suppliers (notes/znb1_always_manual_suppliers.xlsx) on a PO with
PO value <= THRESHOLD is handled automatically and posted within 10 days, so the 2% discount on those invoices that
are late today can be claimed. POs above the threshold stay manual and keep today's result.

Same method and scope as znb1_always_manual_suppliers.py: ZNB1 POs without never-posted invoices; invoice value =
PO value split evenly over the PO's posted invoices; late = posted > 10 days after clock start (GR clock).
-> notes/znb1_auto_threshold_potential.csv (per supplier), notes/znb1_auto_threshold_sensitivity.csv,
   notes/figures/znb1_auto_threshold_potential.png
Run from the repo root:  PYTHONPATH=. conda run -n datasci python -W ignore notes/znb1_auto_threshold_potential.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from common import SLA_SOFT_DAYS, load_cases, load_events, load_invoices

NOTES = Path(__file__).resolve().parent
K = ["PO_ID", "SAP_PO_No"]
THRESHOLD = 1.5e6
SENSITIVITY = [0.25e6, 0.5e6, 1e6, 1.5e6, 2.5e6, 5e6, 10e6, float("inf")]

BLUE, MUTED = "#2a78d6", "#c9c8c2"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"

suppliers = pd.read_excel(NOTES / "znb1_always_manual_suppliers.xlsx", sheet_name="Always manual", header=3).Supplier
suppliers = set(suppliers[suppliers.astype(str).str.startswith("Supplier")])

# --- invoice table, identical to znb1_always_manual_suppliers.py ---
ev = load_events()
z = ev[ev.SAP_PO_Type_Code == "ZNB1"]
po_value = load_cases().set_index("PO_ID").invoice_value_nok
inv = load_invoices().set_index(K)
bad_pos = inv[(inv.po_type == "ZNB1") & ~inv.posted].index.get_level_values("PO_ID").unique()
z = z[~z.PO_ID.isin(bad_pos)]
inv = inv[~inv.index.get_level_values("PO_ID").isin(bad_pos)]
iv = z[z.Activity_ID == "Initial validation"].sort_values("ts").drop_duplicates(K).set_index(K)
t = pd.DataFrame({"supplier": iv.Supplier_ID, "direct": iv.SAP_Next_Processing_Type == "CTRL1"})
t["po_value"] = po_value.reindex(t.index.get_level_values("PO_ID")).values
t["posted"] = inv.posted.reindex(t.index).fillna(False).astype(bool)
t["days"] = inv.time_to_post_days.reindex(t.index)
t["late"] = t.posted & (t.days > SLA_SOFT_DAYS)
n_posted = t.groupby(level="PO_ID").posted.transform("sum").clip(lower=1)
t["inv_value"] = t.po_value / n_posted * t.posted
t["missed_discount"] = 0.02 * t.inv_value * t.late

a = t[t.supplier.isin(suppliers)].copy()
assert a.supplier.nunique() == len(suppliers)
a["auto"] = a.po_value <= THRESHOLD
a["recovered"] = a.missed_discount * a.auto

# Cross-check against the Excel total and the ZNB1 headline scope
total_missed = a.missed_discount.sum()

g = a.groupby("supplier")
po_level = a.reset_index().drop_duplicates("PO_ID")
per = pd.DataFrame({
    "pos": po_level.groupby("supplier").size(),
    "pos_auto": po_level[po_level.po_value <= THRESHOLD].groupby("supplier").size(),
    "invoices": g.size(),
    "invoices_auto": g.auto.sum(),
    "late_invoices": g.late.sum(),
    "late_invoices_auto": g.apply(lambda x: (x.late & x.auto).sum()),
    "missed_discount_nok": g.missed_discount.sum(),
    "recovered_nok": g.recovered.sum(),
}).fillna(0)
per["remaining_nok"] = per.missed_discount_nok - per.recovered_nok
per["recovered_pct"] = 100 * per.recovered_nok / per.missed_discount_nok.where(per.missed_discount_nok > 0)
per = per.sort_values("missed_discount_nok", ascending=False)
per.to_csv(NOTES / "znb1_auto_threshold_potential.csv")

sens = pd.DataFrame([{
    "threshold_nok": th,
    "pos_auto": (po_level.po_value <= th).sum(),
    "invoices_auto": (a.po_value <= th).sum(),
    "late_invoices_recovered": (a.late & (a.po_value <= th)).sum(),
    "recovered_nok": a.missed_discount[a.po_value <= th].sum(),
    "invoice_value_auto_nok": a.inv_value[a.po_value <= th].sum(),
} for th in SENSITIVITY])
sens["recovered_pct"] = 100 * sens.recovered_nok / total_missed
sens.to_csv(NOTES / "znb1_auto_threshold_sensitivity.csv", index=False)

# --- chart: per supplier, recovered vs remaining (top 12 + rest) ---
top = per.head(12).copy()
rest = per.iloc[12:][["recovered_nok", "remaining_nok"]].sum()
top.loc[f"Other {len(per) - 12}"] = rest
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"]})
fig, ax = plt.subplots(figsize=(11, 6.2), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
y = range(len(top))[::-1]
ax.barh(list(y), top.recovered_nok / 1e6, height=0.64, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=2,
        label=f"Recovered: POs ≤{THRESHOLD / 1e6:g}M handled automatically")
ax.barh(list(y), top.remaining_nok / 1e6, left=top.recovered_nok / 1e6, height=0.64, color=MUTED, edgecolor=SURFACE,
        linewidth=2, zorder=2, label=f"Still missed: POs >{THRESHOLD / 1e6:g}M stay manual")
for yy, (_, r) in zip(y, top.iterrows()):
    tot = r.recovered_nok + r.remaining_nok
    ax.text(tot / 1e6 + 0.04, yy, f"{r.recovered_nok / 1e6:.2f}M of {tot / 1e6:.2f}M", va="center", fontsize=8.5,
            color=INK)
ax.set_yticks(list(y))
ax.set_yticklabels(top.index, fontsize=9, color=INK)
ax.set_xlabel("Missed 2% discount today (M NOK)", color=INK2, fontsize=10)
ax.set_xlim(0, top[["recovered_nok", "remaining_nok"]].sum(axis=1).max() / 1e6 * 1.22)
ax.tick_params(axis="both", length=0, colors=INK2)
ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
for s in ax.spines.values():
    s.set_visible(False)
rec = a.recovered.sum()
fig.suptitle(f"Automating always-manual POs up to {THRESHOLD / 1e6:g}M NOK recovers {rec / 1e6:.1f}M NOK "
             f"({rec / total_missed:.0%}) of missed discount", x=0.03, ha="left", y=0.97, fontsize=14,
             fontweight="bold", color=INK)
fig.text(0.03, 0.915, f"ZNB1 · {len(suppliers)} always-manual suppliers · {total_missed / 1e6:.1f}M NOK missed today · "
         f"{int((a.late & a.auto).sum())} of {int(a.late.sum())} late invoices sit on POs ≤{THRESHOLD / 1e6:g}M",
         fontsize=10, color=INK2, ha="left", va="top")
ax.legend(loc="upper left", bbox_to_anchor=(-0.13, 1.1), ncol=2, frameon=False, fontsize=9, labelcolor=INK2,
          handlelength=1.6, borderaxespad=0)
fig.text(0.03, 0.01, "Assumes automatic handling posts within 10 days, so today's late invoices on those POs earn the 2% "
         "discount. Scope: ZNB1 POs without never-posted invoices; GR clock; invoice value = PO value / posted invoices.",
         fontsize=7.5, color=INK2, ha="left")
fig.subplots_adjust(left=0.13, right=0.98, top=0.8, bottom=0.11)
fig.savefig(NOTES / "figures" / "znb1_auto_threshold_potential.png", facecolor=SURFACE)

print(f"suppliers {len(suppliers)}  invoices {len(a)}  late {int(a.late.sum())}  missed {total_missed:,.0f}")
print(f"<= {THRESHOLD:,.0f}: POs {int((po_level.po_value <= THRESHOLD).sum())}/{len(po_level)}  "
      f"invoices {int(a.auto.sum())}  late recovered {int((a.late & a.auto).sum())}  recovered {rec:,.0f}")
print(per.round(0).head(12).to_string())
print(sens.round(1).to_string())
