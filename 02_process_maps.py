"""Step 1: per-PO-type process maps (DFGs), variant analysis and a cross-type comparison."""
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pm4py

from common import OUT, FIG, PO_TYPES, load_events, load_cases

ev = load_events()
cases = load_cases().set_index("PO_ID")
log = ev.rename(columns={"PO_ID": "case:concept:name", "Activity_ID": "concept:name",
                         "ts": "time:timestamp", "User_ID": "org:resource"})
log = log[["case:concept:name", "concept:name", "time:timestamp", "org:resource", "SAP_PO_Type_Code"]]

# Variant string per case (activities in order)
variant = ev.groupby("PO_ID").Activity_ID.agg(" → ".join)
cases["variant"] = variant

type_rows, variant_rows = [], []
for code, name in PO_TYPES.items():
    sub = log[log.SAP_PO_Type_Code == code]
    c = cases[cases.po_type == code]
    n = len(c)

    # Keep the map readable: drop activities/edges below 0.5% of cases for big types
    min_share = 0.005 if n >= 5000 else 0.02 if n >= 300 else 0.0
    dfg, sa, ea = pm4py.discover_dfg(sub)
    act_cases = sub.groupby("concept:name")["case:concept:name"].nunique()
    keep = set(act_cases[act_cases >= min_share * n].index)
    dfg = {k: v for k, v in dfg.items() if k[0] in keep and k[1] in keep and v >= min_share * n}
    # pm4py can only draw start/end arcs to activities that still appear on an edge
    on_edge = {a for e in dfg for a in e}
    sa = {k: v for k, v in sa.items() if k in on_edge}
    ea = {k: v for k, v in ea.items() if k in on_edge}
    pm4py.save_vis_dfg(dfg, sa, ea, str(FIG / "process_maps" / f"dfg_freq_{code}.png"))

    # Performance map (mean transition time) on the same filtered edges
    perf, _, _ = pm4py.discover_performance_dfg(sub)
    perf = {k: v for k, v in perf.items() if k in dfg}
    pm4py.save_vis_performance_dfg(perf, sa, ea, str(FIG / "process_maps" / f"dfg_perf_{code}.png"),
                                   aggregation_measure="median")

    vc = c.groupby("variant").agg(cases=("variant", "size"),
                                  median_tp_days=("throughput_days", "median"),
                                  mean_manual=("manual_count", "mean"))
    vc = vc.sort_values("cases", ascending=False)
    vc["share"] = vc.cases / n
    vc["cum_share"] = vc.share.cumsum()
    for rank, (v, r) in enumerate(vc.head(10).iterrows(), 1):
        variant_rows.append({"po_type": code, "rank": rank, "variant": v, **r.to_dict()})

    type_rows.append({
        "po_type": code, "name": name, "cases": n, "events": len(sub),
        "activities": sub["concept:name"].nunique(), "variants": len(vc),
        "top1_share": vc.share.iloc[0], "top5_share": vc.share.head(5).sum(),
        "variants_for_80pct": int((vc.cum_share < 0.8).sum() + 1),
        "dfg_edges_shown": len(dfg), "map_filter": f">= {min_share:.1%} of cases" if min_share else "none",
    })

types = pd.DataFrame(type_rows).set_index("po_type")
types.to_csv(OUT / "po_type_overview.csv")
pd.DataFrame(variant_rows).to_csv(OUT / "top_variants.csv", index=False)

# Activity presence heatmap: share of cases per type containing each activity
pres = (ev.drop_duplicates(["PO_ID", "Activity_ID"])
          .groupby(["SAP_PO_Type_Code", "Activity_ID"]).PO_ID.nunique()
          .unstack(0).reindex(columns=list(PO_TYPES)))
pres = pres.div(types.cases, axis=1).fillna(0)
pres = pres.loc[pres.max(axis=1).sort_values(ascending=False).index]
pres.to_csv(OUT / "activity_presence.csv")

fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(pres.values, cmap="Greens", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(pres.columns)), [f"{c}\n(n={types.cases[c]:,})" for c in pres.columns], fontsize=8)
ax.set_yticks(range(len(pres.index)), pres.index, fontsize=8)
for i in range(pres.shape[0]):
    for j in range(pres.shape[1]):
        v = pres.values[i, j]
        if v >= 0.005:
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=6.5, color="white" if v > 0.6 else "black")
fig.colorbar(im, ax=ax, label="Share of cases containing activity", shrink=0.6)
ax.set_title("Which activities occur in which PO-type workflow")
fig.tight_layout()
fig.savefig(FIG / "activity_presence.png", dpi=130)
plt.close(fig)

print(types.to_string())
