"""Step 3: control charts (Shewhart rules) and process capability per PO type, on log(x+1) KPIs.

Charts
- X̄–S charts with weekly subgroups (case start week) for PO types with enough cases per week.
  Variable subgroup sizes -> per-subgroup limits; sigma = pooled within-week std.
- I-MR charts on individual cases (ordered by start time) for the very small types.
- As a robustness check for the big types, an I-MR chart on the weekly means ("X̄ as individuals"),
  which treats normal week-to-week variation as common cause. With thousands of cases per week the
  strict X̄ limits become extremely narrow, so both verdicts are reported.

Shewhart / Western Electric rules applied to the standardised centre-line chart:
  R1 one point beyond 3σ; R2 2 of 3 beyond 2σ (same side); R3 4 of 5 beyond 1σ (same side);
  R4 8 in a row on one side; R5 6 in a row steadily increasing or decreasing.
"""
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.special import gammaln
from scipy.stats import norm

from common import OUT, FIG, PO_TYPES, KPIS, SLA_HARD_DAYS, SLA_SOFT_DAYS, load_cases, load_invoices

MIN_CASES_PER_WEEK = 5      # mean cases/week needed for an X̄–S chart
MIN_SUBGROUP = 2            # weeks with fewer cases are left out of X̄–S charts
RULES = ["R1", "R2", "R3", "R4", "R5"]

cases = load_cases()


def c4(n):
    n = np.asarray(n, dtype=float)
    return np.sqrt(2 / (n - 1)) * np.exp(gammaln(n / 2) - gammaln((n - 1) / 2))


def shewhart_rules(z):
    """Return dict rule -> boolean array marking the points that trigger it."""
    z = np.asarray(z, dtype=float)
    n = len(z)
    hits = {r: np.zeros(n, bool) for r in RULES}
    hits["R1"] = np.abs(z) > 3
    for i in range(n):
        for side in (1, -1):
            s = side * z
            if i >= 2 and (s[i - 2:i + 1] > 2).sum() >= 2:
                hits["R2"][i - 2:i + 1] |= s[i - 2:i + 1] > 2
            if i >= 4 and (s[i - 4:i + 1] > 1).sum() >= 4:
                hits["R3"][i - 4:i + 1] |= s[i - 4:i + 1] > 1
            if i >= 7 and (s[i - 7:i + 1] > 0).all():
                hits["R4"][i - 7:i + 1] = True
        if i >= 5:
            d = np.diff(z[i - 5:i + 1])
            if (d > 0).all() or (d < 0).all():
                hits["R5"][i - 5:i + 1] = True
    return hits


def draw(ax, x, y, center, lcl, ucl, hits, title, ylabel):
    ax.plot(x, y, "-", color="#4a6fa5", lw=1, marker="o", ms=3, zorder=2)
    ax.plot(x, np.broadcast_to(center, len(x)), color="#2a7f62", lw=1)
    ax.step(x, np.broadcast_to(ucl, len(x)), where="mid", color="#d1495b", lw=1, ls="--")
    ax.step(x, np.broadcast_to(lcl, len(x)), where="mid", color="#d1495b", lw=1, ls="--")
    bad = np.zeros(len(x), bool)
    for r in hits.values():
        bad |= r
    ax.scatter(np.asarray(x)[bad], np.asarray(y)[bad], color="#d1495b", s=18, zorder=3)
    ax.set_title(title, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    if isinstance(x, pd.DatetimeIndex):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax.tick_params(axis="x", rotation=45)


def summarise(hits):
    return {r: int(h.sum()) for r, h in hits.items()}


rows = []
for code, name in PO_TYPES.items():
    c = cases[cases.po_type == code].sort_values("start")
    weeks = c.groupby("start_week")
    use_xbar = len(c) / max(weeks.ngroups, 1) >= MIN_CASES_PER_WEEK
    chart_type = "X̄–S (weekly)" if use_xbar else "I-MR (cases)"

    fig, axes = plt.subplots(len(KPIS), 2, figsize=(14, 3.0 * len(KPIS)), squeeze=False)
    for k_i, (kpi, label) in enumerate(KPIS.items()):
        col = f"log_{kpi}"
        row = {"po_type": code, "kpi": kpi, "chart": chart_type, "cases": len(c)}
        if use_xbar:
            agg = weeks[col].agg(["mean", "std", "size"])
            agg = agg[agg["size"] >= MIN_SUBGROUP]
            n = agg["size"].values
            xbarbar = np.average(agg["mean"], weights=n)
            sigma = np.sqrt(((n - 1) * agg["std"] ** 2).sum() / (n - 1).sum())  # pooled within-week
            se = sigma / np.sqrt(n)
            z = (agg["mean"].values - xbarbar) / se
            hits = shewhart_rules(z)
            draw(axes[k_i, 0], agg.index, agg["mean"], xbarbar, xbarbar - 3 * se, xbarbar + 3 * se, hits,
                 f"X̄ chart – {label}", f"log({kpi}+1)")
            s_c = sigma * c4(n)
            s_w = sigma * np.sqrt(1 - c4(n) ** 2)
            s_hits = {"R1": (agg["std"].values > s_c + 3 * s_w) | (agg["std"].values < np.maximum(s_c - 3 * s_w, 0))}
            draw(axes[k_i, 1], agg.index, agg["std"], s_c, np.maximum(s_c - 3 * s_w, 0), s_c + 3 * s_w, s_hits,
                 f"S chart – {label}", "within-week std")

            # robustness: weekly means as individuals
            mr = np.abs(np.diff(agg["mean"].values))
            sig_w = mr.mean() / 1.128
            hits_w = shewhart_rules((agg["mean"].values - agg["mean"].mean()) / sig_w)
            row.update({"subgroups": len(agg), "sigma_within": sigma,
                        "s_chart_R1": int(s_hits["R1"].sum()),
                        "weekly_I_violations": int(np.any(np.vstack(list(hits_w.values())), axis=0).sum()),
                        "weekly_I_in_control": not any(h.any() for h in hits_w.values())})
        else:
            y = c[col].values
            mr = np.abs(np.diff(y))
            sigma = mr.mean() / 1.128 if len(mr) else np.nan
            center = y.mean()
            z = (y - center) / sigma if sigma and sigma > 0 else np.zeros_like(y)
            hits = shewhart_rules(z)
            x = np.arange(1, len(y) + 1)
            draw(axes[k_i, 0], x, y, center, center - 3 * sigma, center + 3 * sigma, hits,
                 f"Individuals chart – {label}", f"log({kpi}+1)")
            mr_hits = {"R1": mr > 3.267 * mr.mean()} if len(mr) else {"R1": np.array([], bool)}
            draw(axes[k_i, 1], x[1:], mr, mr.mean(), 0, 3.267 * mr.mean(), mr_hits,
                 f"Moving range – {label}", "MR")
            row.update({"subgroups": len(y), "sigma_within": sigma, "s_chart_R1": int(mr_hits["R1"].sum()),
                        "weekly_I_violations": np.nan, "weekly_I_in_control": np.nan})
        row.update(summarise(hits))
        row["points_flagged"] = int(np.any(np.vstack(list(hits.values())), axis=0).sum())
        row["in_control"] = row["points_flagged"] == 0 and row["s_chart_R1"] == 0
        rows.append(row)
    fig.suptitle(f"{code} – {name}  ({chart_type}, n = {len(c):,} cases)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "control_charts" / f"cc_{code}.png", dpi=110)
    plt.close(fig)

control = pd.DataFrame(rows)
control.to_csv(OUT / "control_chart_results.csv", index=False)

# --- Process capability on log(throughput + 1) -------------------------------
cap_rows = []
for code, name in PO_TYPES.items():
    c = cases[cases.po_type == code]
    y = c.log_throughput_days
    mu, s_overall = y.mean(), y.std(ddof=1)
    s_within = control.loc[(control.po_type == code) & (control.kpi == "throughput_days"), "sigma_within"].iloc[0]
    r = {"po_type": code, "name": name, "cases": len(c), "mean_log_tp": mu,
         "sigma_within": s_within, "sigma_overall": s_overall,
         "median_tp_days": c.throughput_days.median(), "p90_tp_days": c.throughput_days.quantile(0.9)}
    for tag, usl_days in (("soft10", SLA_SOFT_DAYS), ("hard30", SLA_HARD_DAYS)):
        usl = np.log(usl_days + 1)
        r[f"Cpu_{tag}"] = (usl - mu) / (3 * s_within)
        r[f"Ppu_{tag}"] = (usl - mu) / (3 * s_overall)
        r[f"expected_breach_{tag}"] = 1 - norm.cdf((usl - mu) / s_overall)
        r[f"observed_breach_{tag}"] = (c.throughput_days > usl_days).mean()
        r[f"cases_breach_{tag}"] = int((c.throughput_days > usl_days).sum())
    cap_rows.append(r)
cap = pd.DataFrame(cap_rows).set_index("po_type")
cap.to_csv(OUT / "capability.csv")


def sigma_within(c, col):
    """Within-subgroup sigma, estimated the same way as in the control charts above."""
    c = c.dropna(subset=[col]).sort_values("start")
    weeks = c.groupby("start_week")[col]
    if len(c) / max(weeks.ngroups, 1) >= MIN_CASES_PER_WEEK:
        agg = weeks.agg(["std", "size"])
        agg = agg[agg["size"] >= MIN_SUBGROUP]
        return np.sqrt(((agg["size"] - 1) * agg["std"] ** 2).sum() / (agg["size"] - 1).sum())
    return np.abs(np.diff(c[col].values)).mean() / 1.128


# --- Same capability analysis, clock stopped at the FIRST posting ---------------
# The log ends at posting, not payment. First posting is when the invoice first became payable;
# the last event also includes rework after that (reversals, re-postings).
def posting_capability(units):
    """Cpu and breach rates for time-to-first-posting; `units` are cases or invoice cycles."""
    out = []
    for code, name in PO_TYPES.items():
        c = units[(units.po_type == code) & units.posted]
        y = c.log_time_to_post_days
        mu, s_w = y.mean(), sigma_within(c, "log_time_to_post_days")
        r = {"po_type": code, "posted": len(c), "not_posted": int(((units.po_type == code) & ~units.posted).sum()),
             "mean_log_ttp": mu, "sigma_within": s_w, "median_ttp_days": c.time_to_post_days.median()}
        for tag, usl_days in (("soft10", SLA_SOFT_DAYS), ("hard30", SLA_HARD_DAYS)):
            usl = np.log(usl_days + 1)
            r[f"Cpu_{tag}"] = (usl - mu) / (3 * s_w)
            r[f"observed_breach_{tag}"] = (c.time_to_post_days > usl_days).mean()
            r[f"cases_breach_{tag}"] = int((c.time_to_post_days > usl_days).sum())
        out.append(r)
    return pd.DataFrame(out).set_index("po_type")


cap_post = posting_capability(cases)
cap_post["reposted_cases"] = cases[cases.posted].groupby("po_type").n_postings.agg(lambda s: int((s > 1).sum()))
cap_post.to_csv(OUT / "capability_first_post.csv")
cap_inv = posting_capability(load_invoices())
cap_inv.to_csv(OUT / "capability_invoice.csv")

xs = np.arange(len(cap))
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
for ax, tag, title in ((axes[0], "soft10", "10-day soft SLA"), (axes[1], "hard30", "30-day hard SLA")):
    ax.bar(xs - 0.27, cap[f"Cpu_{tag}"], 0.27, label="Case: first to last event", color="#4a6fa5")
    ax.bar(xs, cap_post[f"Cpu_{tag}"], 0.27, label="Case: first event to first posting", color="#e8a33d")
    ax.bar(xs + 0.27, cap_inv[f"Cpu_{tag}"], 0.27, label="Invoice cycle: first event to posting", color="#2a7f62")
    ax.axhline(1, color="#d1495b", ls="--", lw=1, label="Cpu = 1")
    ax.set_xticks(xs, cap.index, fontsize=8)
    ax.set_title(f"Cpu, {title}")
axes[0].set_ylabel("Cpu (log scale KPI)")
axes[0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "capability_end_points.png", dpi=130)
plt.close(fig)

# Capability chart
fig, ax = plt.subplots(figsize=(10, 4.5))
xs = np.arange(len(cap))
ax.bar(xs - 0.2, cap.Cpu_soft10, 0.4, label="Cpu, 10-day soft SLA", color="#e8a33d")
ax.bar(xs + 0.2, cap.Cpu_hard30, 0.4, label="Cpu, 30-day hard SLA", color="#2a7f62")
ax.axhline(1, color="#d1495b", ls="--", lw=1, label="Cpu = 1")
ax.set_xticks(xs, [f"{i}\nn={n:,}" for i, n in zip(cap.index, cap.cases)], fontsize=8)
ax.set_ylabel("Cpu (log scale KPI)")
ax.set_title("Process capability of throughput time per PO type")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "capability.png", dpi=130)
plt.close(fig)

# Distribution of log throughput vs SLA lines
fig, axes = plt.subplots(3, 4, figsize=(15, 9), sharex=True)
for ax, code in zip(axes.flat, PO_TYPES):
    y = cases.loc[cases.po_type == code, "log_throughput_days"]
    ax.hist(y, bins=40, color="#4a6fa5")
    for d, col in ((SLA_SOFT_DAYS, "#e8a33d"), (SLA_HARD_DAYS, "#d1495b")):
        ax.axvline(np.log(d + 1), color=col, ls="--")
    ax.set_title(f"{code} (n={len(y):,})", fontsize=9)
axes.flat[-1].axis("off")
fig.supxlabel("log(throughput days + 1)   – dashed: 10-day and 30-day SLA")
fig.tight_layout()
fig.savefig(FIG / "throughput_distributions.png", dpi=110)
plt.close(fig)

pd.set_option("display.width", 250)
print(control[["po_type", "kpi", "chart", "subgroups", *RULES, "s_chart_R1", "in_control", "weekly_I_in_control"]].to_string())
print(cap.round(3).to_string())
