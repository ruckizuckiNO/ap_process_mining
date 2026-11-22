"""ZNB1 manual queue: is it worked first-in-first-out, by PO value, or by something else?

Queue per clerk = the ZNB1 invoices that clerk eventually posted manually, from entering CTRL1 to the posting.
Tests per clerk (>= 100 postings):
  age_pct            at each posting, share of the clerk's waiting invoices that arrived later than the posted one
                     (1 = always the oldest = FIFO, 0.5 = random, 0 = always the newest)
  value_pct          same for PO value (1 = always the largest, 0.5 = value plays no role)
  within_day_tau     Kendall tau between posting order and arrival order among one day's postings (weighted mean;
                     1 = strictly oldest first)
  value_wait_rho     Spearman rho between PO value and wait among invoices entering the queue on the same day
                     (negative = larger POs handled sooner; ~0 = no value priority)
  supplier_eta2 /    share of the variance in wait explained by supplier / by arrival week
  week_eta2
Run from the repo root:  PYTHONPATH=. conda run -n datasci python -W ignore notes/znb1_queue_order_test.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from common import load_cases, load_events

NOTES = Path(__file__).resolve().parent
K = ["PO_ID", "SAP_PO_No"]

ev = load_events()
z = ev[ev.SAP_PO_Type_Code == "ZNB1"]
pm = z[z.Activity_ID == "Posted manually"].sort_values("ts").drop_duplicates(K).set_index(K)
t = pd.DataFrame({"user": pm.User_ID, "post": pm.ts, "supplier": pm.Supplier_ID})
t["qin"] = z[z.SAP_Next_Processing_Type == "CTRL1"].groupby(K).ts.min().reindex(t.index)
t = t.dropna(subset=["qin"])
t = t[t.post > t.qin]
t["value"] = load_cases().set_index("PO_ID").invoice_value_nok.reindex(t.index.get_level_values("PO_ID")).values
t["wait"] = (t.post - t.qin).dt.total_seconds() / 86400


def eta2(g, col):
    m = g.groupby(col).wait.transform("mean")
    return 1 - ((g.wait - m) ** 2).sum() / ((g.wait - g.wait.mean()) ** 2).sum()


rows = []
for user, g in t.groupby("user"):
    if len(g) < 100:
        continue
    g = g.sort_values("post")
    qin, post, val = g.qin.values, g.post.values, g.value.values
    age, vp = [], []
    for i in range(len(g)):
        m = (qin < post[i]) & (post >= post[i])
        n = m.sum()
        if n < 5:
            continue
        age.append(((qin[m] > qin[i]).sum() + 0.5 * ((qin[m] == qin[i]).sum() - 1)) / (n - 1))
        vp.append(((val[m] < val[i]).sum() + 0.5 * ((val[m] == val[i]).sum() - 1)) / (n - 1))
    taus, w = [], []
    for _, x in g.groupby(g.post.dt.date):
        if len(x) >= 5 and x.qin.nunique() > 1:
            taus.append(kendalltau(x.post, x.qin).statistic)
            w.append(len(x))
    s = g.assign(qd=g.qin.dt.date)
    s = s[s.groupby("qd").value.transform("size") >= 3]
    rho = spearmanr(s.groupby("qd").value.rank(pct=True), s.groupby("qd").wait.rank(pct=True)).statistic
    rows.append({"user": user, "postings": len(g), "median_wait_days": g.wait.median(),
                 "median_waiting_at_posting": np.nan if not age else int(np.median(
                     [((qin < p) & (post >= p)).sum() for p in post])),
                 "age_pct": np.mean(age), "value_pct": np.mean(vp),
                 "within_day_tau": np.average(taus, weights=w), "value_wait_rho": rho,
                 "supplier_eta2": eta2(g, "supplier"), "week_eta2": eta2(g, g.qin.dt.to_period("W"))})

r = pd.DataFrame(rows).sort_values("postings", ascending=False).round(2)
r.to_csv(NOTES / "znb1_queue_order_test.csv", index=False)
pd.set_option("display.width", 220)
print(r.to_string(index=False))
