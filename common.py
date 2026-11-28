"""Shared paths, constants and loaders for the AP process mining case."""
import os
from pathlib import Path

import pandas as pd

# The event log is not in the repo; point AP_EVENT_LOG at your copy
RAW_CSV = Path(os.environ.get("AP_EVENT_LOG", "/Users/data/Downloads/EN event log.csv"))
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
FIG = OUT / "figures"
for d in (OUT, FIG, FIG / "process_maps", FIG / "control_charts"):
    d.mkdir(parents=True, exist_ok=True)

EVENTS_PKL = OUT / "events.pkl"
CASES_CSV = OUT / "cases.csv"
INVOICES_CSV = OUT / "invoices.csv"

# System users identified in 01_prepare.py (see bot_profile.csv for the evidence)
BOT_USERS = ["User1587", "User280"]

# Activities that post the invoice in SAP, i.e. make it payable
POSTING_ACTIVITIES = ["MM Posted Automatically", "Posted manually", "MM Autopost after manual process"]

SLA_HARD_DAYS = 30
SLA_SOFT_DAYS = 10

# Payment-term clock: by default it starts at the invoice's first event (arrival in AP). For these PO types it starts
# at goods receipt instead: the first "Wait for goods receipt" event with object code WAIT_GR, read as the moment the
# goods receipt was booked. END_WAIT_GR (wait ended without a goods receipt, released to the manual queue) does not
# count. Invoices without a WAIT_GR step keep the arrival clock (no goods-receipt evidence in the log).
GR_ACTIVITY = "Wait for goods receipt"
GR_OBJECT_CODE = "WAIT_GR"
GR_CLOCK_PO_TYPES = ["ZNB1"]

PO_TYPES = {
    "NB2": "Return to vendor",
    "ZJD1": "Normal order",
    "ZJD3": "Automatic order",
    "ZNB1": "Manual order",
    "ZNB2": "Third-party order Type 1",
    "ZNB3": "Third-party order Type 2",
    "ZNB4": "Third-party order Type 3",
    "ZNB6": "Third-party order Type 4",
    "ZNBF": "Transportation order",
    "ZRPC": "Return order Type 1",
    "ZRPM": "Return order Type 2",
}

KPIS = {
    "trace_length": "Trace length (events)",
    "throughput_days": "Throughput time (days)",
    "manual_count": "Manual activities (count)",
}

# KPIs on the control charts: one value per posted invoice cycle, time measured on the invoice clock above
CHART_KPIS = {
    "events": "Events per invoice",
    "time_to_post_days": "Days to first posting (invoice clock)",
    "manual_count": "Manual events per invoice",
}


def load_events() -> pd.DataFrame:
    return pd.read_pickle(EVENTS_PKL)


def load_invoices() -> pd.DataFrame:
    return pd.read_csv(INVOICES_CSV, parse_dates=["start", "goods_receipt", "clock_start", "first_post", "start_week"])


def load_cases() -> pd.DataFrame:
    return pd.read_csv(CASES_CSV, parse_dates=["start", "end", "start_week", "first_post", "clock_start"])


# --- Chart wording: an "invoice" in this log is SAP_PO_No, a proxy for the real invoice document --------------------
# Figures call use_invoice_proxy_labels() once; every matplotlib text (titles, axes, ticks, legends, annotations) then
# shows "invoice proxy"/"invoice proxies" instead of "invoice"/"invoices". SAP names stay as they are: the activity
# "Invoice document deleted" and field names such as Case_Invoice_Value_NOK (the underscore keeps them from matching).
PROXY_KEEP = ["Invoice document deleted", "suppliers invoice", "Suppliers invoice", "to invoice"]  # names and verbs


def invoice_proxy_text(s):
    import re
    if not isinstance(s, str) or "nvoice" not in s.lower():
        return s
    keep = {}
    for i, k in enumerate(PROXY_KEEP):
        token = f"\x00{i}\x00"
        s, keep[token] = s.replace(k, token), k

    def rep(m):
        word, hyphen = m.group(1), m.group(2)
        out = ("invoice proxies" if word.lower().endswith("s") else "invoice proxy")
        if hyphen:
            out = out.replace(" ", "-")
        if word.isupper():
            out = out.upper()
        elif word[0].isupper():
            out = out[0].upper() + out[1:]
        return out + hyphen

    s = re.sub(r"(?<![A-Za-z_])(invoices?)(?![A-Za-z_])(?!\s*prox)(-?)", rep, s, flags=re.I)
    for token, k in keep.items():
        s = s.replace(token, k)
    return s


def use_invoice_proxy_labels():
    import matplotlib.text as mtext
    if getattr(mtext.Text, "_invoice_proxy", False):
        return
    orig = mtext.Text.set_text

    def set_text(self, s):
        return orig(self, invoice_proxy_text(s) if isinstance(s, str) else s)

    mtext.Text.set_text = set_text
    mtext.Text._invoice_proxy = True
    if os.environ.get("PROXY_CHECK"):
        _report_clipped_text()


def _report_clipped_text():
    """Debug aid (PROXY_CHECK=1): print texts that end outside the figure when it is saved."""
    import matplotlib.figure as mfig
    orig = mfig.Figure.savefig

    def savefig(self, fname, *a, **k):
        r = self.canvas.get_renderer()
        self.draw(r)
        fb = self.bbox
        for txt in self.findobj(lambda o: hasattr(o, "get_window_extent") and hasattr(o, "get_text")):
            if not txt.get_visible() or not txt.get_text().strip():
                continue
            bb = txt.get_window_extent(r)
            if bb.x1 > fb.x1 + 1 or bb.x0 < fb.x0 - 1:
                print(f"CLIPPED {fname}: {txt.get_text()[:90]!r} x0={bb.x0:.0f} x1={bb.x1:.0f} fig={fb.x1:.0f}")
        return orig(self, fname, *a, **k)

    mfig.Figure.savefig = savefig
