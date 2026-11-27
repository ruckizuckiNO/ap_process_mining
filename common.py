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
