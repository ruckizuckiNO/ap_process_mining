"""Shared dataset for the ZNB1 CTRL1 routing analyses (08_report_ctrl1_drivers.py, 09_report_ctrl1_model.py).

One row per ZNB1 invoice (PO_ID x SAP_PO_No) with the routing code of its first Initial validation, restricted to
CTRL1 (manual control queue) vs WAITGR (goods-receipt check).
"""
import numpy as np
import pandas as pd

from common import load_events

T = "ZNB1"
FEATURES = {
    "supplier": "Supplier (Supplier_ID)",
    "log_value": "PO value, log(NOK + 1)",
    "amount_diff": "Clearing vs invoice amount (relative)",
    "inv_pos": "Order of the invoice on its PO",
    "n_inv": "Number of invoices on the PO",
    "month": "Month of validation",
    "weekday": "Weekday of validation",
    "hour": "Hour of validation",
}


def build():
    """Returns (val, dm, v): first validation and data-management event per invoice, and the analysis table v."""
    ev = load_events()
    ev = ev[ev.SAP_PO_Type_Code == T]
    val = ev[ev.Activity_ID == "Initial validation"].groupby(["PO_ID", "SAP_PO_No"]).first()
    dm = ev[ev.Activity_ID == "Data management"].groupby(["PO_ID", "SAP_PO_No"]).first()
    v = val[val.SAP_Next_Processing_Type.isin(["CTRL1", "WAITGR"])].reset_index().sort_values("ts").reset_index(drop=True)
    v["ctrl"] = v.SAP_Next_Processing_Type == "CTRL1"
    v["inv_pos"] = v.groupby("PO_ID").cumcount() + 1
    v["n_inv"] = v.groupby("PO_ID").PO_ID.transform("size")
    v["value"] = v.Case_Invoice_Value_NOK
    v["diff"] = (v.Case_Clearing_value_NOK - v.Case_Invoice_Value_NOK) / v.Case_Invoice_Value_NOK
    return val, dm, v


def features(v):
    """Model matrix: every column is known at validation time except n_inv (see the model report's caveats)."""
    return pd.DataFrame({
        "supplier": v.Supplier_ID.astype("category").cat.codes, "log_value": np.log1p(v.value),
        "amount_diff": v["diff"].fillna(0), "inv_pos": v.inv_pos, "n_inv": v.n_inv,
        "month": v.ts.dt.month, "weekday": v.ts.dt.dayofweek, "hour": v.ts.dt.hour})
