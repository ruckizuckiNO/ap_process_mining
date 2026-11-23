# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Process mining of an accounts-payable invoice process (11 SAP PO types, see `PO_TYPES` in `common.py`). Plain Python scripts, no tests or linter.

## Environment and running

- Use the conda env `datasci` (defined in `environment.yml`). Install new packages there: conda-forge first, pip only if conda-forge lacks it (e.g. pm4py). Never install into base or create new envs unasked.
- The raw event log is not in the repo. `common.py` reads it from `$AP_EVENT_LOG` (default `/Users/data/Downloads/EN event log.csv`).
- Run the pipeline in order; each step reads the outputs of earlier ones:

```bash
export AP_EVENT_LOG="/path/to/EN event log.csv"
for s in 01_prepare 02_process_maps 03_control_capability 04_prioritise 05_report; do
  conda run -n datasci python -W ignore $s.py
done
```

A single step can be rerun on its own as long as its inputs in `output/` exist (only `01_prepare.py` needs the raw log).

## Architecture

- `common.py` holds all shared paths, constants (bot users, posting activities, SLA limits, PO types, KPIs) and the `load_events/cases/invoices()` loaders. Change constants there, not in the scripts.
- Data flow (everything goes to `output/`, figures to `output/figures/`):
  - `01_prepare.py`: raw CSV → `events.pkl`, `cases.csv` (one row per PO_ID), `invoices.csv` (one row per PO_ID × SAP_PO_No invoice cycle), `bot_profile.csv`, `summary.json`.
  - `02_process_maps.py`: DFGs/variants per PO type (pm4py + graphviz).
  - `03_control_capability.py`: X̄–S / I-MR control charts with Western Electric rules and Cpu/Ppu capability → `control_chart_results.csv`, `capability*.csv`.
  - `04_prioritise.py`: combines the above → `priority.csv`; also the routing decision at *Initial validation* (`SAP_Next_Processing_Type`: `WAITGR`, `CTRL1`, `ZWMGRW`, `DELE`) → `routing.csv`, `routing_codes.csv`, and who posts on the direct validation → *Posted manually* path → `direct_posting.csv`.
  - `05_report.py`: builds `output/report.html` from the CSVs and figures; numbers in the report text come from those files via f-strings.
  - `06_report_znb1.py`: builds `output/report_znb1.html`: total PO value per PO type, then the whole analysis for ZNB1 only. Reuses the stylesheet of `05_report.py` (extracted from its source) and leaves `report.html` untouched.
  - `07_report_clock_compare.py`: builds `output/report_zjd1_znb1_clock.html`, ZJD1 vs ZNB1 per invoice (>10 d / >30 d, missed discount) with the clock from arrival and from goods receipt for both types, plus the clock start/stop assumptions. A side comparison only: it does not change `GR_CLOCK_PO_TYPES`.
  - `08_report_ctrl1_drivers.py`: builds `output/report_znb1_ctrl1.html`, what drives the CTRL1 vs WAITGR routing of ZNB1 invoices at validation. The log has no reason code; the routing follows the supplier (39 suppliers ≈ always CTRL1) and sticks to the PO. Its model figures come from `output/ctrl1_model_results.csv`, so run 09 first.
  - `09_report_ctrl1_model.py`: builds `output/report_znb1_ctrl1_model.html`, the machine-learning check of that routing (logistic regression + gradient boosting, 5-fold CV grouped by PO, permutation importance, drop-one ablation, unseen-supplier and time-split checks, inline SVG charts). Needs scikit-learn. Writes `ctrl1_model_results.csv`.
  - `ctrl1_data.py`: the shared invoice-level dataset and feature matrix for 08 and 09.
- `ap_process_mining.ipynb` is a self-contained copy of the whole analysis that writes to `ap_output/`. Logic changes in the scripts must be mirrored there by hand.
- `explore*.py` are untracked scratch files; don't build on them.
- `events.pkl`, `cases.csv`, `invoices.csv` are gitignored and rebuilt by `01_prepare.py`; the other `output/` files and figures are committed.

## Domain rules

See `docs/domain.md` for what each PO type and activity means.

- PO types fall into four procurement situations in this log: standard purchasing (ZJD1, ZNB1, ZJD3), third-party/dropshipping (ZNB2/3/4/6, the bulk of the volume), transportation (ZNBF) and returns (NB2, ZRPC, ZRPM). Returns are "reverse POs" that follow an earlier purchase, so don't judge them by purchasing benchmarks.

- Events are ordered by timestamp, then `Event_ID` (`Item_No` is not chronological). Cases without a PO type are dropped.
- The log ends at posting, not payment. "First posting" (any of `POSTING_ACTIVITIES`) is when the invoice became payable; SLA clocks run from the clock start to first posting.
- Clock start = the invoice's first event, except for `GR_CLOCK_PO_TYPES` (currently ZNB1 only, decided 2026-11-19): there the payment term is assumed to start at goods receipt, i.e. the first `Wait for goods receipt` event with `SAP_Object_Code` `WAIT_GR` (`GR_OBJECT_CODE`; read as the moment the GR was booked). The log has no GR date itself, so this is a proxy. `END_WAIT_GR` (wait ended without a GR, released to the manual queue) does not count, and invoices without a `WAIT_GR` step keep the first-event clock. `invoices.csv`/`cases.csv` carry `clock_start`; `start` stays the first event (used for weeks/ordering). ZNB1 with the first-event clock: 3,071 late invoices / 27.3M NOK missed discount (upper bound); with the GR clock: 1,687 / 15.9M (headline).
- Field meanings from the course's data description are in `docs/domain.md` ("Variables"): timestamps mark the completion of a stage (one per event, no start times); `Case_Invoice_Value_NOK` = amount on the invoice at the exchange rate on receipt, `Case_Clearing_value_NOK` = amount paid at the rate on clearing (differences are mostly exchange-rate effects). The description calls `SAP_PO_No` case-level, but the log has several per PO; we treat each as an invoice document.
- A PO_ID can contain several invoice cycles (`SAP_PO_No`, `Item_No` restarts at 1); each gets its own SLA clock.
- `BOT_USERS` are system users found by the user profiling in `01_prepare.py`; they are separated from humans in manual-work KPIs.
- SLA: 10 days soft, 30 days hard.
- KPIs are analysed on `log(x + 1)` with durations in days. Durations are never exactly zero. The normal-based expected breach rates and Cpu underestimate breaches for the fast PO types (ZNB2/3/4) because their distributions are bimodal, so prefer observed breach rates when reporting.
