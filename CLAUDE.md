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
  - `04_prioritise.py`: combines the above → `priority.csv`.
  - `05_report.py`: builds `output/report.html` from the CSVs and figures; numbers in the report text come from those files via f-strings.
- `ap_process_mining.ipynb` is a self-contained copy of the whole analysis that writes to `ap_output/`. Logic changes in the scripts must be mirrored there by hand.
- `explore*.py` are untracked scratch files; don't build on them.
- `events.pkl`, `cases.csv`, `invoices.csv` are gitignored and rebuilt by `01_prepare.py`; the other `output/` files and figures are committed.

## Domain rules

- Events are ordered by timestamp, then `Event_ID` (`Item_No` is not chronological). Cases without a PO type are dropped.
- The log ends at posting, not payment. "First posting" (any of `POSTING_ACTIVITIES`) is when the invoice became payable; SLA clocks run from the first event to first posting.
- A PO_ID can contain several invoice cycles (`SAP_PO_No`, `Item_No` restarts at 1); each gets its own SLA clock.
- `BOT_USERS` are system users found by the user profiling in `01_prepare.py`; they are separated from humans in manual-work KPIs.
- SLA: 10 days soft, 30 days hard.
- KPIs are analysed on `log(x + 1)` with durations in days. Durations are never exactly zero. The normal-based expected breach rates and Cpu underestimate breaches for the fast PO types (ZNB2/3/4) because their distributions are bimodal, so prefer observed breach rates when reporting.
