# Accounts-payable process mining

Process mining of an accounts-payable invoice process (11 SAP PO types) at a consumer electronics retailer.

- **Step 1** – process maps and variants per PO type
- **Step 2** – case KPIs (trace length, throughput time, manual activities), with bot/system users separated from humans
- **Step 3** – X̄–S / I-MR control charts on log(x+1) KPIs with Western Electric rules, and process capability (Cpu) against a 10-day soft and 30-day hard SLA
- **Step 4** – prioritisation and recommendations

The full write-up is in [`output/report.html`](output/report.html).

## Run

```bash
conda env create -f environment.yml      # or use an existing env
export AP_EVENT_LOG="/path/to/EN event log.csv"
for s in 01_prepare 02_process_maps 03_control_capability 04_prioritise 05_report; do
  conda run -n datasci python -W ignore $s.py
done
```

The raw event log is not included in this repository.
