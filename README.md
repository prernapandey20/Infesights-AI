# Infesights AI

Universal data-stream analytics with unsupervised anomaly detection and domain-aware
root-cause reports.

## Run

```powershell
venv\Scripts\python.exe -m streamlit run app.py
```

The dashboard opens with a sample commerce dataset. Upload a CSV or JSON file to analyze
your own records. JSON can be an array of objects or an object containing a `records`
or `data` array. Numerical columns are inferred automatically; columns named `timestamp`,
`datetime`, `date`, or `time` are used as the event time when present.

Numeric identifier columns such as row IDs, postal codes, and UUIDs are ignored automatically.
Static uploads run batch anomaly detection immediately; enable live stream mode to cycle
through incoming rows at a natural cadence. Alert thresholds are calibrated from the top 2%
of baseline deviations. Set `GROQ_API_KEY` in the environment to enable SQL-assisted AI
root-cause reports. The agent's DuckDB
investigation is restricted to read-only `SELECT` queries over the uploaded rows.

## Analytics and Integrations

- **Feature drift:** The Feature drift tab compares the initial 70% training window with
	the incoming holdout/stream window using the Kolmogorov-Smirnov test and PSI.
- **Executive exports:** Anomaly reports can be downloaded as Markdown or PDF.
- **Webhook alerts:** Set `INFESIGHTS_WEBHOOK_URL` to receive JSON alerts when anomaly
	confidence reaches the automatically calibrated baseline threshold.
- **Counterfactual review:** Adjust anomalous numeric values and re-evaluate risk without
	mutating the live detector's training buffer.

Run the focused regression tests with:

```powershell
venv\Scripts\python.exe -m unittest test_dynamic_pipeline
```
