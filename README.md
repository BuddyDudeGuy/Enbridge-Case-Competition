# Enbridge Wind Turbine Anomaly Detection

Built for the DSMLC x Enbridge case competition at the University of Calgary.

## The Problem

Wind turbine fleets generate massive volumes of SCADA sensor data across dozens of thermal subsystems, but operators still rely on static alarm thresholds that either fire too late or drown teams in false positives. Faults like gearbox oil overheating or bearing degradation can develop over days or weeks with no alert, while sudden spikes in unrelated sensors trigger unnecessary shutdowns. There is no unified, fleet-wide metric that quantifies how thermally degraded a turbine actually is relative to its own normal operating behavior.

## Our Solution

We built a Unified Thermal Degradation Index (TDI) -- a single 0-100 composite health score per turbine event that replaces static thresholds with learned baselines. The pipeline trains Normal Behavior Models (LightGBM) on healthy operating data for each thermal subsystem (gearbox, generator bearings, cooling, transformer, hydraulic), then flags deviations using CUSUM and EWMA on the residuals. An LSTM-Autoencoder runs as a complementary detection layer and produces learned embeddings for cross-event similarity search. The per-subsystem anomaly signals roll up into the TDI score, giving operators a single number that captures the severity and spread of thermal degradation across the entire turbine.

The framework is asset-agnostic -- the same learn-normal-then-score-deviations pattern applies anywhere you have sensor data and a concept of "healthy" operation: pipelines, compressor stations, solar inverters, industrial motors, or any fleet of instrumented equipment.

## Setup

### Python (analysis and modeling)

```bash
pip install -r requirements.txt
```

### Dashboard (Next.js frontend)

```bash
cd dashboard
npm install
```

## Running the Dashboard

```bash
cd dashboard
npm run dev
```

Opens at [http://localhost:3000](http://localhost:3000). The dashboard shows an interactive globe with wind farm locations, fleet-level KPIs, thermal health monitoring with subsystem drill-down, and event timeline charts.

To build for production:

```bash
cd dashboard
npm run build
npm start
```

## Running Notebooks

```bash
jupyter notebook notebooks/
```

Notebooks are numbered sequentially: EDA (01-08), SHAP analysis (09), residual validation (10), model comparison (11), TDI validation (12).

## Project Structure

```
├── dashboard/         Next.js monitoring dashboard
│   ├── components/    Globe, Header, MonitoringPanel, EventTimeline, PredictiveChart
│   ├── data/          Fleet KPIs and event chart data
│   └── public/        Static figure assets
├── notebooks/         Jupyter notebooks (EDA, modeling, validation)
├── src/
│   ├── data/          Data loading and cleaning
│   ├── features/      Feature engineering
│   ├── models/        NBM, LSTM-AE, TDI, CARE scoring
│   └── visualization/ Chart generation scripts
├── outputs/
│   ├── figures/       Saved plots
│   └── reports/       TDI scores, model results
└── requirements.txt
```

## Future Work

The current dashboard displays pre-computed TDI scores and event timelines. The next step is a conversational fleet intelligence layer -- a RAG (Retrieval-Augmented Generation) system where every historical event, its sensor residuals, SHAP explanations, and maintenance outcomes get embedded into a vector store. An operator could ask "has this turbine shown this pattern before?" or "which farms had cooling anomalies last quarter?" and get grounded, citation-backed answers pulled directly from the fleet's own data rather than a generic model. Vectorizing the event fingerprints (residual signatures + SHAP feature attributions) also enables a similar-fault finder that surfaces the closest historical matches to a developing anomaly, giving maintenance teams a head start on diagnosis before a technician is even dispatched.

## Tech Stack

pandas, LightGBM, PyTorch, SHAP, scikit-learn, statsmodels, Next.js, React, Recharts, cobe (3D globe)
