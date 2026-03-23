# Enbridge Wind Turbine Anomaly Detection

Predictive maintenance system for wind turbine fleets using SCADA sensor data. Built for the DSMLC x Enbridge case competition at the University of Calgary.

Develops a Unified Thermal Degradation Index (TDI) that scores turbine health from 0-100 using Normal Behavior Models (LightGBM), LSTM-Autoencoders, and statistical anomaly detection (CUSUM/EWMA).

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

## Tech Stack

pandas, LightGBM, PyTorch, SHAP, scikit-learn, statsmodels, Next.js, React, Recharts, cobe (3D globe)
