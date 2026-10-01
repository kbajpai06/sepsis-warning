# ICU Early Sepsis Warning System

End-to-end machine-learning system that scores every ICU patient **every hour** for sepsis risk
up to ~6 hours before onset, and explains each alert with the **top 3 clinical drivers**.

SQL (MIMIC-IV) → leakage-safe time-series features → XGBoost / LightGBM vs. PyTorch LSTM →
SHAP explanations → FastAPI + Docker service → MLflow experiment tracking.

> **Research prototype. Not a medical device and not for clinical use.**

## Why it matters
Sepsis is a leading cause of in-hospital death, and each hour of delayed treatment raises mortality.
Monitors stream vitals continuously, but subtle multi-organ trends (rising lactate, falling MAP,
rising HR) are hard to catch by eye across many patients. This project turns those streams into an
hourly risk score *with a reason attached*, because clinicians will not act on a black-box alarm.

## Architecture
```
MIMIC-IV (PostgreSQL)  ─┐
PhysioNet/CinC 2019 .psv├─►  hourly table  ─►  preprocess  ─►  rolling features ─► XGBoost ─┐
Synthetic generator    ─┘  (stay × hour)       (clean, LOCF,    (3/6/12h mean,    LightGBM ─┼─► evaluate ─► artifacts
                                               staleness)       std, slope)       LSTM/GRU ─┘   (row + patient level)
                                                                                      │
                                                  SHAP (TreeExplainer) on XGBoost ◄───┘
                                                                │
                              FastAPI  /predict  ──►  risk score + alert + top-3 drivers   (Docker)
                              MLflow: params, metrics, curves, model files for every run
```

## What is implemented
| Stage | Details |
|---|---|
| **SQL extraction** | `sql/hourly_dataset.sql`: CTE pipeline over `icustays`, `patients`, `chartevents`, `labevents`, `mimiciv_derived.sepsis3`. Builds an hourly stay × hour grid, hourly-averages 7 vitals and 4 labs, adds age/sex, applies Sepsis-3 labels in the PhysioNet 2019 convention (label = 1 from 6 h before onset), excludes stays already septic at ICU entry. |
| **Preprocessing** | Physiological range cleaning, per-patient forward-fill (LOCF), `*_measured` flags and `*_age` (hours since last real measurement) to preserve missingness information. |
| **Feature engineering** | Trailing rolling **mean / std / slope** over **3, 6, 12 h** for 11 signals, plus shock index. 126 features. A test proves features at hour *t* never change when later data changes. |
| **Models** | XGBoost and LightGBM on engineered features; causal 2-layer **LSTM** (or GRU) on raw sequences emitting a risk at every hour. Class imbalance handled via damped class weights; early stopping on AUPRC. |
| **Evaluation** | Patient-level split (no patient in two sets), stratified. Thresholds chosen on **validation** only. Row-level AUROC / AUPRC / sensitivity / specificity / precision **and** patient-level sensitivity, **median lead time (h)** and false-alarm burden. |
| **Explainability** | SHAP TreeExplainer. Values are aggregated per clinical variable and the top 3 *risk-increasing* ones are returned in plain language, e.g. `Elevated lactate + Rapid drop in mean arterial pressure + Elevated shock index`. |
| **Serving** | FastAPI: `/predict`, `/predict/batch`, `/health`, `/model-info`, `/metrics`; Pydantic validation; same feature code as training; handles gaps in hours; Docker image with healthcheck and non-root user. |
| **MLOps** | MLflow logs parameters, per-model metrics, LSTM learning curves, figures and model files; `meta.json` stores feature list, threshold and model version for the API. |
| **Quality** | 15 pytest tests (leakage, LOCF, slopes, split integrity, causal RNN, SHAP correctness, API behaviour) + GitHub Actions CI. |

## Results

| Model | AUROC | AUPRC | Sens. @ thr | Spec. @ thr | Patient sens. | Median lead time (h) |
|---|---|---|---|---|---|---|
| XGBoost | _TODO_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ |
| LightGBM | _TODO_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ |
| LSTM | _TODO_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ |

`docs/synthetic_demo/` contains outputs from the built-in **synthetic** generator (AUROC ≈ 0.998). They only prove the
pipeline runs end to end; the synthetic signal is deliberately easy and says **nothing** about clinical performance.
Real-world sepsis AUROCs are much lower, so expect that.

## Quick start (Windows PowerShell / VS Code)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

pytest -q                                              # run tests
python -m sepsis.train --source synthetic              # demo run (no data access needed)
mlflow ui --backend-store-uri sqlite:///mlflow.db      # inspect experiments at http://localhost:5000
uvicorn sepsis.api.main:app --port 8000                # serve; docs at http://localhost:8000/docs
```
Call the API:
```powershell
python scripts/make_sample_request.py
curl.exe -X POST http://localhost:8000/predict -H "Content-Type: application/json" --data-binary "@scripts/sample_request.json"
```
Docker (after training, so `artifacts/` exists):
```powershell
docker compose up --build
```

## Using real data
1. **PhysioNet/CinC 2019** (open, easiest): download `training_setA/B` into `data/physionet2019/`, then
   `python -m sepsis.train --source challenge2019 --data-dir data/physionet2019`
2. **MIMIC-IV** (credentialed access + CITI training required): load into PostgreSQL, build the
   [mimic-code](https://github.com/MIT-LCP/mimic-code) `mimiciv_derived` tables (needs `sepsis3`), then
   `python -m sepsis.train --source mimic --db-url postgresql+psycopg2://USER:PW@localhost:5432/mimiciv --limit-stays 5000`
   Never commit MIMIC data (it is git-ignored).

## API example
```json
POST /predict  ->  {
  "patient_id": "demo-4", "hour": 69, "risk_score": 0.9995, "threshold": 0.173, "alert": true,
  "risk_level": "HIGH",
  "alert_message": "Elevated lactate + Elevated shock index + Rapid rise in temperature",
  "top_drivers": [{"variable": "lactate", "description": "Elevated lactate (3.11)", "shap": 3.83, "...": "..."}]
}
```

## Repo layout
```
sepsis/  config.py  features.py  evaluate.py  explain.py  train.py
         data/{synthetic,physionet2019,mimic}.py   models/{gbm,lstm}.py   api/{main,service,schemas}.py
sql/hourly_dataset.sql     tests/     scripts/     Dockerfile   docker-compose.yml   .github/workflows/ci.yml
```

## Design decisions & limitations
- **Patient-level splits** prevent the same patient appearing in train and test (a common source of inflated scores).
- **Trailing windows only**: no future data leaks into any feature.
- **Threshold on validation, report on test.** Probabilities are class-weighted and not calibrated; calibrate (isotonic/Platt) before using scores as true probabilities.
- **Alarm fatigue** is tracked explicitly via false-alarm metrics, not just AUROC.
- The API scores one time point from a supplied history; it is not wired to a live hospital data stream.
- Truncates stays to the first `--max-hours` (default 72) for fair comparison between tree and sequence models.
- Single-center data (MIMIC) and a SQL label proxy; external validation would be required for any real use.
- The SQL was written against the mimic-code PostgreSQL schema; verify item IDs/table names for your MIMIC-IV version.

## Data & ethics
MIMIC-IV and the PhysioNet Challenge data are governed by data-use agreements. This repo contains no patient data.
