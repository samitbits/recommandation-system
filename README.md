# End-to-End Data Management Pipeline for a Recommendation System

A complete, reproducible recommendation pipeline built on the **MovieLens 100K** dataset. It
ingests raw data, validates quality, prepares and transforms it, engineers reusable features,
trains two models (a popularity baseline and an SVD collaborative-filtering model), evaluates and
compares them, and automates the whole workflow with **Prefect**. Experiments are tracked with
**MLflow**.

---

## Pipeline Stages

```
Data Ingestion -> Data Validation -> Data Preparation/Transformation ->
Feature Management -> Model Training -> Model Evaluation -> Orchestration
```

See `docs/architecture.md` for the architecture diagram.

---

## Project Structure

```
recommandation-system/
├── config/
│   └── config.yaml              # Central configuration (paths, model params, split)
├── data/
│   ├── raw/                     # Downloaded MovieLens files
│   ├── processed/               # train/test parquet + user-item matrix
│   ├── features/                # user/item/interaction feature tables
│   └── errors/                  # Separated invalid/duplicate records
├── docs/
│   ├── architecture.md          # Architecture diagram
│   └── final_report.md          # Design decisions, results, limitations
├── models/                      # Saved popularity + SVD models (.pkl)
├── orchestration/
│   └── prefect_pipeline.py      # End-to-end Prefect flow
├── reports/
│   ├── validation_report.csv    # Data validation results
│   ├── feature_catalog.md/.csv  # Feature catalog
│   ├── model_comparison.csv     # Model evaluation comparison
│   └── orchestration_run.log    # Captured successful Prefect run
├── src/                         # Pipeline modules
├── requirements.txt
├── run_pipeline.py              # Sequential end-to-end entry point
└── README.md
```

---

## Installation

### 1. Create and activate a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1          # Windows PowerShell
# source venv/bin/activate           # macOS/Linux
```

### 2. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> Requires Python 3.11. All pinned versions ship prebuilt wheels, so no C compiler is needed.

---

## Running the Pipeline

### Option A — Sequential run

```powershell
python run_pipeline.py
```

### Option B — Prefect orchestration (end-to-end flow)

```powershell
python .\orchestration\prefect_pipeline.py
```

Both produce the same artifacts. The dataset is downloaded automatically on first run.

---

## Outputs

| Artifact | Location |
|----------|----------|
| Train/test splits + user-item matrix | `data/processed/`, `data/features/` |
| Feature tables | `data/features/*.parquet` |
| Validation report | `reports/validation_report.csv` |
| Separated error records | `data/errors/*.csv` |
| Feature catalog | `reports/feature_catalog.md` / `.csv` |
| Model comparison | `reports/model_comparison.csv` |
| Saved models | `models/*.pkl` |
| MLflow experiment tracking | `mlruns/` |

### View MLflow experiments

```powershell
mlflow ui --backend-store-uri mlruns
# then open http://127.0.0.1:5000
```

---

## Models

| Model | Type | Description |
|-------|------|-------------|
| Popularity | Baseline | Recommends the globally most-rated items. |
| SVD | Collaborative filtering (matrix factorization, `scikit-surprise`) | Personalized recommendations; filters already-seen items. |

## Evaluation Metrics

Ranking metrics at **K = 10**: Precision@K, Recall@K, NDCG@K, Coverage, Diversity.
For explicit-rating prediction (SVD): RMSE and MAE. Results are written to
`reports/model_comparison.csv` and logged to MLflow. See `docs/final_report.md` for discussion.

---

## Configuration

All paths, model hyperparameters, the train/test split fraction, and evaluation `top_k` live in
`config/config.yaml` and are loaded centrally via `src/utils.load_config()`.

---

## Feature Store

`src/feature_store.py` provides a reusable retrieval layer (`FeatureStore`) that loads the
persisted feature tables and serves features by `user_id` / `movie_id`, so the same features are
available for both training and prediction.

---

## Tests

```powershell
python -m pytest -q
```
