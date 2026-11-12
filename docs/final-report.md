# Final Report: End-to-End Recommendation Pipeline

## 1. Overview

This project implements a complete, reproducible data-management pipeline for a movie
recommendation system using the **MovieLens 100K** dataset. The pipeline covers the full lifecycle:
ingestion, validation, preparation/transformation, feature management, model training, evaluation,
and orchestration, with experiment tracking in MLflow.

Two entry points are provided:
- `run_pipeline.py` — a sequential run.
- `orchestration/prefect_pipeline.py` — the same stages wired as a **Prefect** flow with explicit
  task dependencies.

## 2. Design Decisions

**Dataset.** MovieLens 100K is a well-understood public dataset with explicit 1–5 ratings, user
demographics, and item genres — ideal for demonstrating collaborative filtering and content signals.

**Configuration-first.** All paths, hyperparameters, the split fraction, and evaluation `top_k`
live in `config/config.yaml` and are loaded via `src/utils.load_config()`. Modules resolve paths
relative to the project root, so the pipeline runs from any working directory.

**Validation returns clean data.** The validator checks user/item/rating schemas, missing values,
duplicates, and referential integrity. Invalid and duplicate records are written to `data/errors/`
and **removed** from the data passed downstream, so models only train on valid rows.

**Time-based split.** To avoid look-ahead leakage, the train/test split is chronological: the most
recent 20% of interactions (by timestamp) form the test set. Features are computed from the
**training split only**.

**Feature management.** User, item, and interaction features are generated and persisted as Parquet.
A `FeatureStore` retrieval layer serves features by `user_id`/`movie_id` for both training and
inference. A machine-readable **feature catalog** is generated at `reports/feature_catalog.md/.csv`.

**Two models.** A **popularity baseline** (global most-rated items) and a personalized **SVD**
matrix-factorization model (`scikit-surprise`). SVD filters already-seen items from its
recommendations.

**Experiment tracking.** Each model run logs its parameters, evaluation metrics, and the serialized
model artifact to MLflow (`mlruns/`).

## 3. Tools Used

| Concern | Tool |
|---------|------|
| Data manipulation | pandas, numpy |
| Validation | Pandera |
| Modeling | scikit-surprise (SVD), scikit-learn |
| Experiment tracking | MLflow |
| Orchestration | Prefect 3 |
| Storage | Parquet (pyarrow), CSV |

## 4. Results

Evaluation at **K = 10** on the chronological test set (see `reports/model_comparison.csv`):

| Model | Precision@10 | Recall@10 | NDCG@10 | Coverage | Diversity | RMSE | MAE |
|-------|-------------|-----------|---------|----------|-----------|------|-----|
| Popularity | 0.212 | 0.073 | 0.232 | 0.006 | 0.725 | – | – |
| SVD | 0.169 | 0.049 | 0.153 | 0.096 | 0.752 | 1.030 | 0.821 |

**Interpretation.**
- The **popularity baseline** scores higher on top-K ranking metrics. This is expected on MovieLens:
  popular blockbusters are rated by most users, so a global top-10 list overlaps strongly with each
  user's held-out high-rated items.
- **SVD** provides much higher **coverage** (~16x) and slightly higher **diversity**, meaning it
  recommends a far broader catalog and more varied genres rather than the same few blockbusters.
- Only **SVD** predicts explicit ratings, achieving **RMSE 1.03 / MAE 0.82**.
- The trade-off is classic accuracy-vs-coverage: popularity wins raw hit-rate; SVD personalizes and
  diversifies.

## 5. Limitations and Future Work

- **Top-K ranking from SVD** is computed by scoring all candidate items per user, which is
  acceptable at 100K scale but would need approximate nearest-neighbor retrieval at larger scale.
- **Cold-start:** SVD cannot score unseen users/items; a content-based or hybrid fallback using the
  available genre features would help.
- **No hyperparameter search** — SVD uses fixed, config-driven parameters.
- **Relevance threshold** for ranking metrics is fixed at rating >= 4; sensitivity analysis would be
  valuable.
- **Diversity** is measured on genre vectors of recommended items; richer item embeddings would give
  a finer-grained signal.
- A content-based model (using the preserved genre features) was scoped but not trained; it is the
  natural next addition to form a hybrid recommender.

## 6. Reproducibility

- Pinned `requirements.txt` with wheel-available versions (no compiler needed).
- Deterministic SVD (`random_state`) and a deterministic time-based split.
- One-command execution via `run_pipeline.py` or the Prefect flow; all artifacts are regenerated.
