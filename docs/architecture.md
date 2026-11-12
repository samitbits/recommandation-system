# Pipeline Architecture

## High-Level Flow

```mermaid
flowchart TD
    A["Data Ingestion<br>MovieLens 100K: users, items, ratings"] --> B["Data Validation<br>schemas, missing, duplicates,<br>referential integrity"]

    B -->|clean ratings| C["Data Preparation<br>time-based train/test split<br>user-item matrix"]
    B -->|invalid/duplicate rows| E[(data/errors)]

    C --> D["Feature Engineering<br>user / item / interaction features<br>+ feature catalog"]

    D --> F[Model Training]

    F --> F1[Popularity Baseline]
    F --> F2["SVD Collaborative Filtering"]

    F1 --> G["Evaluation<br>Precision@K, Recall@K, NDCG@K,<br>Coverage, Diversity, RMSE, MAE"]
    F2 --> G

    G --> H["Model Comparison Table<br>reports/model_comparison.csv"]
    G --> I["MLflow Tracking<br>params + metrics + artifacts"]

    D --> J["Feature Store<br>reusable retrieval layer"]

    subgraph Orchestration["Prefect Flow: RecommendationPipeline"]
        A
        B
        C
        D
        F
        G
        I
    end
```

## Storage Layout

```mermaid
flowchart LR
    R["data/raw"] --> P["data/processed<br>train/test + matrix"]
    P --> FE["data/features<br>user/item/interaction"]

    V["reports<br>validation, catalog, comparison"]
    ER["data/errors<br>invalid + duplicate records"]
    M["models<br>popularity + svd.pkl"]
    ML["mlruns<br>MLflow store"]
```
## Components

| Layer | Module | Responsibility |
|-------|--------|----------------|
| Config | `src/utils.py` + `config/config.yaml` | Central config + path helpers |
| Ingestion | `src/download_dataset.py`, `src/ingestion.py` | Download + load users/items/ratings (incl. genres) |
| Validation | `src/validation.py` | Schema + integrity checks; separate error records; return clean data |
| Preparation | `src/preparation.py` | Time-based split, user-item matrix |
| Features | `src/feature_engineering.py`, `src/feature_store.py` | Feature creation, catalog, retrieval layer |
| Models | `src/popularity_model.py`, `src/collaborative_model.py` | Baseline + SVD |
| Evaluation | `src/evaluation.py` | Ranking + rating metrics, comparison table |
| Tracking | `src/mlflow_tracking.py` | MLflow params/metrics/artifacts |
| Orchestration | `orchestration/prefect_pipeline.py` | Prefect flow wiring all stages |

