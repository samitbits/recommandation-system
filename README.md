# End-to-End Data Management Pipeline for Recommendation System

## Objective

Build a complete recommendation pipeline for MovieLens 100K dataset including:

- Data Ingestion
- Data Validation
- Data Preparation
- Feature Engineering
- Popularity Recommender
- Collaborative Filtering (SVD)
- Evaluation
- MLflow Tracking
- Prefect Orchestration

---

## Project Structure

recommendation-pipeline/

data/

models/

reports/

src/

orchestration/

config/

README.md

requirements.txt

run_pipeline.py

---

## Installation

### Create Virtual Environment
python -m venv venv
## Activate
venv\\Scripts\\activate   (for Windows)
source venv/bin/activate  (for Linux)
## Install Dependencies
pip install -r requirements.txt
## Execute Project
python run_pipeline.py
## MLflow
mlflow ui
Open http://127.0.0.1:5000
## Prefect
prefect server start (to start the server)
python orchestration/prefect_pipeline.py (to execute the flow)
Dashboard  http://127.0.0.1:4200
