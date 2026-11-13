#prefect_pipeline.py
import os
import sys

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from prefect import flow
from prefect import task

from src.collaborative_model import CollaborativeModel
from src.download_dataset import download_movielens
from src.evaluation import Evaluation
from src.feature_engineering import FeatureEngineering
from src.ingestion import DataIngestion
from src.mlflow_tracking import MLflowTracker
from src.popularity_model import PopularityModel
from src.preparation import DataPreparation
from src.utils import ensure_directories
from src.utils import load_config
from src.utils import project_path
from src.validation import DataValidator


@task
def setup_task():

    ensure_directories()
    download_movielens()


@task
def ingestion_task():

    return DataIngestion().ingest_all()


@task
def validation_task(users, items, ratings):

    return DataValidator().validate(users, items, ratings)


@task
def preparation_task(ratings):

    return DataPreparation().prepare(ratings)


@task
def feature_task(train, items):

    fe = FeatureEngineering()

    user_features = fe.create_user_features(train, items)
    item_features = fe.create_item_features(train, items)
    interaction_features = fe.create_interaction_features(train)

    fe.save_features(
        user_features,
        item_features,
        interaction_features
    )

    return True


@task
def train_models_task(train):

    popularity_model = PopularityModel()
    popularity_model.fit(train)
    popularity_model.save_model()

    svd_model = CollaborativeModel()
    svd_model.fit(train)
    svd_model.save_model()

    return popularity_model, svd_model


@task
def evaluation_task(
    train,
    test,
    items,
    popularity_model,
    svd_model
):

    return Evaluation().run_comparison(
        train,
        test,
        items,
        popularity_model,
        svd_model
    )


@task
def tracking_task(comparison):

    config = load_config()

    models_dir = project_path(config["paths"]["models"])

    tracker = MLflowTracker()

    for _, row in comparison.iterrows():

        model_name = row["model"]

        metrics = {
            key: row[key]
            for key in comparison.columns
            if key != "model"
        }

        if model_name == "SVD":

            params = config["model"]

            artifact = os.path.join(
                models_dir,
                "svd_model.pkl"
            )

        else:

            params = {"strategy": "most_popular"}

            artifact = os.path.join(
                models_dir,
                "popularity_model.pkl"
            )

        tracker.log_model_run(
            model_name=model_name,
            params=params,
            metrics=metrics,
            artifact_path=artifact
        )


@flow(name="RecommendationPipeline")
def recommendation_pipeline():

    setup_task()

    users, items, ratings = ingestion_task()

    valid_ratings = validation_task(users, items, ratings)

    train, test = preparation_task(valid_ratings)

    feature_task(train, items)

    popularity_model, svd_model = train_models_task(train)

    comparison = evaluation_task(
        train,
        test,
        items,
        popularity_model,
        svd_model
    )

    tracking_task(comparison)

    print("Prefect pipeline completed.")


if __name__ == "__main__":

    recommendation_pipeline()
