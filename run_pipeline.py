#run_pipeline.py
import os

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


def main():

    print("Pipeline Started")

    config = load_config()

    ensure_directories()

    download_movielens()

    users, items, ratings = DataIngestion().ingest_all()

    ratings = DataValidator().validate(users, items, ratings)

    train, test = DataPreparation().prepare(ratings)

    fe = FeatureEngineering()

    user_features = fe.create_user_features(train, items)
    item_features = fe.create_item_features(train, items)
    interaction_features = fe.create_interaction_features(train)

    fe.save_features(
        user_features,
        item_features,
        interaction_features
    )

    popularity_model = PopularityModel()
    popularity_model.fit(train)
    popularity_model.save_model()

    svd_model = CollaborativeModel()
    svd_model.fit(train)
    svd_model.save_model()

    comparison = Evaluation().run_comparison(
        train,
        test,
        items,
        popularity_model,
        svd_model
    )

    print(comparison.to_string(index=False))

    tracker = MLflowTracker()

    models_dir = project_path(config["paths"]["models"])

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

    print("Pipeline Completed")


if __name__ == "__main__":

    main()
