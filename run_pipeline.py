from src.download_dataset import (
    download_movielens
)

from src.utils import (
    ensure_directories
)

from src.ingestion import (
    DataIngestion
)

from src.validation import (
    DataValidator
)

from src.preparation import (
    DataPreparation
)

from src.feature_engineering import (
    FeatureEngineering
)

from src.popularity_model import (
    PopularityModel
)

from src.collaborative_model import (
    CollaborativeModel
)

from src.mlflow_tracking import (
    MLflowTracker
)

from src.evaluation import (
    Evaluation
)

import pandas as pd


def main():

    print("Pipeline Started")

    ensure_directories()

    download_movielens()

    ingestion = DataIngestion()

    users, items, ratings = (

        ingestion.ingest_all()
    )

    validator = DataValidator()

    validator.validate(

        users,

        items,

        ratings
    )

    prep = DataPreparation()

    train, test = prep.prepare(
        ratings
    )

    fe = FeatureEngineering()

    user_features = (

        fe.create_user_features(
            train
        )
    )

    item_features = (

        fe.create_item_features(
            train
        )
    )

    interaction_features = (

        fe.create_interaction_features(
            train
        )
    )

    fe.save_features(

        user_features,

        item_features,

        interaction_features
    )

    popularity_model = (

        PopularityModel()
    )

    popularity_model.fit(
        train
    )

    popularity_model.save_model()

    svd_model = (

        CollaborativeModel()
    )

    svd_model.fit(
        train
    )

    svd_model.save_model()

    tracker = MLflowTracker()

    tracker.log_experiment(

        model_name="SVD",

        params={

            "n_factors":100,

            "lr_all":0.005,

            "reg_all":0.02
        },

        metrics={

            "dummy_metric":1
        }
    )

    print(
        "Pipeline Completed"
    )


if __name__ == "__main__":

    main()