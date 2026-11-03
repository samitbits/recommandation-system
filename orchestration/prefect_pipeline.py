from prefect import flow
from prefect import task

from src.ingestion import DataIngestion
from src.validation import DataValidator
from src.preparation import DataPreparation
from src.feature_engineering import FeatureEngineering


@task
def ingestion_task():

    ingestion = DataIngestion()

    users, items, ratings = (

        ingestion.ingest_all()
    )

    return users, items, ratings


@task
def validation_task(
        users,
        items,
        ratings
):

    validator = DataValidator()

    validator.validate(

        users,

        items,

        ratings
    )

    return ratings


@task
def preparation_task(
        ratings
):

    prep = DataPreparation()

    train, test = prep.prepare(
        ratings
    )

    return train, test


@task
def feature_task(
        train
):

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

    return True


@flow(
    name="RecommendationPipeline"
)
def recommendation_pipeline():

    users, items, ratings = (

        ingestion_task()
    )

    validation_task(

        users,

        items,

        ratings
    )

    train, test = (

        preparation_task(
            ratings
        )
    )

    feature_task(
        train
    )

    print(
        "Prefect pipeline completed."
    )


if __name__ == "__main__":

    recommendation_pipeline()