import pandas as pd
import numpy as np


class FeatureEngineering:

    def create_user_features(
            self,
            ratings
    ):

        latest_date = ratings[
            "date"
        ].max()

        user_features = (

            ratings

            .groupby("user_id")

            .agg(

                user_avg_rating=(
                    "rating",
                    "mean"
                ),

                user_activity_count=(
                    "rating",
                    "count"
                ),

                last_interaction=(
                    "date",
                    "max"
                )
            )

            .reset_index()
        )

        user_features[
            "days_since_last_activity"
        ] = (

            latest_date
            -
            user_features[
                "last_interaction"
            ]

        ).dt.days

        return user_features

    def create_item_features(
            self,
            ratings
    ):

        item_features = (

            ratings

            .groupby("movie_id")

            .agg(

                item_avg_rating=(

                    "rating",
                    "mean"
                ),

                item_popularity=(

                    "rating",
                    "count"
                )
            )

            .reset_index()
        )

        return item_features

    def create_interaction_features(
            self,
            ratings
    ):

        latest_date = ratings[
            "date"
        ].max()

        interaction_features = ratings.copy()

        interaction_features[
            "interaction_age"
        ] = (

            latest_date
            -
            interaction_features[
                "date"
            ]

        ).dt.days

        return interaction_features

    def create_user_item_matrix(
            self,
            ratings
    ):

        return ratings.pivot_table(

            index="user_id",

            columns="movie_id",

            values="rating",

            fill_value=0
        )

    def save_features(
            self,
            user_features,
            item_features,
            interaction_features
    ):

        user_features.to_parquet(

            "data/features/user_features.parquet",

            index=False
        )

        item_features.to_parquet(

            "data/features/item_features.parquet",

            index=False
        )

        interaction_features.to_parquet(

            "data/features/interaction_features.parquet",

            index=False
        )

        print(
            "Features stored successfully"
        )