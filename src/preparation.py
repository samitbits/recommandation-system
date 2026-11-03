import pandas as pd


class DataPreparation:

    def prepare(
            self,
            ratings
    ):

        ratings = ratings.drop_duplicates()

        ratings["date"] = pd.to_datetime(

            ratings["timestamp"],

            unit="s"
        )

        ratings = ratings.sort_values(
            "date"
        )

        split_date = ratings[
            "date"
        ].quantile(0.8)

        train = ratings[
            ratings["date"] <= split_date
        ]

        test = ratings[
            ratings["date"] > split_date
        ]

        train.to_parquet(
            "data/processed/train.parquet"
        )

        test.to_parquet(
            "data/processed/test.parquet"
        )

        return train, test