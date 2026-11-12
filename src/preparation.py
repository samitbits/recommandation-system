#preparation.py
import os

import pandas as pd

from src.utils import load_config
from src.utils import project_path


class DataPreparation:

    def __init__(self):

        config = load_config()

        self.processed_dir = project_path(
            config["paths"]["processed"]
        )

        self.features_dir = project_path(
            config["paths"]["features"]
        )

        self.test_fraction = config["split"][
            "test_fraction"
        ]

        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.features_dir, exist_ok=True)

    def prepare(self, ratings):

        ratings = ratings.drop_duplicates().copy()

        ratings["date"] = pd.to_datetime(
            ratings["timestamp"],
            unit="s"
        )

        ratings = ratings.sort_values("date")

        quantile = 1 - self.test_fraction

        split_date = ratings["date"].quantile(quantile)

        train = ratings[ratings["date"] <= split_date]

        test = ratings[ratings["date"] > split_date]

        train.to_parquet(
            os.path.join(
                self.processed_dir,
                "train.parquet"
            )
        )

        test.to_parquet(
            os.path.join(
                self.processed_dir,
                "test.parquet"
            )
        )

        self._save_interaction_matrix(train)

        return train, test

    def _save_interaction_matrix(self, train):

        matrix = train.pivot_table(
            index="user_id",
            columns="movie_id",
            values="rating",
            fill_value=0
        )

        matrix.to_parquet(
            os.path.join(
                self.features_dir,
                "user_item_matrix.parquet"
            )
        )
