
import os

import pandas as pd

from src.ingestion import GENRE_COLUMNS
from src.utils import load_config
from src.utils import project_path


class FeatureEngineering:

    def __init__(self):

        config = load_config()

        self.features_dir = project_path(
            config["paths"]["features"]
        )

        self.reports_dir = project_path(
            config["paths"]["reports"]
        )

        os.makedirs(self.features_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)

    def create_user_features(self, ratings, items=None):

        latest_date = ratings["date"].max()

        user_features = (
            ratings
            .groupby("user_id")
            .agg(
                user_avg_rating=("rating", "mean"),
                user_activity_count=("rating", "count"),
                last_interaction=("date", "max")
            )
            .reset_index()
        )

        user_features["days_since_last_activity"] = (
            latest_date - user_features["last_interaction"]
        ).dt.days

        if items is not None:

            user_features = user_features.merge(
                self._user_preferred_category(ratings, items),
                on="user_id",
                how="left"
            )

        return user_features

    def _user_preferred_category(self, ratings, items):

        merged = ratings.merge(
            items[["movie_id"] + GENRE_COLUMNS],
            on="movie_id",
            how="left"
        )

        genre_totals = (
            merged
            .groupby("user_id")[GENRE_COLUMNS]
            .sum()
        )

        preferred = (
            genre_totals
            .idxmax(axis=1)
            .rename("preferred_category")
            .reset_index()
        )

        return preferred

    def create_item_features(self, ratings, items=None):

        item_features = (
            ratings
            .groupby("movie_id")
            .agg(
                item_avg_rating=("rating", "mean"),
                item_popularity=("rating", "count")
            )
            .reset_index()
        )

        if items is not None:

            item_features = item_features.merge(
                self._item_category(items),
                on="movie_id",
                how="left"
            )

        return item_features

    def _item_category(self, items):

        def primary_genre(row):

            active = [
                g for g in GENRE_COLUMNS if row[g] == 1
            ]

            return active[0] if active else "unknown"

        categories = items[["movie_id"] + GENRE_COLUMNS].copy()

        categories["item_category"] = categories.apply(
            primary_genre,
            axis=1
        )

        return categories[["movie_id", "item_category"]]

    def create_interaction_features(self, ratings):

        latest_date = ratings["date"].max()

        interaction_features = ratings.copy()

        interaction_features["interaction_age"] = (
            latest_date - interaction_features["date"]
        ).dt.days

        return interaction_features

    def create_user_item_matrix(self, ratings):

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
            os.path.join(
                self.features_dir,
                "user_features.parquet"
            ),
            index=False
        )

        item_features.to_parquet(
            os.path.join(
                self.features_dir,
                "item_features.parquet"
            ),
            index=False
        )

        interaction_features.to_parquet(
            os.path.join(
                self.features_dir,
                "interaction_features.parquet"
            ),
            index=False
        )

        self.generate_catalog()

        print("Features stored successfully")

    def generate_catalog(self):

        catalog = [
            ["user", "user_id", "int", "Unique user identifier"],
            ["user", "user_avg_rating", "float", "Mean rating the user has given"],
            ["user", "user_activity_count", "int", "Number of interactions by the user"],
            ["user", "last_interaction", "datetime", "Timestamp of the user's most recent interaction"],
            ["user", "days_since_last_activity", "int", "Days between the user's last interaction and the dataset's latest date"],
            ["user", "preferred_category", "categorical", "Genre the user has interacted with most often"],
            ["item", "movie_id", "int", "Unique item (movie) identifier"],
            ["item", "item_avg_rating", "float", "Mean rating received by the item"],
            ["item", "item_popularity", "int", "Number of interactions received by the item"],
            ["item", "item_category", "categorical", "Primary genre assigned to the item"],
            ["interaction", "rating", "int", "Explicit rating (1-5) given in the interaction"],
            ["interaction", "date", "datetime", "Interaction date derived from the raw timestamp"],
            ["interaction", "interaction_age", "int", "Days between the interaction and the dataset's latest date"],
        ]

        df = pd.DataFrame(
            catalog,
            columns=[
                "entity",
                "feature",
                "dtype",
                "description"
            ]
        )

        df.to_csv(
            os.path.join(
                self.reports_dir,
                "feature_catalog.csv"
            ),
            index=False
        )

        self._write_catalog_markdown(df)

    def _write_catalog_markdown(self, df):

        lines = [
            "# Feature Catalog",
            "",
            "Reusable features generated by the pipeline (computed from the"
            " training split only to avoid leakage).",
            "",
            "| Entity | Feature | Type | Description |",
            "|--------|---------|------|-------------|",
        ]

        for _, row in df.iterrows():

            lines.append(
                f"| {row['entity']} | {row['feature']} | "
                f"{row['dtype']} | {row['description']} |"
            )

        path = os.path.join(
            self.reports_dir,
            "feature_catalog.md"
        )

        with open(path, "w", encoding="utf-8") as f:

            f.write("\n".join(lines) + "\n")
