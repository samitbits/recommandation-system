import os

import joblib
from surprise import Dataset
from surprise import Reader
from surprise import SVD

from src.utils import load_config
from src.utils import project_path


class CollaborativeModel:

    def __init__(self):

        config = load_config()

        self.params = config["model"]

        self.models_dir = project_path(
            config["paths"]["models"]
        )

        os.makedirs(self.models_dir, exist_ok=True)

        self.seen = {}

    def fit(self, train_df):

        reader = Reader(rating_scale=(1, 5))

        dataset = Dataset.load_from_df(
            train_df[["user_id", "movie_id", "rating"]],
            reader
        )

        trainset = dataset.build_full_trainset()

        self.model = SVD(
            n_factors=self.params["n_factors"],
            lr_all=self.params["lr_all"],
            reg_all=self.params["reg_all"],
            random_state=self.params["random_state"]
        )

        self.model.fit(trainset)

        self.seen = (
            train_df
            .groupby("user_id")["movie_id"]
            .apply(set)
            .to_dict()
        )

        return self

    def predict(self, user_id, movie_id):

        return self.model.predict(
            user_id,
            movie_id
        ).est

    def recommend(self, user_id, all_items, top_n=10):

        already_seen = self.seen.get(user_id, set())

        predictions = []

        for item in all_items:

            if item in already_seen:

                continue

            predicted_rating = self.predict(user_id, item)

            predictions.append((item, predicted_rating))

        predictions.sort(
            key=lambda x: x[1],
            reverse=True
        )

        return [
            movie
            for movie, score
            in predictions[:top_n]
        ]

    def save_model(self):

        joblib.dump(
            self,
            os.path.join(
                self.models_dir,
                "svd_model.pkl"
            )
        )
