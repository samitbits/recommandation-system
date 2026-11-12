#popularity_model.py
import os

import joblib

from src.utils import load_config
from src.utils import project_path


class PopularityModel:

    def __init__(self):

        config = load_config()

        self.models_dir = project_path(
            config["paths"]["models"]
        )

        os.makedirs(self.models_dir, exist_ok=True)

    def fit(self, ratings):

        self.popularity = (
            ratings
            .groupby("movie_id")
            .size()
            .sort_values(ascending=False)
        )

        return self

    def recommend(self, user_id, top_n=10):

        return (
            self.popularity
            .head(top_n)
            .index
            .tolist()
        )

    def save_model(self):

        joblib.dump(
            self,
            os.path.join(
                self.models_dir,
                "popularity_model.pkl"
            )
        )
