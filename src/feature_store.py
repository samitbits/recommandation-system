import os

import pandas as pd

from src.utils import load_config
from src.utils import project_path


class FeatureStore:
    """Reusable feature-retrieval layer for training and prediction.

    Loads the persisted feature tables once and exposes lookups by
    user_id / movie_id so the same features can be served at inference time.
    """

    def __init__(self):

        config = load_config()

        self.features_dir = project_path(
            config["paths"]["features"]
        )

        self._users = None
        self._items = None

    def _load(self, name):

        return pd.read_parquet(
            os.path.join(self.features_dir, name)
        )

    @property
    def user_features(self):

        if self._users is None:

            self._users = self._load(
                "user_features.parquet"
            ).set_index("user_id")

        return self._users

    @property
    def item_features(self):

        if self._items is None:

            self._items = self._load(
                "item_features.parquet"
            ).set_index("movie_id")

        return self._items

    def get_user_features(self, user_id):

        if user_id in self.user_features.index:

            return self.user_features.loc[user_id].to_dict()

        return None

    def get_item_features(self, movie_id):

        if movie_id in self.item_features.index:

            return self.item_features.loc[movie_id].to_dict()

        return None
