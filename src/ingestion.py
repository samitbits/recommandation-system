import pandas as pd

from src.utils import load_config
from src.utils import project_path


GENRE_COLUMNS = [
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western"
]


class DataIngestion:

    def __init__(self):

        config = load_config()

        self.base = project_path(
            config["dataset"]["folder"]
        )

    def load_users(self):

        return pd.read_csv(

            f"{self.base}/u.user",

            sep="|",

            names=[
                "user_id",
                "age",
                "gender",
                "occupation",
                "zipcode"
            ]
        )

    def load_items(self):

        columns = [
            "movie_id",
            "title",
            "release_date",
            "video_release_date",
            "imdb_url"
        ] + GENRE_COLUMNS

        return pd.read_csv(

            f"{self.base}/u.item",

            sep="|",

            encoding="latin-1",

            header=None,

            names=columns
        )

    def load_ratings(self):

        return pd.read_csv(

            f"{self.base}/u.data",

            sep="\t",

            names=[
                "user_id",
                "movie_id",
                "rating",
                "timestamp"
            ]
        )

    def ingest_all(self):

        users = self.load_users()

        items = self.load_items()

        ratings = self.load_ratings()

        return users, items, ratings
