import pandas as pd


class DataIngestion:

    BASE = "data/raw/ml-100k"

    def load_users(self):

        return pd.read_csv(

            f"{self.BASE}/u.user",

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
        ]

        return pd.read_csv(

            f"{self.BASE}/u.item",

            sep="|",

            encoding="latin-1",

            header=None,

            usecols=range(5),

            names=columns
        )

    def load_ratings(self):

        return pd.read_csv(

            f"{self.BASE}/u.data",

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