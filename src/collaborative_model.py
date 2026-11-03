from surprise import Dataset
from surprise import Reader
from surprise import SVD

import joblib


class CollaborativeModel:

    def fit(
            self,
            train_df
    ):

        reader = Reader(
            rating_scale=(1, 5)
        )

        dataset = Dataset.load_from_df(

            train_df[
                [
                    "user_id",
                    "movie_id",
                    "rating"
                ]
            ],

            reader
        )

        trainset = (

            dataset
            .build_full_trainset()
        )

        self.model = SVD(

            n_factors=100,

            lr_all=0.005,

            reg_all=0.02,

            random_state=42
        )

        self.model.fit(
            trainset
        )

        return self

    def predict(
            self,
            user_id,
            movie_id
    ):

        return self.model.predict(

            user_id,

            movie_id

        ).est

    def recommend(
            self,
            user_id,
            all_items,
            top_n=10
    ):

        predictions = []

        for item in all_items:

            predicted_rating = (

                self.predict(
                    user_id,
                    item
                )
            )

            predictions.append(

                (
                    item,
                    predicted_rating
                )
            )

        predictions.sort(

            key=lambda x: x[1],

            reverse=True
        )

        recommendations = [

            movie

            for movie, score

            in predictions[:top_n]
        ]

        return recommendations

    def save_model(self):

        joblib.dump(

            self,

            "models/svd_model.pkl"
        )