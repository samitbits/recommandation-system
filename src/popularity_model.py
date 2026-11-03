import joblib


class PopularityModel:

    def fit(
            self,
            ratings
    ):

        self.popularity = (

            ratings

            .groupby("movie_id")

            .size()

            .sort_values(
                ascending=False
            )
        )

        return self

    def recommend(
            self,
            user_id,
            top_n=10
    ):

        return (

            self.popularity

            .head(top_n)

            .index

            .tolist()
        )

    def save_model(
            self
    ):

        joblib.dump(

            self,

            "models/popularity_model.pkl"
        )