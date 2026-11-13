import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity

from src.ingestion import GENRE_COLUMNS
from src.utils import load_config
from src.utils import project_path


RELEVANT_THRESHOLD = 4


class Evaluation:

    def __init__(self):

        config = load_config()

        self.k = config["evaluation"]["top_k"]

        self.reports_dir = project_path(
            config["paths"]["reports"]
        )

        os.makedirs(self.reports_dir, exist_ok=True)

    def precision_at_k(self, recommended, relevant, k=10):

        recommended = recommended[:k]

        hits = len(set(recommended) & set(relevant))

        return hits / k

    def recall_at_k(self, recommended, relevant, k=10):

        recommended = recommended[:k]

        hits = len(set(recommended) & set(relevant))

        if len(relevant) == 0:

            return 0.0

        return hits / len(relevant)

    def ndcg_at_k(self, recommended, relevant, k=10):

        dcg = 0

        for idx, item in enumerate(recommended[:k]):

            if item in relevant:

                dcg += 1 / np.log2(idx + 2)

        ideal = min(len(relevant), k)

        idcg = sum(
            1 / np.log2(i + 2)
            for i in range(ideal)
        )

        if idcg == 0:

            return 0.0

        return dcg / idcg

    def coverage(self, recommended_items, all_items):

        return (
            len(set(recommended_items))
            / len(set(all_items))
        )

    def diversity(self, feature_matrix):

        if len(feature_matrix) < 2:

            return 0.0

        similarity_matrix = cosine_similarity(feature_matrix)

        return 1 - similarity_matrix.mean()

    def rmse(self, actual, predicted):

        return np.sqrt(
            mean_squared_error(actual, predicted)
        )

    def mae(self, actual, predicted):

        return mean_absolute_error(actual, predicted)

    def _relevant_by_user(self, test):

        relevant = test[test["rating"] >= RELEVANT_THRESHOLD]

        return (
            relevant
            .groupby("user_id")["movie_id"]
            .apply(set)
            .to_dict()
        )

    def _ranking_metrics(
        self,
        recommend_fn,
        relevant_by_user,
        all_recommended,
        genre_lookup,
        all_items
    ):

        precisions = []
        recalls = []
        ndcgs = []

        for user_id, relevant in relevant_by_user.items():

            recommended = recommend_fn(user_id)

            if not recommended:

                continue

            all_recommended.update(recommended)

            precisions.append(
                self.precision_at_k(
                    recommended,
                    relevant,
                    self.k
                )
            )

            recalls.append(
                self.recall_at_k(
                    recommended,
                    relevant,
                    self.k
                )
            )

            ndcgs.append(
                self.ndcg_at_k(
                    recommended,
                    relevant,
                    self.k
                )
            )

        genre_matrix = genre_lookup.reindex(
            sorted(all_recommended)
        ).dropna().values

        return {
            f"precision_at_{self.k}": float(np.mean(precisions)) if precisions else 0.0,
            f"recall_at_{self.k}": float(np.mean(recalls)) if recalls else 0.0,
            f"ndcg_at_{self.k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
            "coverage": self.coverage(all_recommended, all_items),
            "diversity": float(self.diversity(genre_matrix)),
        }

    def run_comparison(
        self,
        train,
        test,
        items,
        popularity_model,
        svd_model
    ):

        all_items = sorted(
            set(train["movie_id"]) | set(test["movie_id"])
        )

        relevant_by_user = self._relevant_by_user(test)

        genre_lookup = (
            items
            .set_index("movie_id")[GENRE_COLUMNS]
        )

        pop_recs = popularity_model.recommend(
            user_id=None,
            top_n=self.k
        )

        pop_metrics = self._ranking_metrics(
            recommend_fn=lambda uid: pop_recs,
            relevant_by_user=relevant_by_user,
            all_recommended=set(),
            genre_lookup=genre_lookup,
            all_items=all_items
        )

        svd_metrics = self._ranking_metrics(
            recommend_fn=lambda uid: svd_model.recommend(
                uid,
                all_items,
                top_n=self.k
            ),
            relevant_by_user=relevant_by_user,
            all_recommended=set(),
            genre_lookup=genre_lookup,
            all_items=all_items
        )

        svd_rating_metrics = self._svd_rating_errors(
            test,
            svd_model
        )

        svd_metrics.update(svd_rating_metrics)

        pop_metrics.update({"rmse": None, "mae": None})

        comparison = pd.DataFrame(
            [
                {"model": "Popularity", **pop_metrics},
                {"model": "SVD", **svd_metrics},
            ]
        )

        out_path = os.path.join(
            self.reports_dir,
            "model_comparison.csv"
        )

        comparison.to_csv(out_path, index=False)

        print("Evaluation completed.")

        return comparison

    def _svd_rating_errors(self, test, svd_model):

        actual = []
        predicted = []

        for row in test.itertuples(index=False):

            actual.append(row.rating)

            predicted.append(
                svd_model.predict(
                    row.user_id,
                    row.movie_id
                )
            )

        return {
            "rmse": float(self.rmse(actual, predicted)),
            "mae": float(self.mae(actual, predicted)),
        }
