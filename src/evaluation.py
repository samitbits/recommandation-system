import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)
from sklearn.metrics.pairwise import (
    cosine_similarity
)


class Evaluation:

    def precision_at_k(

            self,

            recommended,

            relevant,

            k=10
    ):

        recommended = recommended[:k]

        hits = len(

            set(recommended)

            &

            set(relevant)
        )

        return hits / k

    def recall_at_k(

            self,

            recommended,

            relevant,

            k=10
    ):

        recommended = recommended[:k]

        hits = len(

            set(recommended)

            &

            set(relevant)
        )

        return hits / len(relevant)

    def ndcg_at_k(

            self,

            recommended,

            relevant,

            k=10
    ):

        dcg = 0

        for idx, item in enumerate(

                recommended[:k]
        ):

            if item in relevant:

                dcg += (

                    1 /

                    np.log2(
                        idx + 2
                    )
                )

        ideal = min(
            len(relevant),
            k
        )

        idcg = sum(

            1 / np.log2(i + 2)

            for i in range(ideal)

        )

        return dcg / idcg

    def coverage(

            self,

            recommended_items,

            all_items
    ):

        return (

            len(

                set(
                    recommended_items
                )
            )

            /

            len(
                set(all_items)
            )

        )

    def diversity(

            self,

            feature_matrix
    ):

        similarity_matrix = (

            cosine_similarity(
                feature_matrix
            )
        )

        diversity_score = (

            1
            -
            similarity_matrix.mean()
        )

        return diversity_score

    def rmse(

            self,

            actual,

            predicted
    ):

        return np.sqrt(

            mean_squared_error(

                actual,

                predicted
            )
        )

    def mae(

            self,

            actual,

            predicted
    ):

        return mean_absolute_error(

            actual,

            predicted
        )