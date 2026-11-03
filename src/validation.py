import pandas as pd
import pandera as pa
from pandera import Check
from pandera import Column


class DataValidator:

    @staticmethod
    def schema():

        return pa.DataFrameSchema({

            "user_id":

                Column(
                    int,
                    Check.ge(1)
                ),

            "movie_id":

                Column(
                    int,
                    Check.ge(1)
                ),

            "rating":

                Column(
                    int,

                    Check.in_range(
                        1,
                        5
                    )
                ),

            "timestamp":

                Column(
                    int,
                    Check.ge(1)
                )
        })

    def validate(
            self,
            users,
            items,
            ratings
    ):

        report = []

        report.append([
            "missing_values",
            ratings.isna().sum().sum()
        ])

        report.append([
            "duplicates",
            ratings.duplicated().sum()
        ])

        try:

            self.schema().validate(
                ratings
            )

            report.append([
                "schema_validation",
                "PASS"
            ])

        except Exception:

            report.append([
                "schema_validation",
                "FAIL"
            ])

            ratings.to_csv(
                "data/errors/schema_errors.csv",
                index=False
            )

        invalid_users = ratings[
            ~ratings["user_id"].isin(
                users["user_id"]
            )
        ]

        invalid_movies = ratings[
            ~ratings["movie_id"].isin(
                items["movie_id"]
            )
        ]

        invalid_users.to_csv(
            "data/errors/invalid_users.csv",
            index=False
        )

        invalid_movies.to_csv(
            "data/errors/invalid_movies.csv",
            index=False
        )

        pd.DataFrame(
            report,
            columns=[
                "check",
                "result"
            ]
        ).to_csv(
            "reports/validation_report.csv",
            index=False
        )

        print(
            "Validation completed."
        )