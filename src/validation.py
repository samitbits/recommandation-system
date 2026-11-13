#validation.py
import os

import pandas as pd
import pandera as pa

from src import schemas
from src.utils import load_config
from src.utils import project_path


class DataValidator:

    def __init__(self):

        config = load_config()

        self.errors_dir = project_path(
            config["paths"]["errors"]
        )

        self.reports_dir = project_path(
            config["paths"]["reports"]
        )

        os.makedirs(self.errors_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)

    @staticmethod
    def ratings_schema():

        return schemas.ratings_schema()

    @staticmethod
    def users_schema():

        return schemas.users_schema()

    @staticmethod
    def items_schema():

        return schemas.items_schema()

    def _error_path(self, name):

        return os.path.join(self.errors_dir, name)

    def validate(self, users, items, ratings):

        report = []

        report.append(["ratings_rows_in", len(ratings)])
        report.append([
            "missing_values",
            int(ratings.isna().sum().sum())
        ])

        duplicates = ratings[
            ratings.duplicated(keep="first")
        ]

        report.append(["duplicates", len(duplicates)])

        if len(duplicates) > 0:

            duplicates.to_csv(
                self._error_path("duplicate_ratings.csv"),
                index=False
            )

        clean = ratings.drop_duplicates()

        for name, schema, frame in [
            ("users_schema", self.users_schema(), users),
            ("items_schema", self.items_schema(), items),
        ]:

            try:

                schema.validate(
                    frame,
                    lazy=True
                )

                report.append([name, "PASS"])

            except pa.errors.SchemaErrors as exc:

                report.append([name, "FAIL"])

                exc.failure_cases.to_csv(
                    self._error_path(f"{name}_errors.csv"),
                    index=False
                )

        try:

            self.ratings_schema().validate(
                clean,
                lazy=True
            )

            report.append(["ratings_schema", "PASS"])

            schema_bad_idx = set()

        except pa.errors.SchemaErrors as exc:

            report.append(["ratings_schema", "FAIL"])

            exc.failure_cases.to_csv(
                self._error_path("ratings_schema_errors.csv"),
                index=False
            )

            schema_bad_idx = set(
                exc.failure_cases["index"].dropna().astype(int)
            )

        invalid_users = clean[
            ~clean["user_id"].isin(users["user_id"])
        ]

        invalid_movies = clean[
            ~clean["movie_id"].isin(items["movie_id"])
        ]

        report.append([
            "orphan_user_refs",
            len(invalid_users)
        ])
        report.append([
            "orphan_movie_refs",
            len(invalid_movies)
        ])

        if len(invalid_users) > 0:

            invalid_users.to_csv(
                self._error_path("invalid_users.csv"),
                index=False
            )

        if len(invalid_movies) > 0:

            invalid_movies.to_csv(
                self._error_path("invalid_movies.csv"),
                index=False
            )

        bad_ref_idx = set(invalid_users.index) | set(
            invalid_movies.index
        )

        drop_idx = bad_ref_idx | (
            schema_bad_idx & set(clean.index)
        )

        valid = clean.drop(
            index=drop_idx,
            errors="ignore"
        )

        report.append(["valid_rows_out", len(valid)])

        pd.DataFrame(
            report,
            columns=["check", "result"]
        ).to_csv(
            os.path.join(
                self.reports_dir,
                "validation_report.csv"
            ),
            index=False
        )

        print("Validation completed.")

        return valid
