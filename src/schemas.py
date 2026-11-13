#schemas.py
import pandera as pa
from pandera import Check
from pandera import Column


def ratings_schema():

    return pa.DataFrameSchema({
        "user_id": Column(int, Check.ge(1)),
        "movie_id": Column(int, Check.ge(1)),
        "rating": Column(int, Check.in_range(1, 5)),
        "timestamp": Column(int, Check.ge(1)),
    })


def users_schema():

    return pa.DataFrameSchema({
        "user_id": Column(int, Check.ge(1), unique=True),
        "age": Column(int, Check.in_range(1, 120), nullable=True),
        "gender": Column(str, nullable=True),
        "occupation": Column(str, nullable=True),
    })


def items_schema():

    return pa.DataFrameSchema({
        "movie_id": Column(int, Check.ge(1), unique=True),
        "title": Column(str, nullable=True),
    })
