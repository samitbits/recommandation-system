import requests
import zipfile
import os


def download_movielens():

    url = (
        "https://files.grouplens.org/"
        "datasets/movielens/ml-100k.zip"
    )

    output_dir = "data/raw"

    zip_path = os.path.join(
        output_dir,
        "ml-100k.zip"
    )

    extracted = os.path.join(
        output_dir,
        "ml-100k"
    )

    if os.path.exists(extracted):

        print(
            "MovieLens already downloaded."
        )

        return

    response = requests.get(url)

    with open(
            zip_path,
            "wb"
    ) as f:

        f.write(response.content)

    with zipfile.ZipFile(
            zip_path,
            "r"
    ) as zip_ref:

        zip_ref.extractall(
            output_dir
        )

    print(
        "MovieLens downloaded."
    )