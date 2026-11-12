import os
import zipfile

import requests

from src.utils import load_config
from src.utils import project_path


def download_movielens():

    config = load_config()

    url = config["dataset"]["url"]

    output_dir = project_path(
        config["dataset"]["download_dir"]
    )

    os.makedirs(output_dir, exist_ok=True)

    zip_path = os.path.join(
        output_dir,
        "ml-100k.zip"
    )

    extracted = project_path(
        config["dataset"]["folder"]
    )

    if os.path.exists(extracted):

        print("MovieLens already downloaded.")

        return

    response = requests.get(url)

    with open(zip_path, "wb") as f:

        f.write(response.content)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:

        zip_ref.extractall(output_dir)

    print("MovieLens downloaded.")
