import yaml
import os


def load_config():

    with open(
        "config/config.yaml",
        "r"
    ) as f:

        config = yaml.safe_load(f)

    return config


def ensure_directories():

    dirs = [

        "data/raw",

        "data/processed",

        "data/features",

        "data/errors",

        "reports",

        "models"
    ]

    for d in dirs:

        os.makedirs(
            d,
            exist_ok=True
        )