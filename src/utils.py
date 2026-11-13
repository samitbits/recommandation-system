#utils.py

import os

import yaml


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


def project_path(*parts):
    return os.path.join(PROJECT_ROOT, *parts)


_CONFIG_CACHE = None


def load_config():
    global _CONFIG_CACHE

    if _CONFIG_CACHE is None:

        config_file = project_path(
            "config",
            "config.yaml"
        )

        with open(config_file, "r") as f:

            _CONFIG_CACHE = yaml.safe_load(f)

    return _CONFIG_CACHE


def ensure_directories():
    config = load_config()

    for key in [
        "raw",
        "processed",
        "features",
        "errors",
        "reports",
        "models"
    ]:

        os.makedirs(
            project_path(config["paths"][key]),
            exist_ok=True
        )
