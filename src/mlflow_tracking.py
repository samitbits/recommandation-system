#mlflow_tracking.py
import mlflow

from src.utils import load_config
from src.utils import project_path


class MLflowTracker:

    def __init__(self):

        load_config()

        mlflow.set_tracking_uri(
            "file:" + project_path("mlruns")
        )

        mlflow.set_experiment("RecommendationSystem")

    def log_model_run(
        self,
        model_name,
        params,
        metrics,
        artifact_path=None
    ):

        with mlflow.start_run(run_name=model_name):

            mlflow.log_param("model", model_name)

            for key, value in params.items():

                mlflow.log_param(key, value)

            for key, value in metrics.items():

                if value is not None:

                    mlflow.log_metric(key, float(value))

            if artifact_path is not None:

                mlflow.log_artifact(artifact_path)

        print(f"Experiment logged: {model_name}")

    def log_experiment(self, model_name, params, metrics):

        self.log_model_run(
            model_name,
            params,
            metrics
        )
