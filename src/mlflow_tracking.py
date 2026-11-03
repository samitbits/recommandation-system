import mlflow


class MLflowTracker:

    def log_experiment(

            self,

            model_name,

            metrics,

            params
    ):

        mlflow.set_experiment(

            "RecommendationSystem"
        )

        with mlflow.start_run():

            mlflow.log_param(
                "model",
                model_name
            )

            for k, v in params.items():

                mlflow.log_param(
                    k,
                    v
                )

            for k, v in metrics.items():

                mlflow.log_metric(
                    k,
                    v
                )

            print(

                f"Experiment logged: {model_name}"
            )