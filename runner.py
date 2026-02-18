import mlflow

from scripts import evaluate, process_data, train

MLFLOW_TRACKING_URI = "http://158.160.2.37:5000"
EXPERIMENT_NAME = "homework_chuchev"


if __name__ == "__main__":
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="C_rf_max_depth=10"):
        process_data()
        train()
        evaluate()
