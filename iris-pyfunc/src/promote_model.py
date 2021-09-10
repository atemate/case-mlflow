import os

from mlflow.tracking import MlflowClient


def get_last_model_version(mlflow_client, model_name) -> int:
    info_list = mlflow_client.search_model_versions(f"name='{model_name}'")
    version = info_list[-1].version
    return version


if __name__ == "__main__":
    model_name = os.getenv("MY_MODEL_NAME", "iris")
    model_stage = os.getenv("MY_MODEL_STAGE", "Staging")

    mlflow_client = MlflowClient()
    latest_version = get_last_model_version(mlflow_client, model_name)

    mlflow_client.transition_model_version_stage(
        model_name,
        version=latest_version,
        stage=model_stage,
    )
    print(f"Transitioned model {model_name} version {latest_version} to {model_stage}")
