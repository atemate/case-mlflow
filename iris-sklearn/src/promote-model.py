from mlflow.tracking import MlflowClient
import os

model_name = os.getenv("MY_MODEL_NAME", "iris")
model_stage = os.getenv("MY_MODEL_STAGE", "Staging")


client = MlflowClient()
registered_model_info_list = client.list_registered_models()
registered_model_info = registered_model_info_list[-1]

version = registered_model_info.latest_versions[-1].version

client.transition_model_version_stage(
    model_name,
    version=version,
    stage=model_stage,
)
print(f"Transitioned model {model_name} version {version} to {model_stage}")
