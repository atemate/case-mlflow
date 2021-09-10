import os

import mlflow.pyfunc
import pandas as pd

model_name = os.getenv("MY_MODEL_NAME", "iris")
model_stage = os.getenv("MY_MODEL_STAGE", "Staging")

model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_stage}")

df = pd.read_json("input_example.json", orient="split")
df["predicted_class"] = model.predict(df)
print(df)
