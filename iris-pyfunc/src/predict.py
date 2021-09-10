import os
from argparse import ArgumentParser

import mlflow.pyfunc
import pandas as pd


def get_parser():
    p = ArgumentParser(description="Run model prediction on a given input")
    p.add_argument(
        "--json-file",
        help="Path to the input file in json format",
        default="input_example.json",
    )
    return p


def main(args):
    model_name = os.getenv("MY_MODEL_NAME", "iris")
    model_stage = os.getenv("MY_MODEL_STAGE", "Staging")

    model_uri = f"models:/{model_name}/{model_stage}"
    model = mlflow.pyfunc.load_model(model_uri=model_uri)

    df = pd.read_json(args.json_file, orient="split")
    df["predicted_class"] = model.predict(df)

    print(df)


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
