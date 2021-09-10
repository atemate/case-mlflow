import os
from argparse import ArgumentParser

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

CURRENT_DIR = os.path.dirname(__file__)
REQUIREMENTS_TXT = "{}/../requirements.txt".format(CURRENT_DIR)


def get_parser():
    p = ArgumentParser(description="Run model prediction on a given input")
    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--model-name", default=os.getenv("MY_MODEL_NAME", "iris"))
    return p


def main(args):
    max_depth = args.max_depth
    seed = args.seed
    model_name = args.model_name

    ## You can change 'default' experiment name:
    mlflow.set_experiment("iris")

    ## Auto-log sklearn metrics
    mlflow.sklearn.autolog(log_models=False)

    ## Load and split dataset
    iris = datasets.load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=seed
    )

    ## To log custom parameters:
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("seed", seed)

    ## Train the model
    clf = RandomForestClassifier(max_depth=max_depth, random_state=seed)
    clf.fit(x_train, y_train)

    ## These values used to define model's signature
    ## in order to manage the schema of the model server:
    input_example_df = pd.DataFrame(x_train, columns=iris.feature_names)
    input_example_df = input_example_df[:5]
    signature = infer_signature(input_example_df, clf.predict(x_train))

    ## You can either infer schema from the input/output,
    ## or define input_example=input_example_df, or both:
    mlflow.sklearn.log_model(
        clf,
        artifact_path="model",
        registered_model_name=model_name,
        signature=signature,
        input_example=input_example_df,
        pip_requirements=["-r {}".format(REQUIREMENTS_TXT)],
    )


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
