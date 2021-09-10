# import base64
# from io import BytesIO
# import keras
# import numpy as np
# import os
# import PIL
# from PIL import Image
# import pip
# import yaml
# import tensorflow as tf

# import mlflow
# import mlflow.keras
# from mlflow.utils import PYTHON_VERSION
# from mlflow.utils.file_utils import TempDir
# from mlflow.utils.environment import _mlflow_conda_env

import pandas as pd
import mlflow.pyfunc


class RandomForestClassifierPyfunc(object):
    def __init__(self, model_uri):
        self._model = mlflow.pyfunc.load_model(model_uri=model_uri)

    def predict(self, X: pd.DataFrame):
        Y = self._model.predict(X)
        return Y


def _load_pyfunc(path):
    """
    Load the RandomForestClassifierPyfunc model.
    """
    with open(os.path.join(path, "conf.yaml"), "r") as f:
        conf = yaml.safe_load(f)
    keras_model_path = os.path.join(path, "keras_model")
    domain = conf["domain"].split("/")
    image_dims = np.array([int(x) for x in conf["image_dims"].split("/")], dtype=np.int32)
    # NOTE: TensorFlow based models depend on global state (Graph and Session) given by the context.
    # To make sure we score the model in the same session as we loaded it in, we create a new
    # session and a new graph here and store them with the model.
    with tf.Graph().as_default() as g:
        with tf.Session().as_default() as sess:
            keras.backend.set_session(sess)
            keras_model = mlflow.keras.load_model(keras_model_path)
    return KerasImageClassifierPyfunc(g, sess, keras_model, image_dims, domain=domain)


conda_env_template = """
name: flower_classifier
channels:
  - conda-forge
dependencies:
  - python=={python_version}
  - pip=={pip_version}
  - pip:
    - mlflow>=1.6
    - pillow=={pillow_version}
    - keras=={keras_version}
    - {tf_name}=={tf_version}
"""
