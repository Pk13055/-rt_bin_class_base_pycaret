import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from pycaret.classification import compare_models
from pycaret.classification import load_model as import_model
from pycaret.classification import pull
from pycaret.classification import save_model as dump_model
from pycaret.classification import setup

warnings.filterwarnings("ignore")


model_fname = "model.save"
MODEL_NAME = "bin_class_base_pycaret"


class Classifier:
    def __init__(self, preprocess, id_field, target_class, **kwargs) -> None:
        self.preprocess = preprocess
        self.target_class = target_class
        self.id_field = id_field

    def fit(self, train_data, _categorical, _numerical):
        setup(
            data=train_data,
            target=self.target_class,
            ignore_features=[self.id_field],
            # preprocess=False,
            # custom_pipeline=self.preprocess,
            categorical_features=_categorical,
            numeric_features=_numerical,
            silent=True,
            verbose=False,
        )
        model = compare_models()
        self.model = model
        metrics = pull()
        print(metrics)
        return model

    def predict(self, X, verbose=False):
        preds = self.model.predict(X)
        return preds

    def summary(self):
        self.model.get_params()

    def evaluate(self, x_test, y_test):
        """Evaluate the model and return the loss and metrics"""
        if self.model is not None:
            return self.model.score(x_test, y_test)


def save_model(model, model_path):
    # print(os.path.join(model_path, model_fname))
    dump_model(model, os.path.join(model_path, model_fname))
    # print("where the save_model function is saving the model to: " + os.path.join(model_path, model_fname))


def load_model(model_path):
    try:
        model = import_model(os.path.join(model_path, model_fname))
    except:
        raise Exception(
            f"""Error loading the trained {MODEL_NAME} model.
            Do you have the right trained model in path: {model_path}?"""
        )
    return model
