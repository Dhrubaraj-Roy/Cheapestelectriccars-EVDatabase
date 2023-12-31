import logging
import pandas as pd
from model.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from zenml import step
from .config import ModelNameConfig
@step()
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    config: ModelNameConfig,
) -> RegressorMixin:
    """
    Train a regression model based on the specified configuration.

    Args:
        X_train (pd.DataFrame): Training data features.
        X_test (pd.DataFrame): Testing data features.
        y_train (pd.Series): Training data target.
        y_test (pd.Series): Testing data target.
        config (ModelNameConfig): Model configuration.

    Returns:
        RegressorMixin: Trained regression model.
    """
    try:
        model = None
        if config.model_name == "LinearRegression":
            # mlflow.sklearn.autolog()
            model = LinearRegressionModel()
            train_model = model.train(X_train, y_train)
            return train_model
        else:
            raise ValueError("Model name not supported")
    except Exception as e:
        logging.error("error in train model ".format(e))
        raise e
