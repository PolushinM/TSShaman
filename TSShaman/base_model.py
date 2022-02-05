from abc import ABC, abstractmethod
from typing import Union
import pandas as pd
import numpy as np


class ShBaseModel(ABC):

    def __init__(self, review_period, forecast_horizon=1, random_state=0):
        self.X: Union[pd.DataFrame, None] = None
        self.y: pd.Series() = None
        self.review_period = review_period
        self.forecast_horizon = forecast_horizon
        self.random_state = random_state
        self.features_hosts = []
        self.model = None
        return

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, additive_features: list = None, verbose=False, elaboration_level=1):
        """
        :param X: (pd.DataFrame, shape (n_samples, n_features)): the input data
        :param y: (pd.DataFrame, shape (n_samples, )): the target data
        :param additive_features: list of external user features
        :param elaboration_level: how accurate model optimisation should be
                (less value - fast fit, greater value - careful hyperparameters optimisation)
        :param verbose: Enable verbose output.
        Return:
            model (Class)
        """
        raise NotImplementedError("Pure virtual class.")

    def predict(self, X=pd.DataFrame(), forecast_segment=1, verbose=False):

        X_pred = self.generate_empty_predict_frame(forecast_segment)

        y = self.y.values
        self.initialise_rows(self.generate_empty_predict_frame(forecast_segment))
        for i in range(forecast_segment):
            y = np.append(y, self.predict_one_step(y))

        return pd.DataFrame(y[-forecast_segment:], index=X_pred.index)

    def predict_one_step(self, y: np.array):
        row = np.empty((0,), dtype=float)
        for host in self.features_hosts:
            row = np.hstack((row, host.get_one_row(y=y)))
        return self.model.predict(row.reshape(1, -1))

    def generate_empty_predict_frame(self, forecast_period):
        step_time = self.step_time
        time = self.X.index[-1]
        index = []
        for i in range(forecast_period):
            time += step_time
            index.append(time)
        return pd.DataFrame(index=index)

    def generate_and_join_synthetic_features(self, X: pd.DataFrame, y: pd.Series):
        result = pd.DataFrame(index=X.index)
        for host in self.features_hosts:
            result = result.join(host.generate(X, y))
        return result

    def assign_feature_masks(self, columns: list):
        for host in self.features_hosts:
            host.assign_mask(columns)

    def initialise_rows(self, X: pd.DataFrame):
        for host in self.features_hosts:
            host.initialise_rows(X)

    @property
    def step_time(self):
        return self.X.index[1] - self.X.index[0]
