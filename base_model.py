import pandas as pd
import numpy as np


class ShBaseModel(object):

    def __init__(self, review_period, forecast_period=1, random_state=0):
        self.X = None
        self.review_period = review_period
        self.forecast_period = forecast_period
        self.random_state = random_state
        return

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

    def predict(self, X=None, forecast_period=1, verbose=False):
        """
        :param X: (pd.DataFrame, shape (n_samples, n_features)) the input data (if exist)
        :param forecast_period: = n_samples - number of samples (cycles), for which the forecast should be built
        :param verbose: Enable verbose output.
        Return:
            np.array, shape (n_samples, )
        """
        raise NotImplementedError("Pure virtual class.")

    @property
    def step_time(self):
        return self.X.index[1] - self.X.index[0]
