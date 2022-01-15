import pandas as pd
import numpy as np


class ShBaseModel(object):

    def __init__(self, review_period, forecast_period=1, random_state=0):
        self.__review_period = review_period
        self.__forecast_period = forecast_period
        self.__random_state = random_state
        self.__linear_features = []
        self.__mean = 0.0
        self.score = 0.0
        self.__y = pd.Series()
        self.__X = pd.DataFrame()
        self.lr_alpha = 0.001
        self.__shift_mask = []
        self.__ema_mask = []
        self.__dma_mask = []
        self.__tma_mask = []
        self.__qma_mask = []
        self.__timedata_mask = set()
        self.__shift_weights = pd.Series()
        self.__timedata_weights = pd.Series()
        self.__indicators_weights = pd.Series()
        self.__X_pred = pd.DataFrame()

        return

    def fit(self, X: pd.DataFrame, y: pd.Series, additive_features: list = None, cv=16, verbose=False):
        """
        :param X: (pd.DataFrame, shape (n_samples, n_features)): the input data
        :param y: (pd.DataFrame, shape (n_samples, )): the target data
        :param additive_features: list of external user features
        :param cv: number of cross-validation folds
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
        return self.__X.index.to_series()[1] - self.__X.index.to_series()[0]
