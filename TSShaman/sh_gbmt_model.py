from ._logger import *
from .base_model import ShBaseModel
from .feature_generation import TimedataFeaturesHost, LongShiftFeaturesHost, MovingAverageFeaturesHost, \
    ShortShiftFeaturesHost
from .feature_selection import *
from .optimization import *
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import cross_val_score


class ShGBMTModel(ShBaseModel):

    def __init__(self, review_period, forecast_horizon=1, random_state=0, n_jobs=-1):
        super().__init__(review_period, forecast_horizon, random_state)

        self.non_linear_features = []
        self.categorical_features = []
        self.y: pd.Series() = None
        self.alpha: float = 0.001

        self.catboost_estimator = CatBoostRegressor(n_estimators=1000,
                                                    learning_rate=0.1,
                                                    rsm=0.4,
                                                    depth=4,
                                                    loss_function='RMSE',
                                                    fold_permutation_block=None,
                                                    thread_count=n_jobs,
                                                    random_seed=random_state,
                                                    use_best_model=True,
                                                    logging_level='Silent',
                                                    subsample=None,
                                                    early_stopping_rounds=100,
                                                    cat_features=None,
                                                    score_function='L2',
                                                    feature_weights=None)

        self.model = CatBoostRegressor(n_estimators=1000,
                                       learning_rate=0.015,
                                       rsm=0.2,
                                       depth=3,
                                       loss_function='RMSE',
                                       fold_permutation_block=None,
                                       thread_count=n_jobs,
                                       random_seed=random_state,
                                       logging_level='Silent',
                                       subsample=None,
                                       cat_features=None,
                                       score_function='L2',
                                       feature_weights=None)

        self.coef_select_quantile = 0.0

        self.features_hosts = [
            TimedataFeaturesHost(),
            ShortShiftFeaturesHost(name='y',
                                   review_period=review_period,
                                   forecast_horizon=forecast_horizon),
            LongShiftFeaturesHost(name='y',
                                  review_period=review_period,
                                  forecast_horizon=round(forecast_horizon)),
            MovingAverageFeaturesHost(name='y',
                                      review_period=round(review_period*2),
                                      forecast_horizon=round(forecast_horizon))
        ]

        return

    def fit(self, X: pd.DataFrame, y: pd.Series,
            non_linear_features: list = None,
            categorical_features: list = None,
            verbose: bool = False,
            qualification_level: int = 1,
            feature_selection_strength: float = 0.0005) -> object:

        self.y = y.copy()

        self.X = self.generate_and_join_synthetic_features(X, y)

        logger.debug(f'CatBoost model feature selection:')

        features_to_drop = coefficient_feature_select(self.X[self.review_period:], y[self.review_period:],
                                                      estimator=self.catboost_estimator,
                                                      quantile=0.45)
        logger.debug(f'Number of coefficients dropped features={len(features_to_drop)}')
        self.X.drop(features_to_drop, axis=1, inplace=True)

        features_to_drop = catboost_feature_select(self.X[self.review_period:], y[self.review_period:],
                                                   estimator=self.catboost_estimator,
                                                   quantile=0.65)

        logger.debug(f'Number of coefficients dropped features={len(features_to_drop)}')
        self.X.drop(features_to_drop, axis=1, inplace=True)

        logger.debug(f'Number of remaining features={self.X.shape[1]}')
        logger.debug('Remaining CatBoost features: ' + str(self.X.columns.tolist()))

        self.assign_feature_masks(self.X.columns.tolist())

        # Reset dataset, recreate and join synthetic features by masks
        self.X = self.generate_and_join_synthetic_features(X, y)

        tree_count = get_best_cb_tree_count(self.X[self.review_period:],
                                            y[self.review_period:],
                                            estimator=self.catboost_estimator)

        self.model.set_params(n_estimators=tree_count)
        self.model.fit(self.X[self.review_period:], y[self.review_period:])

        catboost_model_score = self.model.score(self.X[self.review_period:], y[self.review_period:])

        logger.debug(f'CatBoost R^2={catboost_model_score:.5f}, Tree count={self.model.tree_count_}')

        return self

    @property
    def score(self):
        pass

