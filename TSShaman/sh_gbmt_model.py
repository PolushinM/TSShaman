from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import cross_val_score

from ._logger import *
from .base_model import ShBaseModel
from .feature_generation import TimedataFeaturesHost, LongShiftFeaturesHost, MovingAverageFeaturesHost, \
    ShortShiftFeaturesHost
from .feature_selection import *
from .optimization import *


class ShGBMTModel(ShBaseModel):

    def __init__(self, review_period, forecast_horizon=1, random_state=0, n_jobs=-1):
        super().__init__(review_period, forecast_horizon, random_state)

        self.non_linear_features = []
        self.categorical_features = []
        self.y: pd.Series() = None
        self.alpha: float = 0.001

        self.estimator = CatBoostRegressor(n_estimators=1000,
                                           learning_rate=0.07,
                                           rsm=0.1,
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

        self.model = CatBoostRegressor(n_estimators=1000,
                                       learning_rate=0.045,
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
            TimedataFeaturesHost(review_period=review_period, forecast_horizon=forecast_horizon),
            ShortShiftFeaturesHost(name='y', review_period=review_period, forecast_horizon=forecast_horizon),
            LongShiftFeaturesHost(name='y', review_period=review_period, forecast_horizon=round(forecast_horizon)),
            MovingAverageFeaturesHost(name='y',
                                      review_period=round(review_period * 2),
                                      forecast_horizon=round(forecast_horizon * 1.1))
        ]
        self.feature_hosts_weights = [2.5, 0.7, 0.3, 1.5]

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
        feature_select_estimator = self.estimator.copy()
        feature_select_estimator.set_params(feature_weights=self.feature_weights)
        features_to_drop = coefficient_feature_select(self.X[self.review_period:], y[self.review_period:],
                                                      estimator=feature_select_estimator,
                                                      quantile=0.5)
        del feature_select_estimator

        logger.debug(f'Number of coefficients dropped features={len(features_to_drop)}')
        self.X.drop(features_to_drop, axis=1, inplace=True)
        self.assign_feature_masks(self.X.columns.tolist())

        feature_select_estimator = self.estimator.copy()
        feature_select_estimator.set_params(learning_rate=0.1,
                                            rsm=0.5,
                                            use_best_model=True,
                                            early_stopping_rounds=100)

        features_to_drop = catboost_feature_select(self.X[self.review_period:], y[self.review_period:],
                                                   estimator=feature_select_estimator,
                                                   quantile=0.75)
        del feature_select_estimator

        logger.debug(f'Number of coefficients dropped features={len(features_to_drop)}')
        self.X.drop(features_to_drop, axis=1, inplace=True)
        self.assign_feature_masks(self.X.columns.tolist())
        logger.debug(f'Number of remaining features={self.X.shape[1]}')

        # Reset dataset, recreate and join synthetic features by masks
        self.X = self.generate_and_join_synthetic_features(X, y)
        logger.debug('Remaining CatBoost features: ' + str(self.X.columns.tolist()))

        tree_estimator = self.estimator.copy()
        tree_estimator.set_params(learning_rate=0.08,
                                  rsm=0.5,
                                  use_best_model=True,
                                  early_stopping_rounds=200,
                                  feature_weights=self.feature_weights)

        tree_count = get_best_cb_tree_count(self.X[self.review_period:],
                                            y[self.review_period:],
                                            estimator=tree_estimator)

        self.model.set_params(n_estimators=round(tree_count ** 0.7 * 6), feature_weights=self.feature_weights)

        best_learning_rate = self.model.grid_search({'learning_rate': [0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14,
                                                                       0.15, 0.16, 0.175, 0.19, 0.205, 0.22, 0.24]},
                                                    self.X[self.review_period:],
                                                    y=y[self.review_period:],
                                                    cv=3,
                                                    partition_random_seed=0,
                                                    search_by_train_test_split=True,
                                                    refit=False,
                                                    shuffle=True,
                                                    stratified=None,
                                                    train_size=0.7,
                                                    verbose=False)['params']['learning_rate']
        logger.debug(f'Best learning rate={best_learning_rate:.3f}')

        self.model.set_params(learning_rate=best_learning_rate ** 0.7 * 0.3)
        self.model.fit(self.X[self.review_period:], y[self.review_period:])

        catboost_model_score = self.model.score(self.X[self.review_period:], y[self.review_period:])

        logger.debug(f'CatBoost R^2={catboost_model_score:.5f}, Tree count={self.model.tree_count_}')

        return self

    @property
    def feature_weights(self) -> dict:
        weights = dict()
        for host, weight in zip(self.features_hosts, self.feature_hosts_weights):
            weights.update((host.gbmt_weights * weight).to_dict())
        return weights

    @property
    def score(self):
        # TODO Score calculation
        pass
