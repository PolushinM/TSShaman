from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDRegressor, Ridge

from ._logger import *
from .base_model import ShBaseModel
from .feature_generation import TimedataFeaturesHost, LongShiftFeaturesHost, ShortShiftFeaturesHost, \
    MovingAverageFeaturesHost
from .feature_selection import l1_feature_select, corcoeff_feature_selection
from .optimization import get_best_l2_alpha, get_best_l1_alpha, get_best_lr_alpha


class ShLinearModel(ShBaseModel):

    def __init__(self, review_period, forecast_horizon=1, random_state=0):
        super().__init__(review_period, forecast_horizon, random_state)

        self.linear_features = []
        self.y: pd.Series() = None
        self.alpha: float = 0.001

        self.long_linear_model = self.LongLinearModel(review_period, forecast_horizon, random_state)
        self.short_linear_model = self.ShortLinearModel(review_period, forecast_horizon, random_state)
        self.stack_model = Ridge(fit_intercept=False, alpha=1000)
        self.stacking_short_base_coef = 0.1

        return

    def fit(self, X: pd.DataFrame, y: pd.Series,
            additive_features: list = None,
            cv: int = 8,
            verbose: bool = False,
            qualification_level: int = 1,
            alpha_multiplier: float = 2,
            feature_selection_strength: float = 0.0005) -> object:
        self.y = y.copy()
        self.X = X.copy()

        self.long_linear_model.fit(X, y,
                                   additive_features=additive_features,
                                   cv=cv,
                                   verbose=verbose,
                                   qualification_level=qualification_level,
                                   alpha_multiplier=alpha_multiplier,
                                   feature_selection_strength=feature_selection_strength)

        self.short_linear_model.fit(X, y,
                                    additive_features=additive_features,
                                    cv=cv,
                                    verbose=verbose,
                                    qualification_level=qualification_level,
                                    alpha_multiplier=alpha_multiplier,
                                    feature_selection_strength=feature_selection_strength)

        long_linear_model_score = cross_val_score(self.long_linear_model.model,
                                                  self.long_linear_model.X[self.review_period:],
                                                  y[self.review_period:], cv=3).mean()

        short_linear_model_score = cross_val_score(self.short_linear_model.model,
                                                   self.long_linear_model.X[self.review_period:],
                                                   y[self.review_period:], cv=3).mean()

        logger.debug(f'Long R^2={long_linear_model_score:.5f}, alpha={self.long_linear_model.alpha:.4f}')
        logger.debug(f'Short R^2={short_linear_model_score:.5f}, alpha={self.short_linear_model.alpha:.4f}')

        stack = np.vstack((self.long_linear_model.model.predict(
            self.long_linear_model.X[self.review_period:]),
                           self.short_linear_model.model.predict(
                               self.short_linear_model.X[self.review_period:]))).T

        self.stack_model.fit(stack, y[self.review_period:])
        logger.debug(f'Stack coefficients {self.stack_model.coef_}')

        self.y_pred = (self.stack_model.coef_[0] / 2.5 - 0.2 + 1 - self.stacking_short_base_coef) * \
                 self.long_linear_model.model.predict(self.long_linear_model.X) + \
                 (self.stack_model.coef_[1] / 2.5 - 0.2 + self.stacking_short_base_coef) * \
                 self.short_linear_model.model.predict(self.short_linear_model.X)
        logger.debug(f'Stack score '
                     f'{r2_score(self.y_pred[self.review_period:], self.y[self.review_period:].values):.5f}')

        return self

    @property
    def residuals(self):
        return pd.Series(self.y.values - self.y_pred, index=self.X.index)

    def predict(self, X=pd.DataFrame(), forecast_segment=1, verbose=False):
        X_pred = self.generate_empty_predict_frame(forecast_segment)
        y_long = self.long_linear_model.predict(X, forecast_segment=forecast_segment, verbose=verbose)
        y_short = self.short_linear_model.predict(X, forecast_segment=forecast_segment, verbose=verbose)

        y = (self.stack_model.coef_[0] / 2.5 - 0.2 + 1 - self.stacking_short_base_coef) * y_long + \
            (self.stack_model.coef_[1] / 2.5 - 0.2 + self.stacking_short_base_coef) * y_short

        return pd.DataFrame(y[-forecast_segment:], index=X_pred.index)

    @property
    def score(self):
        # TODO Score calculation
        pass

    class LinearBaseModel(ShBaseModel):
        def __init__(self, review_period, forecast_horizon=1, random_state=0):
            super().__init__(review_period, forecast_horizon, random_state)

            self.l1_select_multiplier = 1.0
            self.estimator = SGDRegressor(eta0=0.005, power_t=0.25, max_iter=50000, random_state=random_state)
            self.model = SGDRegressor(penalty='l1',
                                      eta0=0.003, power_t=0.23, max_iter=100000,
                                      random_state=self.random_state)
            self.penalty = 'l1'
            self.alpha = 0.0
            self.corrcoef_select_quantile = 0.0
            self.features_hosts = []

            self.calculate_toughened_alpha = lambda alpha, alpha_multiplier: alpha * alpha_multiplier

            return

        def fit(self, X: pd.DataFrame, y: pd.Series,
                additive_features: list = None,
                cv: int = 8,
                verbose: bool = False,
                qualification_level: int = 1,
                alpha_multiplier: float = 2,
                feature_selection_strength: float = 0.0005):
            if additive_features is not None:
                self.linear_features = additive_features
            self.y = y.copy()

            logger.debug(f'Linear model feature generation:')
            self.X = self.generate_and_join_synthetic_features(X, y)

            logger.debug(f'Linear model feature selection:')
            features_to_drop = corcoeff_feature_selection(self.X[self.review_period:],
                                                          y[self.review_period:],
                                                          quantile=self.corrcoef_select_quantile)
            self.X.drop(features_to_drop, axis=1, inplace=True)
            logger.debug(f'Number of corrcoef dropped features={len(features_to_drop)}')

            features_to_drop, _ = l1_feature_select(self.X[self.review_period:], y[self.review_period:],
                                                    estimator=self.estimator,
                                                    strength=feature_selection_strength * self.l1_select_multiplier,
                                                    cv=cv,
                                                    random_state=self.random_state)

            self.X.drop(features_to_drop, axis=1, inplace=True)
            logger.debug(f'Number of remaining features={self.X.shape[1]}')
            logger.debug('Remaining features: ' + str(self.X.columns))

            self.assign_feature_masks(self.X.columns.tolist())

            # Reset dataset, recreate and join synthetic features by masks
            self.X = self.generate_and_join_synthetic_features(X, y)

            best_alpha, _ = get_best_lr_alpha(self.X[self.review_period:],
                                              y[self.review_period:],
                                              estimator=self.estimator,
                                              cv=cv,
                                              penalty=self.penalty,
                                              random_state=self.random_state,
                                              n_jobs=None)
            self.alpha = self.calculate_toughened_alpha(best_alpha, alpha_multiplier)
            logger.debug(f'Alpha multiplier={(self.alpha / best_alpha):.3f}')

            self.model.set_params(alpha=self.alpha, penalty=self.penalty)
            self.model.fit(self.X[self.review_period:].values, y[self.review_period:])

            return self

    class LongLinearModel(LinearBaseModel):
        def __init__(self, review_period, forecast_horizon=1, random_state=0):
            super().__init__(review_period, forecast_horizon, random_state)

            self.linear_features = []
            self.penalty = 'l2'

            self.features_hosts = [
                TimedataFeaturesHost(review_period=review_period,
                                     forecast_horizon=forecast_horizon),
                LongShiftFeaturesHost(name='y',
                                      review_period=review_period,
                                      forecast_horizon=forecast_horizon),
                MovingAverageFeaturesHost(name='y',
                                          review_period=review_period,
                                          forecast_horizon=forecast_horizon)
            ]
            self.corrcoef_select_quantile = 0.3
            self.l1_select_multiplier = 0.95

            self.calculate_toughened_alpha = lambda alpha, alpha_multiplier: alpha * alpha_multiplier / 6

            return

    class ShortLinearModel(LinearBaseModel):
        def __init__(self, review_period, forecast_horizon=1, random_state=0):
            super().__init__(review_period, forecast_horizon, random_state)

            self.penalty = 'l1'

            self.features_hosts = [ShortShiftFeaturesHost(name='y',
                                                          review_period=review_period,
                                                          forecast_horizon=forecast_horizon), ]
            self.corrcoef_select_quantile = 0.6
            self.l1_select_multiplier = 1.3

            self.calculate_toughened_alpha = lambda alpha, alpha_multiplier: alpha * alpha_multiplier * 3000

            return
