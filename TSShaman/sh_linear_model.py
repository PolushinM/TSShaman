from ._logger import *
from .base_model import ShBaseModel
from .feature_generation import TimedataFeaturesHost, LongShiftFeaturesHost, ShortShiftFeaturesHost, \
    MovingAverageFeaturesHost
from .feature_selection import l1_feature_select, corcoeff_feature_selection
from .optimization import get_best_l2_alpha
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDRegressor, Ridge


class ShLinearModel(ShBaseModel):

    def __init__(self, review_period, forecast_horizon=1, random_state=0):
        super().__init__(review_period, forecast_horizon, random_state)

        self.linear_features = []
        self.y: pd.Series() = None
        self.alpha: float = 0.001

        self.long_linear_model = self.LongLinearModel(review_period, forecast_horizon, random_state)
        self.short_linear_model = self.ShortLinearModel(review_period, forecast_horizon, random_state)
        self.stack_model = Ridge(fit_intercept=False, alpha=1000)

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

        long_linear_model_score = cross_val_score(self.long_linear_model.linear_model,
                                                  self.long_linear_model.X[self.review_period:],
                                                  y[self.review_period:], cv=3).mean()

        short_linear_model_score = cross_val_score(self.short_linear_model.linear_model,
                                                   self.long_linear_model.X[self.review_period:],
                                                   y[self.review_period:], cv=3).mean()

        logger.debug(f'Long R^2={long_linear_model_score:.5f}, alpha={self.long_linear_model.alpha:.4f}')
        logger.debug(f'Short R^2={short_linear_model_score:.5f}, alpha={self.short_linear_model.alpha:.4f}')

        stack = np.vstack((self.long_linear_model.linear_model.predict(self.long_linear_model.X[self.review_period:]),
                           self.short_linear_model.linear_model.predict(
                               self.short_linear_model.X[self.review_period:]))).T

        self.stack_model.fit(stack, y[self.review_period:])

        logger.debug(f'Stack coefficients {self.stack_model.coef_}')

        return self

    def predict(self, X=pd.DataFrame(), forecast_segment=1, verbose=False):
        long_coef = 0.1
        X_pred = self.generate_empty_predict_frame(forecast_segment)
        y_long = self.long_linear_model.predict(X, forecast_segment=forecast_segment, verbose=verbose)
        y_short = self.short_linear_model.predict(X, forecast_segment=forecast_segment, verbose=verbose)

        y = (self.stack_model.coef_[0] + long_coef) * y_long + (self.stack_model.coef_[1] - long_coef) * y_short

        return pd.DataFrame(y[-forecast_segment:], index=X_pred.index)

    @property
    def score(self):
        pass

    class LinearBaseModel(ShBaseModel, ABC):
        def __init__(self, review_period, forecast_horizon=1, random_state=0):
            super().__init__(review_period, forecast_horizon, random_state)

            self.estimator = SGDRegressor(eta0=0.005, power_t=0.25, max_iter=50000, random_state=random_state)
            self.linear_model = SGDRegressor(penalty='l2',
                                             eta0=0.003, power_t=0.23, max_iter=100000,
                                             random_state=self.random_state)
            self.alpha = 0.0
            self.corrcoef_select_quantile = 0.0
            self.features_hosts = []

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

            self.X = self.generate_and_join_synthetic_features(X, y)

            logger.debug(f'Linear Model feature selection:')
            features_to_drop = corcoeff_feature_selection(self.X[self.review_period:],
                                                          y[self.review_period:],
                                                          strength=self.corrcoef_select_quantile)
            self.X.drop(features_to_drop, axis=1, inplace=True)
            logger.debug(f'Number of corrcoef dropped features={len(features_to_drop)}')

            features_to_drop, _ = l1_feature_select(self.X[self.review_period:], y[self.review_period:],
                                                    estimator=self.estimator,
                                                    strength=feature_selection_strength,
                                                    cv=cv,
                                                    random_state=self.random_state)
            self.X.drop(features_to_drop, axis=1, inplace=True)
            logger.debug(f'Number of remaining features={self.X.shape[1]}')
            logger.debug('Remaining features: ' + str(self.X.columns))

            self.assign_feature_masks(self.X.columns.tolist())

            # Reset dataset, recreate and join synthetic features by masks with additional features
            self.X = self.generate_and_join_synthetic_features(X, y)

            best_alpha, _ = get_best_l2_alpha(self.X[self.review_period:],
                                              y[self.review_period:],
                                              estimator=self.estimator,
                                              cv=cv)
            self.alpha = self.calculate_toughened_alpha(best_alpha, alpha_multiplier)
            logger.debug(f'Alpha multiplier={(self.alpha / best_alpha):.3f}')

            self.linear_model.set_params(alpha=self.alpha)
            self.linear_model.fit(self.X[self.review_period:].values, y[self.review_period:])

            return self

        def predict(self, X=pd.DataFrame(), forecast_segment=1, verbose=False):
            y = self.y.values
            self.initialise_rows(self.generate_empty_predict_frame(forecast_segment))
            for i in range(forecast_segment):
                y = np.append(y, self.predict_one_step(y))
            return y[-forecast_segment:]

        def predict_one_step(self, y: np.array):
            row = np.empty((0,), dtype=float)
            for host in self.features_hosts:
                row = np.hstack((row, host.get_one_row(y=y)))
            return self.linear_model.predict(row.reshape(1, -1))

        @abstractmethod
        def calculate_toughened_alpha(self, best_alpha: float, alpha_multiplier: float):
            pass

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


    class LongLinearModel(LinearBaseModel):
        def __init__(self, review_period, forecast_horizon=1, random_state=0):
            super().__init__(review_period, forecast_horizon, random_state)

            self.linear_features = []

            self.features_hosts = [
                TimedataFeaturesHost(),
                LongShiftFeaturesHost(name='y',
                                      review_period=review_period,
                                      forecast_horizon=forecast_horizon),
                MovingAverageFeaturesHost(name='y',
                                          review_period=review_period,
                                          forecast_horizon=forecast_horizon)
            ]
            self.corrcoef_select_quantile = 0.25
            return

        def calculate_toughened_alpha(self, best_alpha: float, alpha_multiplier: float):
            return best_alpha * alpha_multiplier


    class ShortLinearModel(LinearBaseModel):
        def __init__(self, review_period, forecast_horizon=1, random_state=0):
            super().__init__(review_period, forecast_horizon, random_state)

            self.features_hosts = [ShortShiftFeaturesHost(name='y',
                                                          review_period=review_period,
                                                          forecast_horizon=forecast_horizon), ]
            self.corrcoef_select_quantile = 0.5
            return

        def calculate_toughened_alpha(self, best_alpha: float, alpha_multiplier: float):
            return best_alpha * alpha_multiplier * 30
