from datetime import datetime

from .sh_linear_model import *
from .sh_gbmt_model import *  # TODO: Add CatBoost in imports
from .cross_validation import *
from ._logger import *


class TSShaman(object):

    def __init__(self, review_period, forecast_horizon=1, random_state=0, verbosity='info'):
        self.review_period = review_period
        self.forecast_horizon = forecast_horizon
        self.random_state = random_state
        self.sh_linear_model = ShLinearModel(review_period, forecast_horizon, random_state)
        self.sh_gbmt_model = ShGBMTModel(review_period, forecast_horizon, random_state)
        set_verbosity(verbosity)

    def fit(self, X: pd.DataFrame,
            y: pd.Series,
            linear_features: list = None,
            boosting_features: list = None,
            categorical_features: list = None,
            cv=16,
            verbose=False,
            omega=0.1,
            qualification_level=1):
        start_fit_time = datetime.now()

        linear_alpha_multiplier = 1.0 + 1800 * omega
        linear_feature_selection_strength = 0.13 * 0.4

        self.sh_linear_model.fit(X, y,
                                 linear_features,
                                 cv,
                                 verbose,
                                 alpha_multiplier=linear_alpha_multiplier,
                                 feature_selection_strength=linear_feature_selection_strength)

        self.sh_gbmt_model.fit(X, self.sh_linear_model.residuals)

        logger.info(f'Fit time={(datetime.now() - start_fit_time).total_seconds():.1f}')

        return self

    def predict(self, X=pd.DataFrame(), forecast_segment=1, verbose=False):
        start_predict_time = datetime.now()
        y_pred = self.sh_linear_model.predict(X, forecast_segment, verbose) \
                 + self.sh_gbmt_model.predict(X, forecast_segment, verbose)
        logger.info(f'Predict time={(datetime.now() - start_predict_time).total_seconds():.1f}\n')
        return y_pred

    @property
    def score(self):
        # TODO Score calculation
        return self.sh_linear_model.score
