from .sh_linear_model import *
# from .sh_catboost_model import * # TODO: Add CatBoost in imports
from .cross_validation import *
from ._logger import *
from datetime import datetime


class TSShaman(object):

    def __init__(self, review_period, forecast_horizon=1, random_state=0, verbosity='info'):
        self.review_period = review_period
        self.forecast_horizon = forecast_horizon
        self.random_state = random_state
        self.sh_linear_model = ShLinearModel(review_period, forecast_horizon, random_state)
        set_verbosity(verbosity)

    def fit(self, X: pd.DataFrame,
            y: pd.Series,
            linear_features: list = None,
            boosting_features: list = None,
            categorical_features: list = None,
            cv=16,
            verbose=False,
            omega=0.001,
            qualification_level=1):
        start_fit_time = datetime.now()

        linear_alpha_multiplier = 1.0 + 100 * omega
        linear_feature_selection_strength = 0.032 / 0.15 * omega

        self.sh_linear_model.fit(X, y,
                                 linear_features,
                                 cv,
                                 verbose,
                                 alpha_multiplier=linear_alpha_multiplier,
                                 feature_selection_strength=linear_feature_selection_strength)

        logger.info(f'Fit time={(datetime.now() - start_fit_time).total_seconds():.1f}')

        return self

    def predict(self, X_linear_features=pd.DataFrame(), forecast_segment=1, verbose=False):
        start_predict_time = datetime.now()
        y_pred = self.sh_linear_model.predict(X_linear_features, forecast_segment, verbose)
        logger.info(f'Predict time={(datetime.now() - start_predict_time).total_seconds():.1f}\n')
        return y_pred

    @property
    def score(self):
        return self.sh_linear_model.score
