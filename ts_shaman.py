from sh_linear_model import *
from sh_catboost_model import *
from cross_validation import *


class TSShaman(object):

    def __init__(self, review_period, forecast_horizon=1, random_state=0):
        self.review_period = review_period
        self.forecast_horizon = forecast_horizon
        self.random_state = random_state
        self.sh_linear_model = ShLinearModel(review_period, forecast_horizon, random_state)

    def fit(self, X: pd.DataFrame,
            y: pd.Series,
            linear_features: list = None,
            boosting_features: list = None,
            categorical_features: list = None,
            cv=16,
            verbose=False,
            omega=0.001,
            qualification_level=1):

        linear_alpha_multiplier = 1.0 + 2000*omega
        linear_feature_selection_strength = 0.2*omega

        self.sh_linear_model.fit(X, y,
                                 linear_features,
                                 cv,
                                 verbose,
                                 alpha_multiplier=linear_alpha_multiplier,
                                 feature_selection_strength=linear_feature_selection_strength)
        return self

    def predict(self, X_linear_features=pd.DataFrame(), forecast_period=1, verbose=False):
        return self.sh_linear_model.predict(X_linear_features, forecast_period, verbose)

    @property
    def score(self):
        return self.sh_linear_model.score
