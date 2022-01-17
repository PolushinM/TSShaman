from sh_linear_model import *
from sh_catboost_model import *
from cross_validation import *


class TSShaman(object):

    def __init__(self, review_period, forecast_period=1, elaboration_level=1, random_state=0):
        self.review_period = review_period
        self.forecast_period = forecast_period
        self.random_state = random_state
        self.linear_model = ShLinearModel(review_period, forecast_period, random_state)

    def fit(self, X: pd.DataFrame, y: pd.Series, additive_features: list = None, cv=16, verbose=False):
        self.linear_model.fit(X, y, additive_features, cv, verbose)
        return self

    def predict(self, X_additive_features=pd.DataFrame(), forecast_period=1, verbose=False):
        return self.linear_model.predict(X_additive_features, forecast_period, verbose)

    @property
    def score(self):
        return self.linear_model.score
