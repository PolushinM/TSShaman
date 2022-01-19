import pandas as pd
import numpy as np
from base_model import ShBaseModel
from feature_generation import TimedataFeatures, ShiftFeatures, MovingAverageFeatures
from feature_selection import l1_feature_select
from optimization import get_best_l2_alpha
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDRegressor


class ShLinearModel(ShBaseModel):

    def __init__(self, review_period, forecast_period=1, random_state=0):
        super().__init__(review_period, forecast_period, random_state)

        self.linear_features = []
        self.mean = 0.0
        self.y = pd.Series()
        self.X = pd.DataFrame()
        self.lr_alpha = 0.001
        self.X_pred = pd.DataFrame()

        self.linear_model = SGDRegressor()

        self.shift_features_obj = ShiftFeatures(name='y',
                                                review_period=review_period,
                                                forecast_period=forecast_period)

        self.timedata_features_obj = TimedataFeatures()

        self.ma_features_obj = MovingAverageFeatures(name='y',
                                                     review_period=review_period,
                                                     forecast_period=forecast_period)
        return

    def fit(self, X: pd.DataFrame, y: pd.Series,
            additive_features=None,
            cv=8,
            verbose=False,
            alpha_mult=2,
            feature_selection_strength=0.0005):

        if additive_features is not None:
            self.linear_features = additive_features
        self.X = pd.DataFrame(index=X.index)
        self.y = y.copy()
        self.mean = y.mean()

        # Generate and join synthetic features
        self.X = self.X.join(self.timedata_features_obj.generate(self.X, y))
        self.X = self.X.join(self.shift_features_obj.generate(y))
        self.X = self.X.join(self.ma_features_obj.generate(y))

        # L1 feature select
        drop_features, _, _, _ = l1_feature_select(self.X[self.review_period:],
                                                   y[self.review_period:],
                                                   strength=feature_selection_strength,
                                                   cv=cv)

        self.X.drop(drop_features, axis=1, inplace=True)

        self.lr_alpha = alpha_mult * get_best_l2_alpha(self.X[self.review_period:],
                                                       y[self.review_period:],
                                                       cv=cv)
        self.linear_model = SGDRegressor(penalty='l2',
                                         eta0=0.002,
                                         power_t=0.2,
                                         max_iter=50000,
                                         alpha=self.lr_alpha)

        # Assign masks
        self.timedata_features_obj.assign_mask(self.X.columns)
        self.shift_features_obj.assign_mask(self.X.columns)
        self.ma_features_obj.assign_masks(self.X.columns)

        # Reset dataset
        self.X = pd.DataFrame(index=X.index)

        # Recreate synthetic features by masks
        shift_features = self.shift_features_obj.generate(y)
        timedata_features = self.timedata_features_obj.generate(self.X, y)
        indicators = self.ma_features_obj.generate(y)

        # Join recreated features
        self.X = self.X.join(timedata_features)
        self.X = self.X.join(X[self.linear_features])
        self.X = self.X.join(shift_features)
        self.X = self.X.join(indicators)

        # Fit linear model
        self.linear_model.fit(self.X[self.review_period:].values, y[self.review_period:])

        return self

    def predict(self, X_additive_features=pd.DataFrame(), forecast_period=1, verbose=False):

        step_time = self.step_time
        time = self.X.index[-1] + step_time
        index = []
        y = self.y.values

        # Generate X DataFrame
        for i in range(0, forecast_period):
            index.append(time)
            time += step_time
        self.X_pred = pd.DataFrame(index=index)

        # Join timedata features
        timedata_features = self.timedata_features_obj.generate(self.X_pred, y=None, calculate_weights=False)
        self.X_pred = self.X_pred.join(timedata_features)

        # Join linear features
        self.X_pred = self.X_pred.join(X_additive_features[self.linear_features])

        # Step by step predict
        for i in range(0, forecast_period):
            # Add one row of shift features
            shift_row = self.shift_features_obj.get_one_row(data=y)
            # Add one row of indicators
            indicators_row = self.ma_features_obj.get_one_row(y=y[-1])
            # Predict and add value to the result
            y = np.append(y, self.linear_model.predict(
                np.hstack((self.X_pred.iloc[i].values, shift_row, indicators_row)).reshape(1, -1)
            ))

        return pd.DataFrame(y[-forecast_period:], index=self.X_pred.index)

    @property
    def score(self):
        return cross_val_score(self.linear_model,
                               self.X[self.review_period:],
                               self.y[self.review_period:],
                               cv=3).mean()
