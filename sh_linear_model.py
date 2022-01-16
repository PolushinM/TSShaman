import pandas as pd
from base_model import *
from feature_generation import *
from feature_selection import *
from optimization import *
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDRegressor


class ShLinearModel(ShBaseModel):

    def fit(self, X: pd.DataFrame, y: pd.Series, additive_features: list = None, cv=16, verbose=False):
        if additive_features is not None:
            self.linear_features = additive_features
        self.X = pd.DataFrame(index=X.index)
        self.y = y.copy()
        self.mean = y.mean()

        # Make synthetic features
        self.shift_features_obj = ShiftFeatures()
        shift_features = self.shift_features_obj.generate(y, name='y',
                                                          review_period=self.review_period,
                                                          forecast_period=self.forecast_period)

        self.timedata_features_obj = TimedataFeatures()
        timedata_features = self.timedata_features_obj.generate(self.X, y)

        self.indicators_obj = Indicators()
        indicators = self.indicators_obj.generate(y, name='y',
                                                  review_period=self.review_period,
                                                  forecast_period=self.forecast_period)

        # Join features
        self.X = self.X.join(timedata_features)
        self.X = self.X.join(shift_features)
        self.X = self.X.join(indicators)

        # L1 feature select
        drop_features, _, _, _ = l1_feature_select(self.X[self.review_period:],
                                                   y[self.review_period:],
                                                   strength=0.0005,
                                                   cv=cv)

        self.X.drop(drop_features, axis=1, inplace=True)

        self.lr_alpha = 2 * get_best_l2_alpha(self.X[self.review_period:],
                                              y[self.review_period:],
                                              cv=cv)

        # Assign masks
        self.timedata_features_obj.assign_mask(self.X.columns)
        self.shift_features_obj.assign_mask(self.X.columns)
        self.indicators_obj.assign_mask(self.X.columns)

        # Reset dataset
        self.X = pd.DataFrame(index=X.index)

        # Recreate synthetic features by masks
        shift_features = self.shift_features_obj.generate(y, name='y',
                                                          review_period=self.review_period,
                                                          forecast_period=self.forecast_period)

        timedata_features = self.timedata_features_obj.generate(self.X, y)

        indicators = self.indicators_obj.generate(y, name='y',
                                                  review_period=self.review_period,
                                                  forecast_period=self.forecast_period)

        # Join recreated features
        self.X = self.X.join(timedata_features)
        self.X = self.X.join(X[self.linear_features])
        self.X = self.X.join(shift_features)
        self.X = self.X.join(indicators)

        # Fit linear model
        self.linear_model = SGDRegressor(penalty='l2',
                                         eta0=0.002,
                                         power_t=0.2,
                                         max_iter=50000,
                                         alpha=self.lr_alpha)

        self.linear_model.fit(self.X[self.review_period:], y[self.review_period:])

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
        timedata_features = self.timedata_features_obj.generate(self.X_pred, y=None,
                                                                calculate_weights=False)

        self.X_pred = self.X_pred.join(timedata_features)

        # Join linear features
        self.X_pred = self.X_pred.join(X_additive_features)

        # Add columns for shift features and indicators
        self.X_pred[self.shift_features_obj.get_one_row(data=y, name='y').columns] = None

        indicators_row = self.indicators_obj.get_one_row(y=y[-1], name='y')

        self.X_pred[indicators_row.index.tolist()] = None

        # Step by step predict
        for i in range(0, forecast_period):
            # Add one row of shift features
            shift_row = self.shift_features_obj.get_one_row(data=y, name='y')

            self.X_pred.loc[self.X_pred.index[i], shift_row.columns.tolist()] = shift_row.iloc[0]

            # Add one row of indicators
            indicators_row = self.indicators_obj.get_one_row(y=y[-1], name='y')

            self.X_pred.loc[self.X_pred.index[i], indicators_row.index.tolist()] = indicators_row

            # Predict and add value to the result
            y = np.append(y, self.linear_model.predict(self.X_pred.iloc[[i]]))

        return pd.DataFrame(y[-forecast_period:], index=self.X_pred.index)

    @property
    def score(self):
        return cross_val_score(self.linear_model, self.X[self.review_period:], self.y[self.review_period:], cv=3).mean()
