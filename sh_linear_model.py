import pandas as pd
import numpy as np
from base_model import ShBaseModel
from feature_generation import TimedataFeaturesHost, LongShiftFeaturesHost, ShortShiftFeaturesHost, \
    MovingAverageFeaturesHost
from feature_selection import l1_feature_select
from optimization import get_best_l2_alpha
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDRegressor, Ridge


class ShLinearModel(ShBaseModel):
    class LongLinearModel(object):
        def __init__(self, review_period, forecast_period=1, random_state=0):
            self.review_period = review_period
            self.forecast_period = forecast_period
            self.random_state = random_state
            self.linear_features = []
            self.y = pd.Series()
            self.X = pd.DataFrame()
            self.alpha = 0.001
            self.X_pred = pd.DataFrame()
            self.linear_model = SGDRegressor()

            self.shift_features_host = LongShiftFeaturesHost(name='y',
                                                             review_period=review_period,
                                                             forecast_period=forecast_period)
            self.timedata_features_host = TimedataFeaturesHost()
            self.ma_features_host = MovingAverageFeaturesHost(name='y',
                                                              review_period=review_period,
                                                              forecast_period=forecast_period)
            return

        def fit(self, X: pd.DataFrame, y: pd.Series,
                additive_features: list = None,
                cv=8,
                verbose=False,
                elaboration_level=1,
                alpha_multiplier=2,
                feature_selection_strength=0.0005):

            if additive_features is not None:
                self.linear_features = additive_features
            self.y = y.copy()

            # Generate and join synthetic features
            self.X = pd.DataFrame(index=X.index) \
                .join(self.timedata_features_host.generate(X, y)) \
                .join(self.shift_features_host.generate(y)) \
                .join(self.ma_features_host.generate(y))

            short_features_to_drop, _ = l1_feature_select(self.X[self.review_period:], y[self.review_period:],
                                                          strength=feature_selection_strength,
                                                          cv=cv,
                                                          random_state=self.random_state)

            self.X.drop(short_features_to_drop, axis=1, inplace=True)

            self.timedata_features_host.assign_mask(self.X.columns)
            self.shift_features_host.assign_mask(self.X.columns)
            self.ma_features_host.assign_masks(self.X.columns)

            # Reset dataset, recreate and join synthetic features by masks with additional features
            self.X = pd.DataFrame(index=X.index) \
                .join(self.timedata_features_host.generate(X, y)) \
                .join(X[self.linear_features]) \
                .join(self.shift_features_host.generate(y)) \
                .join(self.ma_features_host.generate(y))

            self.alpha, _ = get_best_l2_alpha(self.X[self.review_period:],
                                              y[self.review_period:],
                                              cv=cv)
            self.alpha = self.alpha * alpha_multiplier ** (3 / self.X.shape[1])
            self.linear_model = SGDRegressor(penalty='l2',
                                             eta0=0.002,
                                             power_t=0.2,
                                             max_iter=50000,
                                             alpha=self.alpha)

            self.linear_model.fit(self.X[self.review_period:].values, y[self.review_period:])

            return self

        def predict(self, X=pd.DataFrame(), forecast_period=1, verbose=False):
            y = self.y.values
            X_pred = self.generate_empty_predict_frame(forecast_period)
            X_pred = X_pred.join(self.timedata_features_host.generate(X_pred, y=None, calculate_weights=False)) \
                .join(X[self.linear_features])

            for i in range(0, forecast_period):
                y = np.append(y, self.predict_one_step(X_pred.iloc[i].values, y))

            return y[-forecast_period:]

        def generate_empty_predict_frame(self, forecast_period):
            step_time = self.step_time
            time = self.X.index[-1]
            index = []
            for i in range(forecast_period):
                time += step_time
                index.append(time)
            return pd.DataFrame(index=index)

        def predict_one_step(self, X_pred_row: np.array, y: np.array):
            shift_row = self.shift_features_host.get_one_row(data=y)
            indicators_row = self.ma_features_host.get_one_row(y=y[-1])
            return self.linear_model.predict(np.hstack((X_pred_row, shift_row, indicators_row)).reshape(1, -1))

        @property
        def step_time(self):
            return self.X.index[1] - self.X.index[0]

    class ShortLinearModel(object):
        def __init__(self, review_period, forecast_period=1, random_state=0):
            self.review_period = review_period
            self.forecast_period = forecast_period
            self.random_state = random_state
            self.y = pd.Series()
            self.X = pd.DataFrame()
            self.alpha = 0.001
            self.X_pred = pd.DataFrame()
            self.linear_model = SGDRegressor()

            self.shift_features_host = ShortShiftFeaturesHost(name='y',
                                                              review_period=review_period,
                                                              forecast_period=forecast_period)
            return

        def fit(self, X: pd.DataFrame, y: pd.Series,
                additive_features: list = None,
                cv=8,
                verbose=False,
                elaboration_level=1,
                alpha_multiplier=2,
                feature_selection_strength=0.0005):

            if additive_features is not None:
                self.linear_features = additive_features
            self.y = y.copy()

            # Generate and join synthetic features
            self.X = pd.DataFrame(index=X.index) \
                .join(self.shift_features_host.generate(y))

            short_features_to_drop, _ = l1_feature_select(self.X[self.review_period:], y[self.review_period:],
                                                          strength=feature_selection_strength,
                                                          cv=cv,
                                                          random_state=self.random_state)

            self.X.drop(short_features_to_drop, axis=1, inplace=True)

            self.shift_features_host.assign_mask(self.X.columns)

            # Reset dataset, recreate and join synthetic features by masks with additional features
            self.X = pd.DataFrame(index=X.index) \
                .join(self.shift_features_host.generate(y))

            self.alpha, _ = get_best_l2_alpha(self.X[self.review_period:],
                                              y[self.review_period:],
                                              cv=cv)

            self.alpha = self.alpha * alpha_multiplier ** (16 / self.X.shape[1])
            self.linear_model = SGDRegressor(penalty='l2',
                                             eta0=0.002,
                                             power_t=0.2,
                                             max_iter=50000,
                                             alpha=self.alpha)

            self.linear_model.fit(self.X[self.review_period:].values, y[self.review_period:])

            return self

        def predict(self, X=pd.DataFrame(), forecast_period=1, verbose=False):
            y = self.y.values
            X_pred = self.generate_empty_predict_frame(forecast_period)

            for i in range(0, forecast_period):
                y = np.append(y, self.predict_one_step(X_pred.iloc[i].values, y))

            return y[-forecast_period:]

        def generate_empty_predict_frame(self, forecast_period):
            step_time = self.step_time
            time = self.X.index[-1]
            index = []
            for i in range(forecast_period):
                time += step_time
                index.append(time)
            return pd.DataFrame(index=index)

        def predict_one_step(self, X_pred_row: np.array, y: np.array):
            shift_row = self.shift_features_host.get_one_row(data=y)
            return self.linear_model.predict(shift_row.reshape(1, -1))

        @property
        def step_time(self):
            return self.X.index[1] - self.X.index[0]

    def __init__(self, review_period, forecast_period=1, random_state=0):
        super().__init__(review_period, forecast_period, random_state)

        self.linear_features = []
        self.y = pd.Series()
        self.X = pd.DataFrame()
        self.alpha = 0.001
        self.X_pred = pd.DataFrame()
        self.linear_long_model = SGDRegressor()

        self.shift_long_features_obj = LongShiftFeaturesHost(name='y',
                                                             review_period=review_period,
                                                             forecast_period=forecast_period)
        self.shift_short_features_obj = ShortShiftFeaturesHost(name='y',
                                                               review_period=review_period,
                                                               forecast_period=forecast_period)

        self.timedata_features_obj = TimedataFeaturesHost()

        self.ma_features_obj = MovingAverageFeaturesHost(name='y',
                                                         review_period=review_period,
                                                         forecast_period=forecast_period)

        self.long_linear_model = self.LongLinearModel(review_period, forecast_period, random_state)
        self.short_linear_model = self.ShortLinearModel(review_period, forecast_period, random_state)

        return

    def fit(self, X: pd.DataFrame, y: pd.Series,
            additive_features: list = None,
            cv=8,
            verbose=False,
            elaboration_level=1,
            alpha_multiplier=2,
            feature_selection_strength=0.0005):

        self.y = y.copy()
        self.X = X.copy()

        self.long_linear_model.fit(X, y,
                                   additive_features=additive_features,
                                   cv=cv,
                                   verbose=verbose,
                                   elaboration_level=elaboration_level,
                                   alpha_multiplier=alpha_multiplier,
                                   feature_selection_strength=feature_selection_strength)

        self.short_linear_model.fit(X, y,
                                    additive_features=additive_features,
                                    cv=cv,
                                    verbose=verbose,
                                    elaboration_level=elaboration_level,
                                    alpha_multiplier=alpha_multiplier,
                                    feature_selection_strength=feature_selection_strength)

        print('Short R^2=', cross_val_score(self.short_linear_model.linear_model,
                                            self.long_linear_model.X[self.review_period:],
                                            y[self.review_period:],
                                            cv=2).mean())

        stack = np.vstack((self.long_linear_model.linear_model.predict(self.long_linear_model.X[self.review_period:]),
                           self.long_linear_model.linear_model.predict(
                               self.long_linear_model.X[self.review_period:]))).T

        self.stack_model = Ridge(fit_intercept=False, alpha=30).fit(stack, y[self.review_period:])

        print(self.stack_model.coef_)

        return self

    def predict(self, X=pd.DataFrame(), forecast_period=1, verbose=False):
        long_coef = 0.1
        X_pred = self.generate_empty_predict_frame(forecast_period)
        y_long = self.long_linear_model.predict(X, forecast_period=forecast_period, verbose=verbose)
        y_short = self.short_linear_model.predict(X, forecast_period=forecast_period, verbose=verbose)

        y = (self.stack_model.coef_[0] + long_coef) * y_long + (self.stack_model.coef_[1] - long_coef) * y_short

        return pd.DataFrame(y[-forecast_period:], index=X_pred.index)

    '''def fit_old(self, X: pd.DataFrame, y: pd.Series,
            additive_features: list = None,
            cv=8,
            verbose=False,
            elaboration_level=1,
            alpha_multiplier=2,
            feature_selection_strength=0.0005):

        if additive_features is not None:
            self.linear_features = additive_features
        self.y = y.copy()

        # Generate and join synthetic features
        self.X = pd.DataFrame(index=X.index) \
            .join(self.timedata_features_obj.generate(X, y)) \
            .join(self.shift_long_features_obj.generate(y)) \
            .join(self.ma_features_obj.generate(y))

        short_features_to_drop, _ = l1_feature_select(self.X[self.review_period:],
                                                y[self.review_period:],
                                                strength=feature_selection_strength,
                                                cv=cv,
                                                random_state=self.random_state)

        self.X.drop(short_features_to_drop, axis=1, inplace=True)

        self.timedata_features_obj.assign_mask(self.X.columns)
        self.shift_long_features_obj.assign_mask(self.X.columns)
        self.ma_features_obj.assign_masks(self.X.columns)

        # Reset dataset, recreate and join synthetic features by masks with additional features
        self.X = pd.DataFrame(index=X.index) \
            .join(self.timedata_features_obj.generate(X, y)) \
            .join(X[self.linear_features]) \
            .join(self.shift_long_features_obj.generate(y)) \
            .join(self.ma_features_obj.generate(y))

        self.alpha, _ = get_best_l2_alpha(self.X[self.review_period:],
                                          y[self.review_period:],
                                          cv=cv)
        self.alpha = self.alpha * alpha_multiplier**(3/self.X.shape[1])
        self.linear_long_model = SGDRegressor(penalty='l2',
                                              eta0=0.002,
                                              power_t=0.2,
                                              max_iter=50000,
                                              alpha=self.alpha)

        self.linear_long_model.fit(self.X[self.review_period:].values, y[self.review_period:])




        short_features = self.shift_short_features_obj.generate(y)
        short_features_to_drop, _ = l1_feature_select(short_features[self.review_period:],
                                                      y[self.review_period:],
                                                      strength=feature_selection_strength,
                                                      cv=cv,
                                                      random_state=self.random_state)
        short_features.drop(short_features_to_drop, axis=1, inplace=True)
        self.shift_short_features_obj.assign_mask(short_features.columns)


        self.alpha_short, _ = get_best_l2_alpha(short_features[self.review_period:],
                                                y[self.review_period:],
                                                cv=cv)
        self.alpha_short = self.alpha_short * alpha_multiplier**(16/short_features.shape[1])
        short_features = self.shift_short_features_obj.generate(y)
        self.linear_short_model = SGDRegressor(penalty='l2',
                                               eta0=0.002,
                                               power_t=0.2,
                                               max_iter=50000,
                                               alpha=self.alpha_short)
        self.linear_short_model.fit(short_features[self.review_period:].values, y[self.review_period:])
        print('Short R^2=', cross_val_score(self.linear_short_model,
                                            short_features[self.review_period:],
                                            y[self.review_period:],
                                            cv=2).mean())

        stack = np.vstack((self.linear_long_model.predict(self.X[self.review_period:].values),
                          self.linear_short_model.predict(short_features[self.review_period:]))).T
        self.stack_model = Ridge(fit_intercept=False, alpha=1000).fit(stack, y[self.review_period:])
        print(self.stack_model.coef_)

        return self

    def predict_old(self, X=pd.DataFrame(), forecast_period=1, verbose=False):
        long_coef = 0.1
        y = self.y.values
        X_pred = self.generate_empty_predict_frame(forecast_period)
        X_pred = X_pred.join(self.timedata_features_obj.generate(X_pred, y=None, calculate_weights=False)) \
            .join(X[self.linear_features])

        for i in range(0, forecast_period):
            y = np.append(y, (self.stack_model.coef_[0]+long_coef)*self.predict_one_step_long(X_pred.iloc[i].values, y) +
                          (self.stack_model.coef_[1]-long_coef)*self.predict_one_step_short(y))

        return pd.DataFrame(y[-forecast_period:], index=X_pred.index)
        
        
    def predict_one_step_long(self, X_pred_row: np.array, y: np.array):
        shift_row = self.shift_long_features_obj.get_one_row(data=y)
        indicators_row = self.ma_features_obj.get_one_row(y=y[-1])
        return self.linear_long_model.predict(np.hstack((X_pred_row, shift_row, indicators_row)).reshape(1, -1))

    def predict_one_step_short(self, y: np.array):
        shift_row = self.shift_short_features_obj.get_one_row(data=y)
        return self.linear_short_model.predict(shift_row.reshape(1, -1))'''


    def generate_empty_predict_frame(self, forecast_period):
        step_time = self.step_time
        time = self.X.index[-1]
        index = []
        for i in range(forecast_period):
            time += step_time
            index.append(time)
        return pd.DataFrame(index=index)

    @property
    def score(self):
        return cross_val_score(self.long_linear_model.linear_model,
                               self.long_linear_model.X[self.review_period:],
                               self.long_linear_model.y[self.review_period:],
                               cv=3).mean()
