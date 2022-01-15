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
            self.__linear_features = additive_features
        self.__X = pd.DataFrame(index=X.index)
        self.__y = y.copy()
        self.__mean = y.mean()

        # Make synthetic features
        shift_features, shift_weights = generate_shift_features(y, name='y',
                                                                review_period=self.__review_period,
                                                                forecast_period=self.__forecast_period)

        timedata_features, timedata_weights = generate_timedata_features(self.__X, y)

        indicators, indicators_weights, _, _, _, _ = generate_indicators(y, name='y',
                                                                         review_period=self.__review_period,
                                                                         forecast_period=self.__forecast_period)

        # Join features
        self.__X = self.__X.join(timedata_features * timedata_weights)
        self.__X = self.__X.join(shift_features * shift_weights)
        self.__X = self.__X.join(indicators * indicators_weights)

        # L1 feature select
        drop_features, _, _, _ = l1_feature_select(self.__X[self.__review_period:],
                                                   y[self.__review_period:],
                                                   strength=0.0005,
                                                   cv=cv)

        self.__X.drop(drop_features, axis=1, inplace=True)

        self.lr_alpha = 2 * get_best_l2_alpha(self.__X[self.__review_period:],
                                              y[self.__review_period:],
                                              cv=cv)

        # Drop non-significant features
        '''self.__X.drop(coef_feature_select(self.__X[self.__review_period:], 
                                             y[self.__review_period:], 
                                             cv=cv), 
                      axis=1, 
                      inplace=True)'''

        # Get masks
        self.__shift_mask, \
        self.__ema_mask, \
        self.__dma_mask, \
        self.__tma_mask, \
        self.__qma_mask, \
        self.__timedata_mask = get_feature_masks(self.__X.columns)

        # Reset dataset
        self.__X = pd.DataFrame(index=X.index)

        # Recreate synthetic features by masks
        shift_features, \
        self.__shift_weights = generate_shift_features(y, name='y',
                                                       review_period=self.__review_period,
                                                       forecast_period=self.__forecast_period,
                                                       mask=self.__shift_mask)

        timedata_features, \
        self.__timedata_weights = generate_timedata_features(self.__X, y,
                                                             mask=self.__timedata_mask)

        indicators, \
        self.__indicators_weights, \
        self.__ema_buffer, \
        self.__dma_buffer, \
        self.__tma_buffer, \
        self.__qma_buffer = generate_indicators(y, name='y',
                                                review_period=self.__review_period,
                                                forecast_period=self.__forecast_period,
                                                ema_mask=self.__ema_mask,
                                                dma_mask=self.__dma_mask,
                                                tma_mask=self.__tma_mask,
                                                qma_mask=self.__qma_mask)

        # Join recreated features
        self.__X = self.__X.join(timedata_features * self.__timedata_weights)
        self.__X = self.__X.join(X[self.__linear_features])
        self.__X = self.__X.join(shift_features * self.__shift_weights)
        self.__X = self.__X.join(indicators * self.__indicators_weights)

        # Fit linear model
        self.linear_model = SGDRegressor(penalty='l2',
                                         eta0=0.002,
                                         power_t=0.2,
                                         max_iter=50000,
                                         alpha=self.lr_alpha)
        self.score = cross_val_score(self.linear_model,
                                     self.__X[self.__review_period:],
                                     y[self.__review_period:],
                                     cv=cv).mean()
        self.linear_model.fit(self.__X[self.__review_period:],
                              y[self.__review_period:])

        return self

    def predict(self, X_additive_features=pd.DataFrame(), forecast_period=1, verbose=False):

        step_time = self.step_time
        time = self.__X.index[-1] + step_time
        index = []
        y = self.__y.values
        ema_buffer = self.__ema_buffer
        dma_buffer = self.__dma_buffer
        tma_buffer = self.__tma_buffer
        qma_buffer = self.__qma_buffer

        # Generate X DataFrame
        for i in range(0, forecast_period):
            index.append(time)
            time += step_time
        self.__X_pred = pd.DataFrame(index=index)

        # Join timedata features
        timedata_features, _ = generate_timedata_features(self.__X_pred,
                                                          y=None,
                                                          mask=self.__timedata_mask,
                                                          calculate_weights=False)

        self.__X_pred = self.__X_pred.join(timedata_features * self.__timedata_weights)

        # Join linear features
        self.__X_pred = self.__X_pred.join(X_additive_features)

        # Add columns for shift features and indicators
        self.__X_pred[calc_shift_row(data=y, mean=self.__mean,
                                     name='y', mask=self.__shift_mask).columns] = None

        indicators_row, _, _, _, _ = calc_indicators_row(y=y[-1], mean=self.__mean,
                                                         ema_buffer=ema_buffer,
                                                         dma_buffer=dma_buffer,
                                                         tma_buffer=tma_buffer,
                                                         qma_buffer=qma_buffer,
                                                         ema_mask=self.__ema_mask,
                                                         dma_mask=self.__dma_mask,
                                                         tma_mask=self.__tma_mask,
                                                         qma_mask=self.__qma_mask,
                                                         name='y')

        self.__X_pred[indicators_row.index.tolist()] = None

        # Step by step predict
        for i in range(0, forecast_period):
            # Add one row of shift features
            shift_row = calc_shift_row(data=y,
                                       mean=self.__mean,
                                       name='y',
                                       mask=self.__shift_mask)

            self.__X_pred.loc[self.__X_pred.index[i], shift_row.columns.tolist()] = \
                shift_row.iloc[0] * self.__shift_weights

            # Add one row of indicators
            indicators_row, \
            ema_buffer, \
            dma_buffer, \
            tma_buffer, \
            qma_buffer = calc_indicators_row(y=y[-1], mean=self.__mean,
                                             ema_buffer=ema_buffer,
                                             dma_buffer=dma_buffer,
                                             tma_buffer=tma_buffer,
                                             qma_buffer=qma_buffer,
                                             ema_mask=self.__ema_mask,
                                             dma_mask=self.__dma_mask,
                                             tma_mask=self.__tma_mask,
                                             qma_mask=self.__qma_mask,
                                             name='y')

            self.__X_pred.loc[self.__X_pred.index[i], indicators_row.index.tolist()] = \
                indicators_row * self.__indicators_weights

            # Predict and add value to the result
            y = np.append(y, self.linear_model.predict(self.__X_pred.iloc[i].values.reshape(1, -1))[0])

        return pd.DataFrame(y, index=self.__X_pred.index)