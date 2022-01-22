import pandas as pd
import numpy as np
import re
from typing import Union
from math import pi


class ShiftFeaturesBaseHost(object):

    def __init__(self,
                 review_period: int,
                 forecast_horizon: int,
                 name: str = ''):

        self.review_period = review_period
        self.forecast_horizon = forecast_horizon
        self.name = name
        self.mean = 0.0
        self.std = 1.0
        self.weights = None
        self.mask = None
        self.corrcoef_power = 0.5
        return

    def assign_mask(self, columns) -> None:
        self.mask = []
        for col in columns:
            if len(re.findall('_shift', col)) > 0:
                search = re.split('_shift', col)
                lag = int(search[-1])
                self.conditional_append_mask(lag)
        return

    def generate(self, data: pd.Series) -> pd.DataFrame:

        self.mean = data.mean()
        self.std = data.std()

        lags = self.lag_list
        if len(lags) == 0:
            return pd.DataFrame()
        features, self.weights = self.calculate_features_and_weights(data, lags)

        return features * self.weights

    def calculate_features_and_weights(self, data, lags):
        features = pd.DataFrame()
        weights = pd.DataFrame()
        for lag in lags:
            shifted_data = data.shift(lag, fill_value=self.mean)
            weights[self.name + '_shift' + str(lag)] = pd.Series(self.calculate_weight(data, shifted_data, lag))
            features[self.name + '_shift' + str(lag)] = shifted_data - self.mean
        return features, weights.iloc[0]

    def get_one_row(self, data: np.array) -> np.array:
        # This is a key function that must be performed very fast
        i: int = 0
        features_row = np.ndarray((len(self.mask),), dtype=float)
        for lag in self.mask:
            features_row[i] = data[-lag] - self.mean
            i += 1
        return features_row * self.weights.values

    @property
    def lag_list(self):
        # Filling feature periods (lags) lists
        if self.mask is None:
            lags = self.generate_initial_lags()
        else:
            lags = self.generate_masked_lags()
        return lags

    def conditional_append_mask(self, lag):
        """Pure virtual function, appends mask by lag if it satisfies the condition"""
        raise NotImplementedError()

    def calculate_weight(self, data, shifted_data, lag):
        """Pure virtual function"""
        raise NotImplementedError()

    def generate_initial_lags(self) -> list:
        """Pure virtual function"""
        raise NotImplementedError()

    def generate_masked_lags(self) -> list:
        """Pure virtual function"""
        raise NotImplementedError()


class LongShiftFeaturesHost(ShiftFeaturesBaseHost):

    def conditional_append_mask(self, lag):
        if lag >= self.forecast_horizon:
            self.mask.append(lag)

    def calculate_weight(self, data, shifted_data, lag):
        return abs(np.corrcoef(data.values, shifted_data.values)[0, 1]) ** self.corrcoef_power / self.std

    def generate_initial_lags(self) -> list:
        return list(range(self.forecast_horizon, self.review_period + 1))

    def generate_masked_lags(self) -> list:
        lags = []
        for i in self.mask:
            if i >= self.forecast_horizon:
                lags.append(i)
        return lags


class ShortShiftFeaturesHost(ShiftFeaturesBaseHost):

    lag_weight_power = 0.5

    def conditional_append_mask(self, lag):
        if lag < self.forecast_horizon:
            self.mask.append(lag)

    def calculate_weight(self, data, shifted_data, lag):
        return abs(np.corrcoef(data.values, shifted_data.values)[0, 1]) ** self.corrcoef_power \
               * (lag / self.forecast_horizon) ** self.lag_weight_power / self.std

    def generate_initial_lags(self) -> list:
        return list(range(1, self.forecast_horizon))

    def generate_masked_lags(self) -> list:
        lags = []
        for i in self.mask:
            if i < self.forecast_horizon:
                lags.append(i)
        return lags


class MovingAverageFeaturesHost(object):

    def __init__(self,
                 review_period: int,
                 forecast_horizon: int,
                 name: str = '',
                 ema_number=32,
                 short_ema_divisor=6,
                 ema_step_multiplier=8):

        self.review_period = review_period
        self.forecast_horizon = forecast_horizon
        self.name = name
        self.ema_number = ema_number
        self.short_ema_divisor = short_ema_divisor
        self.ema_step_multiplier = ema_step_multiplier
        self.ema_windows = self.__full_ema_windows
        self.alphas = 2 / (self.ema_windows + 1)
        self.is_mask_used = False
        self.masks = (list(self.ema_windows), list(self.ema_windows), list(self.ema_windows), list(self.ema_windows))
        self.mean = 0.0
        self.ema_buffer = None
        self.dma_buffer = None
        self.tma_buffer = None
        self.qma_buffer = None
        self.weights = None
        self.__bin_mask = None
        return

    def assign_masks(self, columns) -> None:
        self.is_mask_used = True
        self.masks = self.__parse_masks(columns)
        self.ema_windows = self.__masked_ema_windows
        self.alphas = 2 / (self.ema_windows + 1)
        self.__bin_mask = self.__get_bin_mask()
        return

    def generate(self, data: pd.Series):
        self.mean = data.mean()

        # Memory allocation for resulting values
        result_ema = np.ndarray((data.shape[0], len(self.alphas)), dtype=float)
        result_dma = np.ndarray((data.shape[0], len(self.alphas)), dtype=float)
        result_tma = np.ndarray((data.shape[0], len(self.alphas)), dtype=float)
        result_qma = np.ndarray((data.shape[0], len(self.alphas)), dtype=float)

        # Fill buffer vectors
        self.ema_buffer = np.ones((len(self.alphas),), dtype=float) * (data.iloc[0] - self.mean)
        self.dma_buffer = self.ema_buffer
        self.tma_buffer = self.ema_buffer
        self.qma_buffer = self.ema_buffer
        result_ema[0] = self.ema_buffer
        result_dma[0] = self.ema_buffer
        result_tma[0] = self.ema_buffer
        result_qma[0] = self.ema_buffer

        # Calculation of EMA, DMA, TMA, QMA
        for i in range(1, data.shape[0]):
            self.ema_buffer = self.ema_buffer * (1 - self.alphas) + self.alphas * (data[i] - self.mean)
            self.dma_buffer = self.dma_buffer * (1 - self.alphas) + self.alphas * self.ema_buffer
            self.tma_buffer = self.tma_buffer * (1 - self.alphas) + self.alphas * self.dma_buffer
            self.qma_buffer = self.qma_buffer * (1 - self.alphas) + self.alphas * self.tma_buffer
            result_ema[i] = self.ema_buffer
            result_dma[i] = self.dma_buffer
            result_tma[i] = self.tma_buffer
            result_qma[i] = self.qma_buffer

        # Convert result into pandas DaraFrame
        features = pd.DataFrame(np.hstack((result_ema, result_dma, result_tma, result_qma)),
                                index=data.index,
                                columns=self.non_filtered_columns)

        # Drop redundant columns
        features.drop(self.columns_to_drop, axis=1, inplace=True)

        # Get weights and return
        if features.empty:
            self.weights = pd.Series()
            return pd.DataFrame()
        else:
            self.weights = get_corr_weights(features, data)
            return features * self.weights

    def get_one_row(self, y: float) -> np.array:
        # Calculation of single rows of features
        # This is a key function that must be performed very fast
        self.ema_buffer = self.ema_buffer * (1 - self.alphas) + self.alphas * (y - self.mean)
        self.dma_buffer = self.dma_buffer * (1 - self.alphas) + self.alphas * self.ema_buffer
        self.tma_buffer = self.tma_buffer * (1 - self.alphas) + self.alphas * self.dma_buffer
        self.qma_buffer = self.qma_buffer * (1 - self.alphas) + self.alphas * self.tma_buffer

        return np.delete(np.hstack((self.ema_buffer, self.dma_buffer, self.tma_buffer, self.qma_buffer)),
                         self.__bin_mask) * self.weights.values

    def __parse_masks(self, columns):
        # Get list of features, remaining after feature selection
        ema_mask = list()
        dma_mask = list()
        tma_mask = list()
        qma_mask = list()
        for col in columns:
            if len(re.findall('_ema_', col)) > 0:
                search = re.split('_ema_', col)
                ema_mask.append(round(float(search[-1]), 1))
            if len(re.findall('_dma_', col)) > 0:
                search = re.split('_dma_', col)
                dma_mask.append(round(float(search[-1]), 1))
            if len(re.findall('_tma_', col)) > 0:
                search = re.split('_tma_', col)
                tma_mask.append(round(float(search[-1]), 1))
            if len(re.findall('_qma_', col)) > 0:
                search = re.split('_qma_', col)
                qma_mask.append(round(float(search[-1]), 1))
        return ema_mask, dma_mask, tma_mask, qma_mask

    def __get_bin_mask(self):
        mask = np.ndarray((len(self.non_filtered_columns),), dtype=bool)
        drop_columns_set = set(self.columns_to_drop)
        for i in range(len(self.non_filtered_columns)):
            if self.non_filtered_columns[i] in drop_columns_set:
                mask[i] = True
            else:
                mask[i] = False
        return mask

    @property
    def __full_ema_windows(self):

        ema_step = (self.review_period /
                    self.forecast_horizon *
                    self.ema_step_multiplier) ** (1 / (self.ema_number - 1))

        ema_windows = np.array([self.forecast_horizon / self.short_ema_divisor * ema_step ** i
                                for i in range(self.ema_number)])

        return ema_windows

    @property
    def __masked_ema_windows(self):
        general_mask = set()
        for item in self.masks:
            general_mask.update(item)
        ema_windows = list(general_mask)
        ema_windows.sort()
        return np.array(self.ema_windows)

    @property
    def non_filtered_columns(self):
        return [self.name + '_ema_' + f'{window:.1f}' for window in self.ema_windows] + \
               [self.name + '_dma_' + f'{window:.1f}' for window in self.ema_windows] + \
               [self.name + '_tma_' + f'{window:.1f}' for window in self.ema_windows] + \
               [self.name + '_qma_' + f'{window:.1f}' for window in self.ema_windows]

    @property
    def columns_to_drop(self):
        remaining_columns = set()
        columns_to_drop = list()
        remaining_columns.update([self.name + '_ema_' + f'{window:.1f}' for window in self.masks[0]])
        remaining_columns.update([self.name + '_dma_' + f'{window:.1f}' for window in self.masks[1]])
        remaining_columns.update([self.name + '_tma_' + f'{window:.1f}' for window in self.masks[2]])
        remaining_columns.update([self.name + '_qma_' + f'{window:.1f}' for window in self.masks[3]])
        for col in self.non_filtered_columns:
            if col not in remaining_columns:
                columns_to_drop.append(col)
        return columns_to_drop


class TimedataFeaturesHost(object):

    def __init__(self):
        self.weights = None
        self.mask = None
        return

    def assign_mask(self, columns) -> None:
        timedata_set = {'index_hour_lin', 'index_hour_sin', 'index_hour_cos',
                        'index_day_lin', 'index_day_sin', 'index_day_cos',
                        'index_week_lin', 'index_week_sin', 'index_week_cos', 'index_is_weekend',
                        'index_year_lin', 'index_year_sin', 'index_year_cos'}
        self.mask = set()
        for col in columns:
            if col in timedata_set:
                self.mask.add(col)
        return

    def generate(self, data: pd.DataFrame, y: Union[pd.Series, None] = None, calculate_weights=True) -> pd.DataFrame:

        if pd.api.types.is_datetime64_ns_dtype(data.index):
            features = pd.DataFrame(index=data.index)
            hour_features = self.generate_hour_features(data.index.to_series(), name='index')
            day_features = self.generate_day_features(data.index.to_series(), name='index')
            week_features = self.generate_week_features(data.index.to_series(), name='index')
            weekend_feature = self.generate_weekend_feature(data.index.to_series(), name='index')
            year_features = self.generate_year_features(data.index.to_series(), name='index')
            if not hour_features.empty:
                features = features.join(hour_features)
            if not day_features.empty:
                features = features.join(day_features)
            if not week_features.empty:
                features = features.join(week_features)
            if not weekend_feature.empty:
                features = features.join(weekend_feature)
            if not year_features.empty:
                features = features.join(year_features)
            if features.empty:
                return pd.DataFrame()
            if calculate_weights:
                self.weights = get_corr_weights(features, y)
            return features * self.weights
        else:
            return pd.DataFrame()

    def generate_hour_features(self, series: pd.Series, name='') -> pd.DataFrame:

        # Check data type
        if not pd.api.types.is_datetime64_ns_dtype(series):
            return pd.DataFrame()
        result = pd.DataFrame(index=series.values)

        # Check if the period of observation is too short for hour encoding {mask=None}
        if abs(pd.Timedelta(series[0] - series[-1]).total_seconds()) < 3600 and (self.mask is None):
            return pd.DataFrame()

        # Generate linear encoding vector
        hour_lin = (np.array(series.dt.minute / 60 + series.dt.second / 3600) - 0.5) / 0.3

        # Check if the variance of values too small for hour encoding {mask=None}
        if np.std(hour_lin) < 0.7 and (self.mask is None):
            return pd.DataFrame()

        # Generate SIN and COS encoding vectors
        hour_sin = np.sin(hour_lin * (2 * pi * 0.3)) / 0.7071
        hour_cos = np.cos(hour_lin * (2 * pi * 0.3)) / 0.7071

        # Check if the variance of values too small for hour encoding
        if (np.std(hour_sin) < 0.8 or np.std(hour_cos) < 0.8) and (self.mask is None):
            return pd.DataFrame()
        if self.mask is None or ((name + '_hour_lin') in self.mask):
            result = result.join(pd.DataFrame(hour_lin, index=series.values, columns=[name + '_hour_lin']))
        if self.mask is None or ((name + '_hour_sin') in self.mask):
            result = result.join(pd.DataFrame(hour_sin, index=series.values, columns=[name + '_hour_sin']))
        if self.mask is None or ((name + '_hour_cos') in self.mask):
            result = result.join(pd.DataFrame(hour_cos, index=series.values, columns=[name + '_hour_cos']))
        return result

    def generate_day_features(self, series: pd.Series, name='') -> pd.DataFrame:

        # Check data type
        if not pd.api.types.is_datetime64_ns_dtype(series):
            return pd.DataFrame()
        result = pd.DataFrame(index=series.values)

        # Check if the period of observation is too short for day encoding
        if abs(pd.Timedelta(series[0] - series[-1]).total_seconds()) < 86400 and (self.mask is None):
            return pd.DataFrame()

        # Generate linear encoding vector
        day_lin = (np.array(series.dt.hour / 24 + series.dt.minute / 1140) - 0.5) / 0.3

        # Check if the variance of values too small for day encoding
        if np.std(day_lin) < 0.7 and (self.mask is None):
            return pd.DataFrame()

        # Generate SIN and COS encoding vectors
        day_sin = np.sin(day_lin * (2 * pi * 0.3)) / 0.7071
        day_cos = np.cos(day_lin * (2 * pi * 0.3)) / 0.7071

        # Check if the variance of values too small for day encoding
        if (np.std(day_sin) < 0.8 or np.std(day_cos) < 0.8) and (self.mask is None):
            return pd.DataFrame()
        if self.mask is None or ((name + '_day_lin') in self.mask):
            result = result.join(pd.DataFrame(day_lin, index=series.values, columns=[name + '_day_lin']))
        if self.mask is None or ((name + '_day_sin') in self.mask):
            result = result.join(pd.DataFrame(day_sin, index=series.values, columns=[name + '_day_sin']))
        if self.mask is None or ((name + '_day_cos') in self.mask):
            result = result.join(pd.DataFrame(day_cos, index=series.values, columns=[name + '_day_cos']))
        return result

    def generate_week_features(self, series: pd.Series, name='') -> pd.DataFrame:

        # Check data type
        if not pd.api.types.is_datetime64_ns_dtype(series):
            return pd.DataFrame()

        # Check if the period of observation is too short for week encoding
        if abs(pd.Timedelta(series[0] - series[-1]).total_seconds()) < 604800 and (self.mask is None):
            return pd.DataFrame()

        # Generate linear encoding vector
        week_lin = np.array(series.dt.dayofweek / 7 + series.dt.hour / 168 - 0.5) / 0.3

        # Check if the variance of values too small for day encoding
        if np.std(week_lin) < 0.7 and (self.mask is None):
            return pd.DataFrame()

        # Generate SIN, COS and weekend encoding vectors
        week_sin = np.sin(week_lin * (2 * pi * 0.3)) / 0.7071
        week_cos = np.cos(week_lin * (2 * pi * 0.3)) / 0.7071

        # Check if the variance of values too small for week encoding
        if (np.std(week_sin) < 0.8 or np.std(week_cos) < 0.8) and (self.mask is None):
            return pd.DataFrame()
        if (self.mask is not None) and ((name + '_week_lin') not in self.mask):
            return pd.DataFrame()

        columns = [name + '_week_lin',
                   name + '_week_sin',
                   name + '_week_cos']

        return pd.DataFrame(np.vstack((week_lin, week_sin, week_cos)).T,
                            columns=columns,
                            index=series.values)

    def generate_weekend_feature(self, series: pd.Series, name='') -> pd.DataFrame:

        # Check data type
        if not pd.api.types.is_datetime64_ns_dtype(series):
            return pd.DataFrame()

        # Check if the period of observation is too short for week encoding
        if abs(pd.Timedelta(series[0] - series[-1]).total_seconds()) < 604800 and (self.mask is None):
            return pd.DataFrame()

        # Generate SIN, COS and weekend encoding vectors
        weekend = (np.array(series.dt.dayofweek) // 5 - 0.2857) / 0.488

        # Check if the variance of values too small for weekend encoding
        if np.std(weekend) < 0.7 and (self.mask is None):
            return pd.DataFrame()
        if (self.mask is not None) and ((name + '_is_weekend') not in self.mask):
            return pd.DataFrame()
        columns = [name + '_is_weekend']

        return pd.DataFrame(weekend, columns=columns, index=series.values)

    def generate_year_features(self, series: pd.Series, name='') -> pd.DataFrame:

        # Check data type
        if not pd.api.types.is_datetime64_ns_dtype(series):
            return pd.DataFrame()

        # Check if the period of observation is too short for year encoding
        if abs(pd.Timedelta(series[0] - series[-1]).total_seconds()) < 31536000 and (self.mask is None):
            return pd.DataFrame()

        # Generate linear encoding vector
        year_lin = (np.array(series.dt.dayofyear / 366) - 0.5) / 0.3

        # Check if the variance of values too small for day encoding
        if np.std(year_lin) < 0.7 and (self.mask is None):
            return pd.DataFrame()

        # Generate SIN, COS and weekend encoding vectors
        year_sin = np.sin(year_lin * (2 * pi * 0.3)) / 0.7071
        year_cos = np.cos(year_lin * (2 * pi * 0.3)) / 0.7071

        # Check if the variance of values too small for year encoding
        if (np.std(year_sin) < 0.8 or np.std(year_cos) < 0.8) and (self.mask is None):
            return pd.DataFrame()
        if (self.mask is not None) and ((name + '_year_lin') not in self.mask):
            return pd.DataFrame()

        columns = [name + '_year_lin', name + '_year_sin', name + '_year_cos']

        return pd.DataFrame(np.vstack((year_lin, year_sin, year_cos)).T,
                            columns=columns,
                            index=series.values)


def get_corr_weights(X: pd.DataFrame, y: pd.Series, corrcoef_power=0.5):
    weights = pd.DataFrame()
    for name, col in X.iteritems():
        weights[name] = pd.Series(abs(np.corrcoef(y.values, col.values)[0, 1]) ** corrcoef_power / col.std())
    return weights.iloc[0]
