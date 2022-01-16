import pandas as pd
import numpy as np
import re
from typing import Union
from math import pi


class TimedataFeatures(object):

    def __init__(self):
        self.weights = None
        self.mask = None
        return

    def assign_mask(self, columns) -> None:
        timedata_set = {'index_hour_lin', 'index_hour_sin',
                        'index_hour_cos', 'index_day_lin',
                        'index_day_sin', 'index_day_cos',
                        'index_week_lin', 'index_week_sin',
                        'index_week_cos', 'index_is_weekend',
                        'index_year_lin', 'index_year_sin',
                        'index_year_cos'}
        self.mask = set()
        for col in columns:
            if col in timedata_set:
                self.mask.add(col)
        return

    def generate(self, data: pd.DataFrame, y: Union[pd.Series, None], calculate_weights=True):
        features = pd.DataFrame()

        if pd.api.types.is_datetime64_ns_dtype(data.index):
            hour_features = self.generate_hour_features(data.index.to_series(), name='index')
            day_features = self.generate_day_features(data.index.to_series(), name='index')
            week_features = self.generate_week_features(data.index.to_series(), name='index')
            weekend_feature = self.generate_weekend_feature(data.index.to_series(), name='index')
            year_features = self.generate_year_features(data.index.to_series(), name='index')
            if not hour_features.empty:
                features = features.join(hour_features, how='outer')
            if not day_features.empty:
                features = features.join(day_features, how='outer')
            if not week_features.empty:
                features = features.join(week_features, how='outer')
            if not weekend_feature.empty:
                features = features.join(weekend_feature, how='outer')
            if not year_features.empty:
                features = features.join(year_features, how='outer')

        for series in data.iteritems():
            if pd.api.types.is_datetime64_ns_dtype(series[1]):
                hour_features = self.generate_hour_features(series[1], name=str(series[0]) + '_')
                day_features = self.generate_day_features(series[1], name=str(series[0]) + '_')
                week_features = self.generate_week_features(series[1], name=str(series[0]) + '_')
                weekend_feature = self.generate_weekend_feature(series[1], name=str(series[0]) + '_')
                year_features = self.generate_year_features(series[1], name=str(series[0]) + '_')
                if not hour_features.empty:
                    features = features.join(hour_features, how='outer')
                if not day_features.empty:
                    features = features.join(day_features, how='outer')
                if not week_features.empty:
                    features = features.join(week_features, how='outer')
                if not weekend_feature.empty:
                    features = features.join(weekend_feature, how='outer')
                if not year_features.empty:
                    features = features.join(year_features, how='outer')
        if calculate_weights:
            self.weights = get_corr_weights(features, y).iloc[0]

        return features * self.weights

    def generate_hour_features(self, series: pd.Series, name=''):
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

    def generate_day_features(self, series: pd.Series, name=''):
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

    def generate_week_features(self, series: pd.Series, name=''):
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

    def generate_weekend_feature(self, series: pd.Series, name=''):
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
        return pd.DataFrame(weekend,
                            columns=columns,
                            index=series.values)

    def generate_year_features(self, series: pd.Series, name=''):
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


# Autocorrelation features
class ShiftFeatures(object):

    def __init__(self):
        mean = 0.0
        self.weights = None
        self.mask = None
        return

    def assign_mask(self, columns) -> None:
        self.mask = []
        for col in columns:
            if len(re.findall('_shift', col)) > 0:
                search = re.split('_shift', col)
                self.mask.append(int(search[-1]))
        return

    def generate(self, data: pd.Series, name: str = '', review_period: int = 90, forecast_period: int = 1):

        features = pd.DataFrame()
        weights = pd.DataFrame()
        self.mean = data.mean()
        std = data.std()

        # Filling feature periods lists
        short_list = []
        long_list = []
        if self.mask is None:
            short_list = list(range(1, forecast_period))
            long_list = list(range(forecast_period, review_period + 1))
        else:
            for i in self.mask:
                if i < forecast_period:
                    short_list.append(i)
                else:
                    long_list.append(i)

        # Generate short (<forecast_period) features
        for i in short_list:
            shifted_data = data.shift(i, fill_value=self.mean)
            weights[name + '_shift' + str(i)] = pd.Series(1 / std * i / forecast_period *
                                                          abs(np.corrcoef(data.values, shifted_data.values)[0, 1]))

            features[name + '_shift' + str(i)] = shifted_data - self.mean

        # Generate long (>forecast_period) features
        for i in long_list:
            shifted_data = data.shift(i, fill_value=self.mean)
            weights[name + '_shift' + str(i)] = pd.Series(
                1 / std * abs(np.corrcoef(data.values, shifted_data.values)[0, 1]))
            features[name + '_shift' + str(i)] = shifted_data - self.mean

        self.weights = weights.iloc[0]

        return features * self.weights

    def get_one_row(self, data: np.array, name: str = ''):
        features = pd.DataFrame()
        # Generate short features
        for i in self.mask:
            features[name + '_shift' + str(i)] = pd.Series(data[-i] - self.mean)
        return features * self.weights


class Indicators(object):

    def __init__(self):
        self.mean = 0.0
        self.weights = None
        self.ema_mask = None
        self.dma_mask = None
        self.tma_mask = None
        self.qma_mask = None
        self.is_masked = False
        return

    def assign_mask(self, columns) -> None:
        self.ema_mask = []
        self.dma_mask = []
        self.tma_mask = []
        self.qma_mask = []
        self.is_masked = True

        for col in columns:
            if len(re.findall('_ema_', col)) > 0:
                search = re.split('_ema_', col)
                self.ema_mask.append(round(float(search[-1]), 1))
            if len(re.findall('_dma_', col)) > 0:
                search = re.split('_dma_', col)
                self.dma_mask.append(round(float(search[-1]), 1))
            if len(re.findall('_tma_', col)) > 0:
                search = re.split('_tma_', col)
                self.tma_mask.append(round(float(search[-1]), 1))
            if len(re.findall('_qma_', col)) > 0:
                search = re.split('_qma_', col)
                self.qma_mask.append(round(float(search[-1]), 1))
        return

    def generate(self, data: pd.Series,
                 name: str = '',
                 review_period: int = 168,
                 forecast_period: int = 1):

        self.mean = data.mean()

        if self.is_masked:
            general_mask = set()
            if self.ema_mask is not None:
                general_mask.update(self.ema_mask)
            if self.dma_mask is not None:
                general_mask.update(self.dma_mask)
            if self.tma_mask is not None:
                general_mask.update(self.tma_mask)
            if self.qma_mask is not None:
                general_mask.update(self.qma_mask)
            ema_windows = list(general_mask)
            ema_windows.sort()
            ema_windows = np.array(ema_windows)
        else:
            # Calculate logarithmic step for EMA windows, calculate windows and alphas for EMA, DMA, TMA, QMA
            ema_step = (review_period / forecast_period * 4) ** (1 / 31)
            ema_windows = np.array([forecast_period / 2 * ema_step ** i for i in range(32)])
        alphas = 2 / (ema_windows + 1)

        # Fill buffer vectors
        ema_buffer = np.ones((len(alphas),), dtype=float) * (data.iloc[0] - self.mean)
        dma_buffer = ema_buffer
        tma_buffer = ema_buffer
        qma_buffer = ema_buffer

        # Memory allocation for resulting values
        result_ema = np.ndarray((data.shape[0], len(alphas)), dtype=float)
        result_dma = np.ndarray((data.shape[0], len(alphas)), dtype=float)
        result_tma = np.ndarray((data.shape[0], len(alphas)), dtype=float)
        result_qma = np.ndarray((data.shape[0], len(alphas)), dtype=float)
        result_ema[0] = ema_buffer
        result_dma[0] = ema_buffer
        result_tma[0] = ema_buffer
        result_qma[0] = ema_buffer

        # Columns names
        columns = [name + '_ema_' + f'{window:.1f}' for window in ema_windows] + \
                  [name + '_dma_' + f'{window:.1f}' for window in ema_windows] + \
                  [name + '_tma_' + f'{window:.1f}' for window in ema_windows] + \
                  [name + '_qma_' + f'{window:.1f}' for window in ema_windows]

        # Calculate of EMA, DMA, TMA, QMA
        for i in range(1, data.shape[0]):
            qma_buffer = qma_buffer * (1 - alphas) + alphas * tma_buffer
            tma_buffer = tma_buffer * (1 - alphas) + alphas * dma_buffer
            dma_buffer = dma_buffer * (1 - alphas) + alphas * ema_buffer
            ema_buffer = ema_buffer * (1 - alphas) + alphas * (data[i] - self.mean)
            result_ema[i] = ema_buffer
            result_dma[i] = dma_buffer
            result_tma[i] = tma_buffer
            result_qma[i] = qma_buffer

        # Convert result into pandas DaraFrame
        features = pd.DataFrame(np.hstack((result_ema, result_dma, result_tma, result_qma)), index=data.index,
                                columns=columns)

        # Return in case of non-masked generation
        if self.is_masked:
            # Clean features by masks in case of masked generation
            drop_columns = set()
            if self.ema_mask is not None:
                drop_columns.update([name + '_ema_' + f'{window:.1f}' for window in self.ema_mask])
            if self.dma_mask is not None:
                drop_columns.update([name + '_dma_' + f'{window:.1f}' for window in self.dma_mask])
            if self.tma_mask is not None:
                drop_columns.update([name + '_tma_' + f'{window:.1f}' for window in self.tma_mask])
            if self.qma_mask is not None:
                drop_columns.update([name + '_qma_' + f'{window:.1f}' for window in self.qma_mask])
            for col in features.columns:
                if col not in drop_columns:
                    features.drop(col, axis=1, inplace=True)

        # Get weights
        self.weights = get_corr_weights(features, data).iloc[0]

        # Store buffers
        self.ema_buffer = ema_buffer
        self.dma_buffer = dma_buffer
        self.tma_buffer = tma_buffer
        self.qma_buffer = qma_buffer

        return features * self.weights

    # Calculation of single rows of features
    def get_one_row(self, y: float, name: str = ''):

        # Calculate windows and alphas for EMA, DMA, TMA, QMA
        general_mask = set()
        if self.ema_mask is not None:
            general_mask.update(self.ema_mask)
        if self.dma_mask is not None:
            general_mask.update(self.dma_mask)
        if self.tma_mask is not None:
            general_mask.update(self.tma_mask)
        if self.qma_mask is not None:
            general_mask.update(self.qma_mask)
        ema_windows = list(general_mask)
        ema_windows.sort()
        ema_windows = np.array(ema_windows)
        alphas = 2 / (ema_windows + 1)

        # Columns names
        columns = [name + '_ema_' + f'{window:.1f}' for window in ema_windows] + \
                  [name + '_dma_' + f'{window:.1f}' for window in ema_windows] + \
                  [name + '_tma_' + f'{window:.1f}' for window in ema_windows] + \
                  [name + '_qma_' + f'{window:.1f}' for window in ema_windows]

        # Calculate EMA, DMA, TMA, QMA
        self.qma_buffer = self.qma_buffer * (1 - alphas) + alphas * self.tma_buffer
        self.tma_buffer = self.tma_buffer * (1 - alphas) + alphas * self.dma_buffer
        self.dma_buffer = self.dma_buffer * (1 - alphas) + alphas * self.ema_buffer
        self.ema_buffer = self.ema_buffer * (1 - alphas) + alphas * (y - self.mean)

        # Convert result into pandas DaraFrame
        features = pd.Series(
            np.hstack((self.ema_buffer, self.dma_buffer, self.tma_buffer, self.qma_buffer)),
            index=columns)

        # Clean features by masks in case of masked generation
        drop_columns = set()
        if self.ema_mask is not None:
            drop_columns.update([name + '_ema_' + f'{window:.1f}' for window in self.ema_mask])
        if self.dma_mask is not None:
            drop_columns.update([name + '_dma_' + f'{window:.1f}' for window in self.dma_mask])
        if self.tma_mask is not None:
            drop_columns.update([name + '_tma_' + f'{window:.1f}' for window in self.tma_mask])
        if self.qma_mask is not None:
            drop_columns.update([name + '_qma_' + f'{window:.1f}' for window in self.qma_mask])
        for col in features.index:
            if col not in drop_columns:
                features.drop(col, inplace=True)

        return features * self.weights


def get_corr_weights(data: pd.DataFrame, y: pd.Series):
    weights = pd.DataFrame()
    for nm, col in data.iteritems():
        weights[nm] = pd.Series(abs(np.corrcoef(y.values, col.values)[0, 1]) / col.std())
    return weights
