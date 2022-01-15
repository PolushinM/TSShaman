import pandas as pd
import numpy as np
import re
from typing import Union


def generate_hour_features(series: pd.Series, name='', mask=None):
    # Check data type
    if not pd.api.types.is_datetime64_ns_dtype(series):
        return pd.DataFrame()
    result = pd.DataFrame(index=series.values)

    # Check if the period of observation is too short for hour encoding {mask=None}
    if abs(pd.Timedelta(series[0] - series[-1]).total_seconds()) < 3600 and (mask is None):
        return pd.DataFrame()

    # Generate linear encoding vector
    hour_lin = (np.array(series.dt.minute / 60 + series.dt.second / 3600) - 0.5) / 0.3

    # Check if the variance of values too small for hour encoding {mask=None}
    if np.std(hour_lin) < 0.7 and (mask is None):
        return pd.DataFrame()

    # Generate SIN and COS encoding vectors
    hour_sin = np.sin(hour_lin * (2 * pi * 0.3)) / 0.7071
    hour_cos = np.cos(hour_lin * (2 * pi * 0.3)) / 0.7071

    # Check if the variance of values too small for hour encoding
    if (np.std(hour_sin) < 0.8 or np.std(hour_cos) < 0.8) and (mask is None):
        return pd.DataFrame()
    if mask is None or ((name + '_hour_lin') in mask):
        result = result.join(pd.DataFrame(hour_lin, index=series.values, columns=[name + '_hour_lin']))
    if mask is None or ((name + '_hour_sin') in mask):
        result = result.join(pd.DataFrame(hour_sin, index=series.values, columns=[name + '_hour_sin']))
    if mask is None or ((name + '_hour_cos') in mask):
        result = result.join(pd.DataFrame(hour_cos, index=series.values, columns=[name + '_hour_cos']))
    return result


def generate_day_features(series: pd.Series, name='', mask=None):
    # Check data type
    if not pd.api.types.is_datetime64_ns_dtype(series):
        return pd.DataFrame()
    result = pd.DataFrame(index=series.values)
    # Check if the period of observation is too short for day encoding
    if abs(pd.Timedelta(series[0] - series[-1]).total_seconds()) < 86400 and (mask is None):
        return pd.DataFrame()
    # Generate linear encoding vector
    day_lin = (np.array(series.dt.hour / 24 + series.dt.minute / 1140) - 0.5) / 0.3
    # Check if the variance of values too small for day encoding
    if np.std(day_lin) < 0.7 and (mask is None):
        return pd.DataFrame()
    # Generate SIN and COS encoding vectors
    day_sin = np.sin(day_lin * (2 * pi * 0.3)) / 0.7071
    day_cos = np.cos(day_lin * (2 * pi * 0.3)) / 0.7071
    # Check if the variance of values too small for day encoding
    if (np.std(day_sin) < 0.8 or np.std(day_cos) < 0.8) and (mask is None):
        return pd.DataFrame()
    if mask is None or ((name + '_day_lin') in mask):
        result = result.join(pd.DataFrame(day_lin, index=series.values, columns=[name + '_day_lin']))
    if mask is None or ((name + '_day_sin') in mask):
        result = result.join(pd.DataFrame(day_sin, index=series.values, columns=[name + '_day_sin']))
    if mask is None or ((name + '_day_cos') in mask):
        result = result.join(pd.DataFrame(day_cos, index=series.values, columns=[name + '_day_cos']))
    return result


def generate_week_features(series: pd.Series, name='', mask=None):
    # Check data type
    if not pd.api.types.is_datetime64_ns_dtype(series):
        return pd.DataFrame()
    # Check if the period of observation is too short for week encoding
    if abs(pd.Timedelta(series[0] - series[-1]).total_seconds()) < 604800 and (mask is None):
        return pd.DataFrame()
    # Generate linear encoding vector
    week_lin = np.array(series.dt.dayofweek / 7 + series.dt.hour / 168 - 0.5) / 0.3
    # Check if the variance of values too small for day encoding
    if np.std(week_lin) < 0.7 and (mask is None):
        return pd.DataFrame()
    # Generate SIN, COS and weekend encoding vectors
    week_sin = np.sin(week_lin * (2 * pi * 0.3)) / 0.7071
    week_cos = np.cos(week_lin * (2 * pi * 0.3)) / 0.7071
    # Check if the variance of values too small for week encoding
    if (np.std(week_sin) < 0.8 or np.std(week_cos) < 0.8) and (mask is None):
        return pd.DataFrame()
    if (mask is not None) and ((name + '_week_lin') not in mask):
        return pd.DataFrame()
    columns = [name + '_week_lin',
               name + '_week_sin',
               name + '_week_cos']
    return pd.DataFrame(np.vstack((week_lin, week_sin, week_cos)).T,
                        columns=columns,
                        index=series.values)


def generate_weekend_feature(series: pd.Series, name='', mask=None):
    # Check data type
    if not pd.api.types.is_datetime64_ns_dtype(series):
        return pd.DataFrame()
    # Check if the period of observation is too short for week encoding
    if abs(pd.Timedelta(series[0] - series[-1]).total_seconds()) < 604800 and (mask is None):
        return pd.DataFrame()
    # Generate SIN, COS and weekend encoding vectors
    weekend = (np.array(series.dt.dayofweek) // 5 - 0.2857) / 0.488
    # Check if the variance of values too small for weekend encoding
    if np.std(weekend) < 0.7 and (mask is None):
        return pd.DataFrame()
    if (mask is not None) and ((name + '_is_weekend') not in mask):
        return pd.DataFrame()
    columns = [name + '_is_weekend']
    return pd.DataFrame(weekend,
                        columns=columns,
                        index=series.values)


def generate_year_features(series: pd.Series, name='', mask=None):
    # Check data type
    if not pd.api.types.is_datetime64_ns_dtype(series):
        return pd.DataFrame()
    # Check if the period of observation is too short for year encoding
    if abs(pd.Timedelta(series[0] - series[-1]).total_seconds()) < 31536000 and (mask is None):
        return pd.DataFrame()
    # Generate linear encoding vector
    year_lin = (np.array(series.dt.dayofyear / 366) - 0.5) / 0.3
    # Check if the variance of values too small for day encoding
    if np.std(year_lin) < 0.7 and (mask is None):
        return pd.DataFrame()
    # Generate SIN, COS and weekend encoding vectors
    year_sin = np.sin(year_lin * (2 * pi * 0.3)) / 0.7071
    year_cos = np.cos(year_lin * (2 * pi * 0.3)) / 0.7071
    # Check if the variance of values too small for year encoding
    if (np.std(year_sin) < 0.8 or np.std(year_cos) < 0.8) and (mask is None):
        return pd.DataFrame()
    if (mask is not None) and ((name + '_year_lin') not in mask):
        return pd.DataFrame()
    columns = [name + '_year_lin', name + '_year_sin', name + '_year_cos']
    return pd.DataFrame(np.vstack((year_lin, year_sin, year_cos)).T,
                        columns=columns,
                        index=series.values)


def get_corr_weights(data: pd.DataFrame, y: pd.Series):
    weights = pd.DataFrame()
    for nm, col in data.iteritems():
        weights[nm] = pd.Series(abs(np.corrcoef(y.values, col.values)[0, 1]) / col.std())
    return weights


def generate_timedata_features(data: pd.DataFrame, y: Union[pd.Series, None], mask=None, calculate_weights='True'):
    features = pd.DataFrame()

    if pd.api.types.is_datetime64_ns_dtype(data.index):
        hour_features = generate_hour_features(data.index.to_series(), name='index', mask=mask)
        day_features = generate_day_features(data.index.to_series(), name='index', mask=mask)
        week_features = generate_week_features(data.index.to_series(), name='index', mask=mask)
        weekend_feature = generate_weekend_feature(data.index.to_series(), name='index', mask=mask)
        year_features = generate_year_features(data.index.to_series(), name='index', mask=mask)
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
            hour_features = generate_hour_features(series[1], name=series[0] + '_', mask=mask)
            day_features = generate_day_features(series[1], name=series[0] + '_', mask=mask)
            week_features = generate_week_features(series[1], name=series[0] + '_', mask=mask)
            weekend_feature = generate_weekend_feature(series[1], name=series[0] + '_', mask=mask)
            year_features = generate_year_features(series[1], name=series[0] + '_', mask=mask)
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
        weights = get_corr_weights(features, y).iloc[0]
    else:
        weights = None

    return features, weights


# Generating autocorrelation features

def generate_shift_features(data: pd.Series, name: str = '', review_period: int = 90, forecast_period: int = 1,
                            mask: list = None):
    features = pd.DataFrame()
    weights = pd.DataFrame()
    mean = data.mean()
    std = data.std()

    # Filling feature periods lists
    short_list = []
    long_list = []
    if mask is None:
        short_list = list(range(1, forecast_period))
        long_list = list(range(forecast_period, review_period + 1))
    else:
        for i in mask:
            if i < forecast_period:
                short_list.append(i)
            else:
                long_list.append(i)

    # Generate short (<forecast_period) features
    for i in short_list:
        shifted_data = data.shift(i, fill_value=mean)
        weights[name + '_shift' + str(i)] = pd.Series(
            1 / std * abs(np.corrcoef(data.values, shifted_data.values)[0, 1]) * i / forecast_period)
        features[name + '_shift' + str(i)] = shifted_data - mean

    # Generate long (>forecast_period) features
    for i in long_list:
        shifted_data = data.shift(i, fill_value=mean)
        weights[name + '_shift' + str(i)] = pd.Series(
            1 / std * abs(np.corrcoef(data.values, shifted_data.values)[0, 1]))
        features[name + '_shift' + str(i)] = shifted_data - mean

    return features, weights.iloc[0]


def generate_indicators(data: pd.Series,
                        name: str = '',
                        review_period: int = 168,
                        forecast_period: int = 1,
                        ema_mask: list = None,
                        dma_mask: list = None,
                        tma_mask: list = None,
                        qma_mask: list = None):
    mean = data.mean()

    # Calculate logarithmic step for EMA windows, calculate windows and alphas for EMA, DMA, TMA, QMA
    if (ema_mask is None) and (dma_mask is None) and (tma_mask is None) and (qma_mask is None):
        ema_step = (review_period / forecast_period * 4) ** (1 / 31)
        ema_windows = np.array([forecast_period / 2 * ema_step ** i for i in range(32)])
    else:
        general_mask = set()
        if ema_mask is not None:
            general_mask.update(ema_mask)
        if dma_mask is not None:
            general_mask.update(dma_mask)
        if tma_mask is not None:
            general_mask.update(tma_mask)
        if qma_mask is not None:
            general_mask.update(qma_mask)
        ema_windows = list(general_mask)
        ema_windows.sort()
        ema_windows = np.array(ema_windows)
    alphas = 2 / (ema_windows + 1)

    # Fill buffer vectors
    ema_buffer = np.ones((len(alphas),), dtype=float) * (data.iloc[0] - mean)
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
        ema_buffer = ema_buffer * (1 - alphas) + alphas * (data[i] - mean)
        result_ema[i] = ema_buffer
        result_dma[i] = dma_buffer
        result_tma[i] = tma_buffer
        result_qma[i] = qma_buffer

    # Convert result into pandas DaraFrame
    features = pd.DataFrame(np.hstack((result_ema, result_dma, result_tma, result_qma)), index=data.index,
                            columns=columns)

    # Return in case of non-masked generation
    if not ((ema_mask is None) and (dma_mask is None) and (tma_mask is None) and (qma_mask is None)):
        # Clean features by masks in case of masked generation
        drop_columns = set()
        if ema_mask is not None:
            drop_columns.update([name + '_ema_' + f'{window:.1f}' for window in ema_mask])
        if dma_mask is not None:
            drop_columns.update([name + '_dma_' + f'{window:.1f}' for window in dma_mask])
        if tma_mask is not None:
            drop_columns.update([name + '_tma_' + f'{window:.1f}' for window in tma_mask])
        if qma_mask is not None:
            drop_columns.update([name + '_qma_' + f'{window:.1f}' for window in qma_mask])
        for col in features.columns:
            if col not in drop_columns:
                features.drop(col, axis=1, inplace=True)

    # Get weights
    weights = get_corr_weights(features, data).iloc[0]

    return features, weights, ema_buffer, dma_buffer, tma_buffer, qma_buffer


def get_feature_masks(columns) -> object:
    shift_mask = []
    ema_mask = []
    dma_mask = []
    tma_mask = []
    qma_mask = []
    timedata_mask = set()
    timedata_set = {'index_hour_lin', 'index_hour_sin',
                    'index_hour_cos', 'index_day_lin',
                    'index_day_sin', 'index_day_cos',
                    'index_week_lin', 'index_week_sin',
                    'index_week_cos', 'index_is_weekend',
                    'index_year_lin', 'index_year_sin',
                    'index_year_cos'}

    for col in columns:
        if len(re.findall('_shift', col)) > 0:
            search = re.split('_shift', col)
            shift_mask.append(int(search[-1]))
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
        if col in timedata_set:
            timedata_mask.add(col)

    return shift_mask, ema_mask, dma_mask, tma_mask, qma_mask, timedata_mask


# Calculation of single rows of features
def calc_indicators_row(y: float, mean: float,
                        ema_buffer: np.array,
                        dma_buffer: np.array,
                        tma_buffer: np.array,
                        qma_buffer: np.array,
                        ema_mask: list,
                        dma_mask: list,
                        tma_mask: list,
                        qma_mask: list,
                        name: str = '', ):
    # Calculate windows and alphas for EMA, DMA, TMA, QMA
    general_mask = set()
    if ema_mask is not None:
        general_mask.update(ema_mask)
    if dma_mask is not None:
        general_mask.update(dma_mask)
    if tma_mask is not None:
        general_mask.update(tma_mask)
    if qma_mask is not None:
        general_mask.update(qma_mask)
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

    qma_buffer = qma_buffer * (1 - alphas) + alphas * tma_buffer
    tma_buffer = tma_buffer * (1 - alphas) + alphas * dma_buffer
    dma_buffer = dma_buffer * (1 - alphas) + alphas * ema_buffer
    ema_buffer = ema_buffer * (1 - alphas) + alphas * (y - mean)

    # Convert result into pandas DaraFrame
    features = pd.Series(
        np.hstack(
            (ema_buffer, dma_buffer, tma_buffer, qma_buffer)),
        index=columns)

    # Clean features by masks in case of masked generation
    drop_columns = set()
    if ema_mask is not None:
        drop_columns.update([name + '_ema_' + f'{window:.1f}' for window in ema_mask])
    if dma_mask is not None:
        drop_columns.update([name + '_dma_' + f'{window:.1f}' for window in dma_mask])
    if tma_mask is not None:
        drop_columns.update([name + '_tma_' + f'{window:.1f}' for window in tma_mask])
    if qma_mask is not None:
        drop_columns.update([name + '_qma_' + f'{window:.1f}' for window in qma_mask])
    for col in features.index:
        if col not in drop_columns:
            features.drop(col, inplace=True)

    return features, ema_buffer, dma_buffer, tma_buffer, qma_buffer


def calc_shift_row(data: np.array, mean: float, name: str = '', mask=None):
    features = pd.DataFrame()

    # Generate short features
    for i in mask:
        features[name + '_shift' + str(i)] = pd.Series(data[-i] - mean)

    return features
