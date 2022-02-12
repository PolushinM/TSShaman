import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from catboost import Pool, EFstrType

from .optimization import get_best_l1_alpha
from ._logger import *


def l1_feature_select(X: pd.DataFrame, y, estimator, strength, cv, random_state, n_jobs=-1):
    X_std = X / X.std()

    best_alpha, best_score = get_best_l1_alpha(X_std, y,
                                               estimator=estimator,
                                               cv=cv,
                                               random_state=random_state,
                                               n_jobs=n_jobs)
    threshold_score = 1.0

    if strength > 0:
        while threshold_score > best_score * (1 - strength):
            best_alpha *= 1.2
            estimator.set_params(alpha=best_alpha)
            threshold_score = cross_val_score(estimator, X_std, y, cv=cv, n_jobs=n_jobs).mean()
    else:
        estimator.set_params(alpha=best_alpha)
        threshold_score = best_score

    estimator.fit(X_std, y)
    drop_features = []
    columns = X.columns
    for i in range(len(estimator.coef_)):
        if abs(estimator.coef_[i]) < 1e-3:
            drop_features.append(columns[i])
    logger.debug(f'Threshold score={threshold_score:.4f}')
    logger.debug(f'Number of dropped={len(drop_features)}')

    return drop_features, best_alpha


def corcoeff_feature_selection(X: pd.DataFrame, y: pd.Series, quantile: float = 0.5):
    corrcoefs = np.ndarray((X.shape[1],), dtype=float)
    i: int = 0
    dropped_features = []

    for _, col in X.iteritems():
        corrcoefs[i] = abs(np.corrcoef(y.values, col.values)[0, 1])
        i += 1
    threshold = np.quantile(corrcoefs, quantile)
    logger.debug(f'Corrcoef threshold={threshold:.5f}')

    i = 0
    for col in X.columns:
        if corrcoefs[i] <= threshold:
            dropped_features.append(col)
        i += 1

    return dropped_features


def coefficient_feature_select(X, y, estimator, quantile):
    dropped_features = []

    eval_set_begin_index = - round(y.shape[0] * 0.7)
    estimator.fit(X[:eval_set_begin_index],
                  y[:eval_set_begin_index],
                  eval_set=(
                      X[eval_set_begin_index:],
                      y[eval_set_begin_index:])
                  )

    coefs_table = estimator.get_feature_importance(data=Pool(X),
                                                   reference_data=None,
                                                   type=EFstrType.FeatureImportance,
                                                   prettified=False,
                                                   thread_count=-1,
                                                   verbose=False)

    threshold = np.quantile(np.array(coefs_table), quantile)
    logger.debug(f'Coefficient threshold={threshold / coefs_table.sum() * 100:.3f}%')

    i = 0
    for coef in coefs_table:
        if coef <= threshold:
            dropped_features.append(X.columns[i])
        i += 1

    return dropped_features


def catboost_feature_select(X, y, estimator, quantile):

    eval_set_begin_index = - round(y.shape[0] * 0.65)

    dropped_features = estimator.select_features(X[:eval_set_begin_index],
                                                 y[:eval_set_begin_index],
                                                 eval_set=(
                                                     X[eval_set_begin_index:],
                                                     y[eval_set_begin_index:]),
                                                 features_for_select=X.columns.tolist(),
                                                 num_features_to_select=round((len(X.columns) * (1 - quantile))**0.5 * 12),
                                                 algorithm=None,
                                                 steps=3,
                                                 shap_calc_type=None,
                                                 train_final_model=False,
                                                 verbose=None,
                                                 logging_level=None,
                                                 plot=False)['eliminated_features_names']

    return dropped_features
