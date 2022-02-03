from .optimization import get_best_l1_alpha
from ._logger import *
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score


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


def corcoeff_feature_selection(X: pd.DataFrame, y: pd.Series, strength: float = 0.5):
    corrcoefs = np.ndarray((X.shape[1],), dtype=float)
    i: int = 0
    dropped_features = []

    for _, col in X.iteritems():
        corrcoefs[i] = abs(np.corrcoef(y.values, col.values)[0, 1])
        i += 1
    threshold = np.quantile(corrcoefs, strength)
    logger.debug(f'Corrcoef threshold={threshold:.5f}')

    i = 0
    for col in X.columns:
        if corrcoefs[i] < threshold:
            dropped_features.append(col)
        i += 1

    return dropped_features
