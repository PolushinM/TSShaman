import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score
from optimization import get_best_l1_alpha
from _logger import *


def l1_feature_select(X: pd.DataFrame, y, estimator, strength, cv, random_state, n_jobs=-1):

    X_std = X / X.std()

    best_alpha, best_score = get_best_l1_alpha(X_std, y, estimator=estimator, cv=cv, random_state=random_state, n_jobs=n_jobs)
    threshold_score = 1.0

    while threshold_score > best_score * (1 - strength):
        best_alpha *= 1.2
        estimator.set_params(alpha=best_alpha)
        threshold_score = cross_val_score(estimator, X_std, y, cv=cv, n_jobs=n_jobs).mean()

    estimator.fit(X_std, y)
    drop_features = []
    columns = X.columns
    for i in range(len(estimator.coef_)):
        if abs(estimator.coef_[i]) < 0.002:
            drop_features.append(columns[i])
    logger.debug(f'Threshold score={threshold_score:.4f}')
    logger.debug(f'Number of dropped={len(drop_features)}')

    return drop_features, best_alpha
