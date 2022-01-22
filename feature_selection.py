import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import SGDRegressor
from optimization import get_best_l1_alpha


def l1_feature_select(X: pd.DataFrame, y, estimator, strength, cv, random_state, n_jobs=-1):

    X_std = X / X.std()

    best_alpha, best_score = get_best_l1_alpha(X_std, y, estimator=estimator, cv=cv, random_state=random_state, n_jobs=n_jobs)
    sgd_score = 1.0

    while sgd_score > best_score * (1 - strength):
        best_alpha *= 1.25
        estimator.set_params(alpha=best_alpha)
        sgd_score = cross_val_score(estimator, X_std, y, cv=cv, n_jobs=n_jobs).mean()

    estimator.fit(X_std, y)
    drop_features = []
    columns = X.columns
    for i in range(len(estimator.coef_)):
        if abs(estimator.coef_[i]) < 0.002:
            drop_features.append(columns[i])
    print(f'sgd_score={sgd_score:.4f}')
    print(len(drop_features))

    return drop_features, best_alpha
