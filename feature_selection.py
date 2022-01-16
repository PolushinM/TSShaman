import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import SGDRegressor


def l1_feature_select(X: pd.DataFrame, y, strength=0.002, cv=10, random_state=0):
    alphas = [0.0005 * 3 ** i for i in range(20)]
    clf = GridSearchCV(
        estimator=SGDRegressor(penalty='l1',
                               eta0=0.02,
                               power_t=0.35,
                               max_iter=10000,
                               random_state=random_state),
        param_grid={'alpha': alphas},
        cv=cv)

    clf.fit(X, y)

    # Results of alpha grid search
    gs_results = clf.cv_results_['mean_test_score'] - clf.cv_results_['std_test_score']
    best_result_num = np.argmax(gs_results)
    best_result = gs_results[best_result_num]
    best_alpha = alphas[best_result_num]
    i = best_result_num
    while i < len(gs_results) and gs_results[i] > best_result * (1 - strength):
        i += 1

    model = SGDRegressor(penalty='l1',
                         alpha=alphas[i],
                         eta0=0.01,
                         power_t=0.35,
                         max_iter=10000,
                         random_state=random_state)
    model.fit(X, y)
    drop_features = []
    columns = X.columns
    for i in range(len(model.coef_)):
        if abs(model.coef_[i]) < 0.005:
            drop_features.append(columns[i])
    score = cross_val_score(model, X, y).mean()
    coefs = model.coef_

    return drop_features, best_alpha, score, coefs


def drop_feature_select(coefs: pd.DataFrame, threshold=0.05):
    drop_features = []
    summ = coefs.abs().sum()[0] * threshold
    for i in range(len(coefs) - 1, 0, -1):
        summ -= abs(coefs.iloc[i].values[0])
        if summ > 0:
            drop_features.append(coefs.index[i])
    return drop_features


def coef_feature_select(X: pd.DataFrame, y, penalty='l2', alpha=0.01, cv: int = 10, random_state: int = 0):
    thresholds = [0.01 * 1.53 ** i for i in range(10)]
    scores = []
    model = SGDRegressor(penalty=penalty,
                         eta0=0.002,
                         power_t=0.2,
                         max_iter=50000,
                         alpha=alpha,
                         random_state=random_state)

    # X = X / X.std()
    model.fit(X, y)
    coefs = pd.DataFrame(model.coef_,
                         index=X.columns,
                         columns=['Coef']).sort_values(by='Coef', key=abs, ascending=False)

    model = SGDRegressor(penalty=penalty,
                         eta0=0.015,
                         power_t=0.2,
                         max_iter=10000,
                         random_state=random_state)

    for threshold in thresholds:
        drop_features = drop_feature_select(coefs, threshold=threshold)
        X_dropped = X.drop(drop_features, axis=1)
        cv_scores = cross_val_score(model, X_dropped, y, cv=cv)
        scores.append(cv_scores.mean() - cv_scores.std())
    best_result_num = scores.index(max(scores))
    best_threshold = thresholds[best_result_num]

    thresholds_accurate = [best_threshold / 1.8 * 1.125 ** i for i in range(11)]

    for threshold in thresholds_accurate:
        drop_features = drop_feature_select(coefs, threshold=threshold)
        X_dropped = X.drop(drop_features, axis=1)
        cv_scores = cross_val_score(model, X_dropped, y, cv=cv)
        scores.append(cv_scores.mean() - cv_scores.std())
    best_result_num = scores.index(max(scores))
    thresholds = thresholds + thresholds_accurate
    threshold = round(thresholds[best_result_num], 3)

    return drop_feature_select(coefs, threshold)
