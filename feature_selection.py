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
    X_std = X / X.std()
    model.fit(X_std, y)
    drop_features = []
    columns = X.columns
    for i in range(len(model.coef_)):
        if abs(model.coef_[i]) < 0.005:
            drop_features.append(columns[i])
    score = cross_val_score(model, X, y).mean()
    coefs = model.coef_

    return drop_features, best_alpha, score, coefs

