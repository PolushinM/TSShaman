import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDRegressor


# Best alpha estimation

def get_best_l1_alpha(X, y, cv=16, random_state=0):
    return get_best_lr_alpha(X, y, cv=cv, random_state=random_state, penalty='l1')


def get_best_l2_alpha(X, y, cv=16, random_state=0):
    return get_best_lr_alpha(X, y, cv=cv, random_state=random_state, penalty='l2')


def get_best_lr_alpha(X, y, cv=16, random_state=0, penalty='l1'):
    alphas = [0.005 * 5 ** i for i in range(12)]
    clf = GridSearchCV(
        estimator=SGDRegressor(penalty=penalty,
                               eta0=0.01,
                               power_t=0.35,
                               max_iter=10000,
                               random_state=random_state),
        param_grid={'alpha': alphas},
        cv=cv)
    clf.fit(X, y)

    # Results of alpha grid search
    gs_results = clf.cv_results_['mean_test_score'] - clf.cv_results_['std_test_score']
    best_result_num = np.argmax(gs_results)
    best_alpha = alphas[best_result_num]

    alphas = [best_alpha / 4.9 * 1.89 ** i for i in range(6)]
    clf = GridSearchCV(
        estimator=SGDRegressor(penalty=penalty,
                               eta0=0.005,
                               power_t=0.25,
                               max_iter=10000,
                               random_state=random_state),
        param_grid={'alpha': alphas},
        cv=cv)
    clf.fit(X, y)

    # Results of alpha grid search
    gs_results = clf.cv_results_['mean_test_score'] - clf.cv_results_['std_test_score']
    best_result_num = np.argmax(gs_results)

    return alphas[best_result_num]