import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import SGDRegressor


def get_best_l1_alpha(X, y, estimator, cv=16, random_state=0, n_jobs=None):
    return get_best_lr_alpha(X, y, estimator=estimator, cv=cv, random_state=random_state, penalty='l1', n_jobs=n_jobs)


def get_best_l2_alpha(X, y, estimator, cv=16, random_state=0, n_jobs=None):
    return get_best_lr_alpha(X, y, estimator=estimator, cv=cv, random_state=random_state, penalty='l2', n_jobs=n_jobs)


def get_best_lr_alpha(X, y, cv, estimator, random_state, penalty, n_jobs):

    estimator.set_params(random_state=random_state, penalty=penalty)

    alpha_table = exponential_alpha_greed_search(X, y, estimator,
                                                 start=0.0005,
                                                 exponential_step=50,
                                                 number=6,
                                                 cv=cv,
                                                 n_jobs=n_jobs)

    alpha_table = alpha_table.append(exponential_alpha_greed_search(X, y, estimator,
                                                                    start=alpha_table.index[0]/20,
                                                                    exponential_step=4.47,
                                                                    number=4,
                                                                    cv=cv,
                                                                    n_jobs=n_jobs))
    alpha_table.sort_values(ascending=False, inplace=True)

    alpha_table = add_point_to_table(X, y, alpha_table, 0, 1, estimator, cv, n_jobs=n_jobs, reverse=True)
    alpha_table = add_point_to_table(X, y, alpha_table, 0, 4, estimator, cv, n_jobs=n_jobs)
    alpha_table = add_point_to_table(X, y, alpha_table, 0, 2, estimator, cv, n_jobs=n_jobs)
    alpha_table = add_point_to_table(X, y, alpha_table, 0, 1, estimator, cv, n_jobs=n_jobs)
    alpha_table = add_point_to_table(X, y, alpha_table, 0, 1, estimator, cv, n_jobs=n_jobs)

    return alpha_table.index[0], alpha_table.iloc[0]


def exponential_alpha_greed_search(X, y, estimator,
                                   start,
                                   exponential_step,
                                   number,
                                   cv,
                                   n_jobs):

    alphas = [start * exponential_step ** i for i in range(number)]
    clf = GridSearchCV(estimator, param_grid={'alpha': alphas}, cv=cv, n_jobs=n_jobs).fit(X, y)
    return pd.Series(data=clf.cv_results_['mean_test_score'], index=alphas).sort_values(ascending=False)


def add_point_to_table(X, y, alpha_table, index1, index2, estimator, cv, n_jobs, reverse=False):

    left_alpha = alpha_table.index[index1]
    right_alpha = alpha_table.index[index2]
    left_score = alpha_table.iloc[index1]
    right_score = alpha_table.iloc[index2]

    if reverse:
        divider = (right_alpha / left_alpha) ** 0.8
        right_alpha = left_alpha / divider
        right_score = alpha_table.iloc[index1] * 0.8

    interval = (left_alpha, right_alpha, left_score, right_score)
    next_point = calculate_next_point(X, y, estimator=estimator, interval=interval, cv=cv, n_jobs=n_jobs)

    alpha_table[next_point[0]] = next_point[1]

    return alpha_table.sort_values(ascending=False)


def calculate_next_point(X, y, interval, estimator, cv, n_jobs):
    left_alpha, right_alpha, left_score, right_score = interval
    new_alpha = ((left_alpha ** left_score * right_alpha ** right_score) ** (1 / (left_score + right_score)))
    new_score = cross_val_score(estimator, X, y, cv=cv, n_jobs=n_jobs).mean()
    return new_alpha, new_score

