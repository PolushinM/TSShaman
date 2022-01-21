import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import SGDRegressor
from random import random, seed


def get_best_l1_alpha(X, y, cv=16, random_state=0):
    return get_best_lr_alpha(X, y, cv=cv, random_state=random_state, penalty='l1')


def get_best_l2_alpha(X, y, cv=16, random_state=0):
    return get_best_lr_alpha(X, y, cv=cv, random_state=random_state, penalty='l2')


def get_best_lr_alpha_old(X, y, cv=16, random_state=0, penalty='l1'):
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
    gs_results = clf.cv_results_['mean_test_score']
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
    gs_results = clf.cv_results_['mean_test_score']
    best_result_num = np.argmax(gs_results)

    return alphas[best_result_num], clf.cv_results_['mean_test_score'][best_result_num]


def calculate_next_point(X, y, interval, penalty, random_state, cv):
    left_alpha, right_alpha, left_score, right_score = interval
    new_alpha = ((left_alpha ** left_score * right_alpha ** right_score) ** (1 / (left_score + right_score)))
    new_score = cross_val_score(SGDRegressor(penalty=penalty,
                                             alpha=new_alpha,
                                             eta0=0.005,
                                             power_t=0.25,
                                             max_iter=10000,
                                             random_state=random_state),
                                X, y, cv=cv).mean()

    return new_alpha, new_score


def get_best_lr_alpha(X, y, cv=16, random_state=0, penalty='l1'):
    alphas = [0.0005 * 50 ** i for i in range(6)]

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
    alpha_table = pd.Series(data=clf.cv_results_['mean_test_score'], index=alphas)
    alpha_table.sort_values(ascending=False, inplace=True)
    alphas = [alpha_table.index[0]/20 * 4.47 ** i for i in range(4)]
    clf = GridSearchCV(
        estimator=SGDRegressor(penalty=penalty,
                               eta0=0.01,
                               power_t=0.35,
                               max_iter=10000,
                               random_state=random_state),
        param_grid={'alpha': alphas},
        cv=cv)
    clf.fit(X, y)
    alpha_table = alpha_table.append(pd.Series(data=clf.cv_results_['mean_test_score'], index=alphas))

    left_alpha = alpha_table.index[0]
    right_alpha = alpha_table.index[1]
    divider = (right_alpha / left_alpha)**0.8

    alpha_table.sort_values(ascending=False, inplace=True)
    left_alpha = alpha_table.index[0]
    right_alpha = left_alpha / divider
    left_score = alpha_table.iloc[0]
    right_score = alpha_table.iloc[0] * 0.8
    interval = (left_alpha, right_alpha, left_score, right_score)
    next_point = calculate_next_point(X, y, interval=interval, penalty=penalty, random_state=random_state, cv=cv)
    alpha_table[next_point[0]] = next_point[1]

    alpha_table.sort_values(ascending=False, inplace=True)
    left_alpha = alpha_table.index[0]
    right_alpha = alpha_table.index[4]
    left_score = alpha_table.iloc[0]
    right_score = alpha_table.iloc[4]
    interval = (left_alpha, right_alpha, left_score, right_score)
    next_point = calculate_next_point(X, y, interval=interval, penalty=penalty, random_state=random_state, cv=cv)
    alpha_table[next_point[0]] = next_point[1]

    alpha_table.sort_values(ascending=False, inplace=True)
    left_alpha = alpha_table.index[0]
    right_alpha = alpha_table.index[2]
    left_score = alpha_table.iloc[0]
    right_score = alpha_table.iloc[2]
    interval = (left_alpha, right_alpha, left_score, right_score)
    next_point = calculate_next_point(X, y, interval=interval, penalty=penalty, random_state=random_state, cv=cv)
    alpha_table[next_point[0]] = next_point[1]
    alpha_table.sort_values(ascending=False, inplace=True)

    alpha_table.sort_values(ascending=False, inplace=True)
    left_alpha = alpha_table.index[0]
    right_alpha = alpha_table.index[1]
    left_score = alpha_table.iloc[0]
    right_score = alpha_table.iloc[1]
    interval = (left_alpha, right_alpha, left_score, right_score)
    next_point = calculate_next_point(X, y, interval=interval, penalty=penalty, random_state=random_state, cv=cv)
    alpha_table[next_point[0]] = next_point[1]
    alpha_table.sort_values(ascending=False, inplace=True)

    alpha_table.sort_values(ascending=False, inplace=True)
    left_alpha = alpha_table.index[0]
    right_alpha = alpha_table.index[1]
    left_score = alpha_table.iloc[0]
    right_score = alpha_table.iloc[1]
    interval = (left_alpha, right_alpha, left_score, right_score)
    next_point = calculate_next_point(X, y, interval=interval, penalty=penalty, random_state=random_state, cv=cv)
    alpha_table[next_point[0]] = next_point[1]
    alpha_table.sort_values(ascending=False, inplace=True)

    return alpha_table.index[0], alpha_table.iloc[0]
