from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    train_score = np.zeros(cv)
    val_score = np.zeros(cv)
    range_arr = np.arange(len(X))
    for i in range(cv):
        train_samples = X[range_arr % cv != i]
        train_labels = y[range_arr % cv != i]
        val_samples = X[range_arr % cv == i]
        val_labels = y[range_arr % cv == i]
        estimator.fit(train_samples, train_labels)
        train_pred = estimator.predict(train_samples)
        val_pred = estimator.predict(val_samples)
        train_score[i] = scoring(train_labels, train_pred)
        val_score[i] = scoring(val_labels, val_pred)

    return np.mean(train_score), np.mean(val_score)
