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
    split_index = np.array_split(np.arange(0, len(y)), cv)
    train_score = np.zeros(cv)
    val_score = np.zeros(cv)
    for i in range(cv):
        train_samples = X[~split_index[i]]
        train_labels = y[~split_index[i]]
        estimator.fit(train_samples, train_labels)
        train_score[i] = scoring(train_labels, estimator.predict(train_samples))
        val_score[i] = scoring(y[split_index[i]], estimator.predict(X[split_index[i]]))

    return np.mean(train_score), np.mean(val_score)
