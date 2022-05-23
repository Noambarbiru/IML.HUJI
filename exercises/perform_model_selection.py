from __future__ import annotations

import random

import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    x = np.linspace(-1.2, 2, n_samples)
    y_ = (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)
    y = y_ + np.random.normal(0, np.sqrt(noise), size=n_samples)

    train_x, train_y, test_x, test_y = split_train_test(pd.DataFrame(x), pd.DataFrame(y), 2 / 3)
    train_x, train_y, test_x, test_y = train_x.to_numpy().reshape((-1,)), train_y.to_numpy().reshape(
        (-1,)), test_x.to_numpy().reshape((-1,)), test_y.to_numpy().reshape((-1,))
    fig = go.Figure(
        layout=go.Layout(
            title=rf"$\textbf{{y =(x+3)(x+2)(x+1)(x-1)(x-2) + ε, ε ~ N(0, {noise})}}$"))
    fig.add_trace(go.Scatter(x=x, y=y_, mode='markers', name="true (noiseless) model"))
    fig.add_trace(go.Scatter(x=train_x, y=train_y, mode='markers', name="train set"))
    fig.add_trace(go.Scatter(x=test_x, y=test_y, mode='markers', name="test set"))
    fig.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    avg_train_error = np.zeros(11)
    avg_val_error = np.zeros(11)
    k_arr = np.arange(0, 11)
    for k in k_arr:
        model = PolynomialFitting(k)
        avg_train_error[k], avg_val_error[k] = cross_validate(model, train_x, train_y, mean_square_error)
    fig = go.Figure(
        layout=go.Layout(
            title=rf"$\textbf{{The Average Training- And Validation Errors As Function Of K (with σ^2 = {noise})}}$"))
    fig.add_trace(go.Scatter(x=k_arr, y=avg_train_error, mode='markers', name="average train error"))
    fig.add_trace(go.Scatter(x=k_arr, y=avg_val_error, mode='markers', name="average validation error"))
    fig.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    k_star = np.argmin(avg_val_error)
    model = PolynomialFitting(k_star).fit(train_x, train_y)
    test_error_k_star = mean_square_error(test_y, model.predict(test_x))
    # print("For σ^2 = %d, the k who gives the minimum validation error is %d.\nThe error of the model of k = %d on the test is %.2f.\n"
    #       % (noise,k_star, k_star, test_error_k_star))


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    data, target = datasets.load_diabetes(return_X_y=True)
    index_train = np.random.choice(np.arange(0, len(data)), n_samples, replace=False)
    train_x, train_y, test_x, test_y = data[index_train], target[index_train], data[~index_train], target[~index_train]

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    regularization_parameter_ridge = np.linspace(5 * 10 ** -4, 0.1, n_evaluations)
    avg_train_error_ridge = np.zeros(n_evaluations)
    avg_val_error_ridge = np.zeros(n_evaluations)
    for i in range(n_evaluations):
        model_ridge = RidgeRegression(regularization_parameter_ridge[i])
        avg_train_error_ridge[i], avg_val_error_ridge[i] = cross_validate(model_ridge, train_x, train_y,
                                                                          mean_square_error)
    fig = go.Figure(
        layout=go.Layout(
            title=rf"$\textbf{{Ridge Regularization: The Average Training- And Validation Errors As Function Of λ}}$"))
    fig.add_trace(go.Scatter(x=regularization_parameter_ridge, y=avg_train_error_ridge, mode='markers',
                             name="ridge average train error"))
    fig.add_trace(
        go.Scatter(x=regularization_parameter_ridge, y=avg_val_error_ridge, mode='markers',
                   name="ridge average validation error"))
    fig.show()

    regularization_parameter_lasso = np.linspace(0.01, 2, n_evaluations)
    avg_train_error_lasso = np.zeros(n_evaluations)
    avg_val_error_lasso = np.zeros(n_evaluations)
    for i in range(n_evaluations):
        model_lasso = Lasso(alpha=regularization_parameter_lasso[i])
        avg_train_error_lasso[i], avg_val_error_lasso[i] = cross_validate(model_lasso, train_x, train_y,
                                                                          mean_square_error)
    fig = go.Figure(
        layout=go.Layout(
            title=rf"$\textbf{{Lasso Regularization: The Average Training- And Validation Errors As Function Of λ}}$"))
    fig.add_trace(go.Scatter(x=regularization_parameter_lasso, y=avg_train_error_lasso, mode='markers',
                             name="lasso average train error"))
    fig.add_trace(
        go.Scatter(x=regularization_parameter_lasso, y=avg_val_error_lasso, mode='markers',
                   name="lasso average validation error"))
    fig.show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    best_regularization_parameter_ridge = regularization_parameter_ridge[np.argmin(avg_val_error_ridge)]
    best_regularization_parameter_lasso = regularization_parameter_lasso[np.argmin(avg_val_error_lasso)]
    ridge_test_error = mean_square_error(test_y, RidgeRegression(best_regularization_parameter_ridge).fit(train_x,
                                                                                                          train_y).predict(
        test_x))
    lasso_test_error = mean_square_error(test_y,
                                         Lasso(best_regularization_parameter_lasso).fit(train_x, train_y).predict(
                                             test_x))
    linear_test_error = mean_square_error(test_y, LinearRegression().fit(train_x, train_y).predict(test_x))
    # print(
    #     "For ridge: best parameter- %f error- %.2f\nFor lasso: best parameter- %f error- %.2f\nFor linear: error- %.2f\n"
    #     % (best_regularization_parameter_ridge, ridge_test_error, best_regularization_parameter_lasso, lasso_test_error,
    #        linear_test_error))


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)
    select_regularization_parameter()
