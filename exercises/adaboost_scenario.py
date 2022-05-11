import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaBoost = AdaBoost(lambda: DecisionStump(), n_learners)
    adaBoost.fit(train_X, train_y)
    error_train = [adaBoost.partial_loss(train_X, train_y, i) for i in range(1, n_learners + 1)]
    error_test = [adaBoost.partial_loss(test_X, test_y, i) for i in range(1, n_learners + 1)]
    fig = go.Figure(
        layout=go.Layout(
            title=rf"$\textbf{{error on data as a function of the number of fitted learners, with {noise} noise}}$"))
    fig.add_trace(go.Scatter(x=np.arange(1, n_learners), y=error_train, mode='lines', name="train data error"))
    fig.add_trace(go.Scatter(x=np.arange(1, n_learners), y=error_test, mode='lines', name="test data error"))
    fig.write_image("Q1_" + str(noise) + ".png")

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    symbols = np.array(["star", "circle", "x"])
    fig = make_subplots(rows=2, cols=2, subplot_titles=[rf"$\textbf{{{i} learners}}$" for i in T],
                        horizontal_spacing=0.01, vertical_spacing=.03)
    for i in range(len(T)):
        fig.add_traces([decision_surface(lambda x: adaBoost.partial_predict(x, T[i]), lims[0],
                                         lims[1], showscale=False),
                        go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                   marker=dict(color=test_y.astype(int), symbol=symbols[test_y.astype(int)],
                                               colorscale=[custom[0], custom[-1]],
                                               line=dict(color="black", width=1)))],
                       rows=(i // 2) + 1, cols=(i % 2) + 1)

    fig.update_layout(title=rf"$\textbf{{Decision Boundaries For Different Num Of Learners, with {noise} noise}}$",
                      margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)
    fig.write_image("Q2_" + str(noise) + ".png")

    # Question 3: Decision surface of best performing ensemble
    learners_number_for_min_error = error_test.index(min(error_test))
    fig = go.Figure(
        layout=go.Layout(
            title=rf"$\textbf{{Decision Surface, with {noise} noise, For size: {learners_number_for_min_error}, accuracy:{1 - adaBoost.partial_loss(test_X, test_y, learners_number_for_min_error)}}}$",
            margin=dict(t=100)))
    fig.add_traces([decision_surface(
        lambda x: adaBoost.partial_predict(x, learners_number_for_min_error),
        lims[0], lims[1], showscale=False),
        go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                   marker=dict(color=test_y.astype(int), symbol=symbols[test_y.astype(int)],
                               colorscale=[custom[0], custom[-1]],
                               line=dict(color="black", width=1)))])
    fig.write_image("Q3_" + str(noise) + ".png")

    # Question 4: Decision surface with weighted samples
    D = (adaBoost.D_ / np.max(adaBoost.D_))
    D *= 10
    if noise == 0:
        D *= 20
    fig = go.Figure(
        layout=go.Layout(
            title=rf"$\textbf{{ the training set with a point size proportional, with {noise} noise}}$",
            margin=dict(t=100)))
    fig.add_traces([decision_surface(adaBoost.predict, lims[0], lims[1], showscale=False),
                    go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                               marker=dict(color=test_y.astype(int), symbol=symbols[test_y.astype(int)], size=D,
                                           colorscale=[custom[0], custom[-1]],
                                           line=dict(color="black", width=1)))])
    fig.write_image("Q4_" + str(noise) + ".png")


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    np.random.seed(0)
    fit_and_evaluate_adaboost(0.4)
