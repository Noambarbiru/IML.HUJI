from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        data_x, data_y = load_dataset(f)
        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def callback_function(fit: Perceptron, x: np.ndarray, y: int):
            losses.append(fit.loss(data_x, data_y))

        Perceptron(include_intercept=False,callback=callback_function).fit(data_x, data_y)

        # Plot figure of loss as function of fitting iteration
        x_axis = np.arange(1, len(losses) + 1)
        go.Figure([go.Scatter(x=x_axis, y=losses, mode='lines')],
                  layout=go.Layout(title=r"$\text{Training Loss as function of Training Fitting Iteration of " + n +
                                         r" Dataset}$",
                                   xaxis_title=r"$\text{training iterations}$",
                                   yaxis_title=r"$\text{training loss values}$",
                                   height=300)).show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        data_x, data_y = load_dataset(f)
        # Fit models and predict over training set

        lda_model = LDA().fit(data_x, data_y)
        lda_y_pred = lda_model.predict(data_x)

        gnb_model = GaussianNaiveBayes().fit(data_x, data_y)
        gnb_y_pred = gnb_model.predict(data_x)

        # Plot a figure with two subplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        lda_accuracy = accuracy(data_y, lda_y_pred)
        gnb_accuracy = accuracy(data_y, gnb_y_pred)

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=(
                                f"Gaussian Naive Bayes predictions: accuracy = {gnb_accuracy}",
                                f"LDA predictions: accuracy = {lda_accuracy}"))
        fig.update_layout(title_text=f"prediction on dataset {f}", title_x=0.5, title_y=1)
        fig.update_xaxes(title_text="feature 1 values")
        fig.update_yaxes(title_text="feature 2 values")

        # Add traces for data-points setting symbols and colors
        symbols = np.array(['circle', 'star', 'triangle-up'])
        fig.add_trace(go.Scatter(x=data_x[:, 0], y=data_x[:, 1], mode="markers", showlegend=False,
                                 marker=dict(color=gnb_y_pred, symbol=symbols[data_y.astype(int)],
                                             line=dict(color="black", width=1),
                                             colorscale=[[0, 'green'], [0.5, 'red'], [1, 'blue']])),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=data_x[:, 0], y=data_x[:, 1], mode="markers", showlegend=False,
                                 marker=dict(color=lda_y_pred, symbol=symbols[data_y.astype(int)],
                                             line=dict(color="black", width=1),
                                             colorscale=[[0, 'green'], [0.5, 'red'], [1, 'blue']])),
                      row=1, col=2)
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                 marker=dict(color=['green']), legendgroup=str(np.unique(data_y)[0]),
                                 name="predicted label color - "+str(np.unique(data_y)[0])), row=1, col=1)
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                 marker=dict(color=['red']), legendgroup=str(np.unique(data_y)[1]),
                                 name="predicted label color - "+str(np.unique(data_y)[1])), row=1, col=1)
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                 marker=dict(color=['blue']), legendgroup=str(np.unique(data_y)[2]),
                                 name="predicted label color - "+str(np.unique(data_y)[2])), row=1, col=1)
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                 marker=dict(color=['black'],symbol=symbols[0]), legendgroup=str(np.unique(data_y)[0]),
                                 name="true label shape- "+str(np.unique(data_y)[0])), row=1, col=1)
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                 marker=dict(color=['black'],symbol=symbols[1]), legendgroup=str(np.unique(data_y)[1]),
                                 name="true label shape- "+str(np.unique(data_y)[1])), row=1, col=1)
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                 marker=dict(color=['black'],symbol=symbols[2]), legendgroup=str(np.unique(data_y)[2]),
                                 name="true label shape - "+str(np.unique(data_y)[2])), row=1, col=1)

        # Add `X` dots specifying fitted Gaussians' means
        fig.add_trace(go.Scatter(x=gnb_model.mu_[:, 0], y=gnb_model.mu_[:, 1], mode="markers", showlegend=False,
                                 marker=dict(color='black', symbol='x',), marker_size=15), row=1, col=1)
        fig.add_trace(go.Scatter(x=lda_model.mu_[:, 0], y=lda_model.mu_[:, 1], mode="markers", showlegend=False,
                                 marker=dict(color='black', symbol='x'), marker_size=15), row=1, col=2)

        # Add ellipses depicting the covariances of the fitted Gaussians
        for i in range(len(np.unique(data_y))):
            fig.add_trace(get_ellipse(lda_model.mu_[i], lda_model.cov_).update(showlegend=False), row=1, col=2)
            fig.add_trace(get_ellipse(gnb_model.mu_[i], np.diag(gnb_model.vars_[i])).update(showlegend=False), row=1,
                          col=1)

        fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
