from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    data = pd.read_csv(filename)
    data = data.dropna(0)
    data["floors"] = data["floors"].astype(int)
    data["zipcode"] = data["zipcode"].astype(int)
    for feature in ["id", "date", "lat", "long"]:
        data = data.drop(feature, 1)

    for feature in ["price", "sqft_living", "sqft_lot", "floors", "grade", "sqft_above", "yr_built",
                    "sqft_living15", "sqft_lot15"]:
        data = data[data[feature] > 0]
    for feature in ["bedrooms", "bathrooms", "view", "sqft_basement", "yr_renovated"]:
        data = data[data[feature] >= 0]
    data = data[data["waterfront"].isin([0, 1])]
    data = data[data["condition"].isin([1, 2, 3, 4, 5])]

    q_hi_lot15 = data["sqft_lot15"].quantile(0.99)
    q_hi_lot = data["sqft_lot"].quantile(0.99)
    data = data[(data["sqft_lot15"] < q_hi_lot15)
                & (data["sqft_lot"] < q_hi_lot)]

    data = data.drop_duplicates()

    x = data.drop("price", 1)
    y = data["price"]
    return x, y


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    sigma_y = np.sqrt(np.var(y))

    for feature in X:
        sigma_x_y = np.cov(X[feature], y)[1][0]
        sigma_x = np.sqrt(np.var(X[feature]))
        f_cov = sigma_x_y / (sigma_x * sigma_y)

        title = f"Correlation of {feature} feature with response<br><sub>Pearson correlation = {f_cov}"
        fig = go.Figure([go.Scatter(x=X[feature], y=y, mode="markers",
                                    marker=dict(color="navy"))],
                        layout=go.Layout(title=title,
                                         xaxis={"title": f"{feature} values"},
                                         yaxis={"title": f"response values"},
                                         font=dict(
                                             size=18)))
        fig.write_image("%spearson-correlation-%s-with-response.png" % (output_path, feature),
                        format="png")


if __name__ == '__main__':
    address_of_data = "house_prices.csv"
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data(address_of_data)

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, y)

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(X, y, .75)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    lr = LinearRegression()
    present = np.arange(10, 101, 1)
    mean_loss = np.zeros(len(present))
    std_loss = np.zeros(len(present))
    for p in range(len(present)):
        result_p = np.zeros(10)
        for i in range(10):
            df = pd.concat([train_X, train_y], axis=1)
            df = df.sample(frac=float(present[p] / 100))
            X_for_train = df.iloc[:, :-1]
            y_for_train = df.iloc[:, -1:]
            lr.fit(X_for_train.to_numpy(), y_for_train.to_numpy())
            result_p[i] = lr.loss(test_X.to_numpy(), test_y.to_numpy())
        mean_loss[p] = np.mean(result_p, axis=0)
        std_loss[p] = np.std(result_p, axis=0)

    data = [go.Scatter(x=present, y=mean_loss - 2 * std_loss, fill=None, mode="lines", line=dict(color="lightgrey"),
                       showlegend=False),
            go.Scatter(x=present, y=mean_loss + 2 * std_loss, fill='tonexty', mode="lines",
                       line=dict(color="lightgrey"),
                       showlegend=False),
            go.Scatter(x=present, y=mean_loss, mode="markers+lines", name="Mean loss of Prediction",
                       marker=dict(color="navy", opacity=.7))]
    fig = go.Figure(data=data,
                    layout=go.Layout(title=f"Mean Loss As Function Of p% Of Training Set",
                                     xaxis={"title": f"percentages of the training set"},
                                     yaxis={"title": f"mean loss over the test set"},
                                     legend=dict(x=0, y=1),
                                     font=dict(size=18),
                                     height=800, width=1000))
    fig.write_image("mean-loss-of-LS-over-training-set-percentage.png", format="png")
