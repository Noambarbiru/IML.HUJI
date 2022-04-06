import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    data = pd.read_csv(filename, parse_dates=[2])
    data = data[data["Month"].isin(np.arange(1, 13))]
    data = data[data["Day"].isin(np.arange(1, 32))]
    data = data[data["Year"] >= 1995]
    data = data[data["Temp"] >= -55.]
    data = data[data["Temp"] < 120.]
    dates = data.Date
    data["DayOfYear"] = (dates - dates.astype('datetime64[Y]')).dt.days
    return data.drop('Temp', 1), data['Temp']


if __name__ == '__main__':
    address_of_output = "C:/Users/brnoo/Desktop/output/"
    address_of_data = "C:/Users/brnoo/Desktop/GitHub/IML.HUJI/datasets/City_Temperature.csv"
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    X, y = load_data(address_of_data)

    # Question 2 - Exploring data for specific country
    df = pd.concat([X, y], axis=1)
    df_israel = df[df["Country"] == 'Israel']
    df_israel["Year"] = df_israel["Year"].astype(str)
    fig = px.scatter(df_israel, x='DayOfYear', y='Temp', color='Year',
                     title=f"relation of the average daily temperature (Temp) and the `DayOfYear`",
                     width=1000, height=800)
    fig.update_layout(font=dict(size=18))
    fig.write_image(address_of_output + "Israel-Temp-as-function-of-DayOfYear.png",
                    format="png")
    df_israel["Year"] = df_israel["Year"].astype(int)

    get_std = lambda x: np.sqrt(np.var(x, axis=0))
    df_std_Temp_for_Month = df_israel.groupby(['Month'], as_index=False).agg({'Temp': get_std})
    fig = px.bar(df_std_Temp_for_Month, x="Month", y="Temp",
                 title=f"Standard Deviation Of The Daily Temperatures For Each Month"
                 , labels={'Temp': 'std of Temp'}, width=1000, height=800)
    fig.update_layout(font=dict(size=18))
    fig.write_image(address_of_output + "Israel-STD-Temp-as-function-of-Month.png",
                    format="png")

    # Question 3 - Exploring differences between countries
    df_mean_Temp_for_Month = df.groupby(['Country', 'Month'], as_index=False).agg({'Temp': [get_std, 'mean']})
    df_mean_Temp_for_Month.columns = ['Country', 'Month', 'std', 'mean']
    fig = px.line(df_mean_Temp_for_Month, x="Month", y="mean", color='Country',
                  error_y='std'
                  , title=f"Average Temperature Of Each Month For Different Countries"
                  , labels={'mean': 'average temperature'}, width=1000, height=800)
    fig.update_layout(font=dict(size=18))
    fig.write_image(address_of_output + "Mean-Temp-as-function-of-Month.png",
                    format="png")

    # Question 4 - Fitting model for different values of `k`
    train_X, train_y, test_X, test_y = split_train_test(df_israel.DayOfYear, df_israel.Temp, .75)
    degree = np.arange(1, 11)
    loss = np.zeros(10)
    for k in range(10):
        pf = PolynomialFitting(degree[k])
        pf.fit(train_X.to_numpy(), train_y.to_numpy())
        loss[k] = round(pf.loss(test_X.to_numpy(), test_y.to_numpy()), 2)
        print("k: " + str(degree[k]) + ", test error: " + str(loss[k]))
    test_error_israel = pd.DataFrame({'k': degree, 'loss': loss})
    fig = px.bar(test_error_israel, x="k", y="loss",
                 title=f"Test Error For Each Polynomial Fitting Degree (k)"
                 , labels={'loss': 'test error'}, text='loss', width=1000, height=800)
    fig.update_layout(font=dict(size=18))
    fig.write_image(address_of_output + "Israel-poly-fitting-test-error-bar.png",
                    format="png")


    # Question 5 - Evaluating fitted model on different countries
    chosen_k = 5
    pf = PolynomialFitting(chosen_k)
    pf.fit(df_israel.DayOfYear.to_numpy(), df_israel.Temp.to_numpy())
    countries = df[df["Country"] != "Israel"]
    countries = countries.Country.drop_duplicates().to_numpy()
    loss = np.zeros(len(countries))
    for i in range(len(countries)):
        X = df[df["Country"] == countries[i]]
        y = X.Temp
        X = X.DayOfYear
        loss[i] = round(pf.loss(X.to_numpy(), y.to_numpy()), 2)
    test_error_countries = pd.DataFrame({'Country': countries, 'loss': loss})
    fig = px.bar(test_error_countries, x="Country", y="loss",
                 title=f"Test Error On Countries Datasets On The Model Of Israel"
                 , labels={'loss': 'test error'}, text='loss', width=1000, height=800)
    fig.update_layout(font=dict(size=18))
    fig.write_image(address_of_output + "countries-test-error-bar-on-israel-fit-model.png",
                    format="png")
