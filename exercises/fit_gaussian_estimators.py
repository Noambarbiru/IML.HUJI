from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mu = 10
    var = 1
    n = 1000
    samples_univariate_gaussian = np.random.normal(mu, np.sqrt(var), n)
    univariate_gaussian = UnivariateGaussian()
    univariate_gaussian.fit(samples_univariate_gaussian)
    print("(%f,%f)" % (univariate_gaussian.mu_, univariate_gaussian.var_))


    # Question 2 - Empirically showing sample mean is consistent
    sample_size_arr = np.arange(10, 1010, 10)
    estimated_expectation = np.zeros(len(sample_size_arr))
    for index in range(len(sample_size_arr)):
        univariate_gaussian.fit(samples_univariate_gaussian[0:sample_size_arr[index]])
        estimated_expectation[index] = univariate_gaussian.mu_
    estimated_expectation = abs(estimated_expectation - mu)
    go.Figure([go.Scatter(x=sample_size_arr, y=estimated_expectation, mode='markers+lines')],
              layout=go.Layout(title=r"$\text{Absolute distance between the estimated and true value of the "
                                     r"expectation as a function of the sample size for samples drawn from N(10,1)}$",
                               xaxis_title="$m\\text{ - number of samples}$",
                               yaxis_title="r$\\text{|}\\hat\mu\\text{ - }\\mu\\text{|}$",
                               height=300)).show()



    # Question 3 - Plotting Empirical PDF of fitted model
    univariate_gaussian.fit(samples_univariate_gaussian)
    pdf_of_samples = univariate_gaussian.pdf(samples_univariate_gaussian)
    go.Figure([go.Scatter(x=samples_univariate_gaussian, y=pdf_of_samples, mode='markers')],
              layout=go.Layout(title=r"$\text{The PDF of samples drawn from N(10,1) under the fitted model}$",
                               xaxis_title="$\\text{sample values}$",
                               yaxis_title="r$\\text{sample values PDFs}$",
                               height=300)).show()

def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mean = [0, 0, 4, 0]
    cov = [[1, 0.2, 0, 0.5],
           [0.2, 2, 0, 0],
           [0, 0, 1, 0],
           [0.5, 0, 0, 1]]
    n = 1000
    samples_multivariate_gaussian = np.random.multivariate_normal(mean, cov, n)
    multivariate_gaussian = MultivariateGaussian()
    multivariate_gaussian.fit(samples_multivariate_gaussian)
    print(multivariate_gaussian.mu_)
    print(multivariate_gaussian.cov_)


    # Question 5 - Likelihood evaluation
    size = 200
    val = np.linspace(-10, 10, size)
    likelihood = np.zeros([size, size])

    for i in range(size):
        for j in range(size):
            likelihood[i, j] = MultivariateGaussian.log_likelihood([val[i], 0, val[j], 0], cov,
                                                                   samples_multivariate_gaussian)

    go.Figure(go.Heatmap(x=val, y=val, z=likelihood), layout=go.Layout(title=r"$\text{log-likelihood"
                                                                             r" of samples with expectation "
                                                                             r"µ = [f1,0,f3,0] pattern "
                                                                             r"and the true cov matrix when the"
                                                                             r" true expectation is [0,0,4,0]}$",
                                                                       xaxis_title="$\\text{f3 values}$",
                                                                       yaxis_title="r$\\text{f1 values}$",
                                                                       height=550, width=1200)).show()

    # Question 6 - Maximum likelihood
    ind = np.unravel_index(np.argmax(likelihood), likelihood.shape)
    f1_max_likelihood = val[ind[0]]
    f3_max_likelihood = val[ind[1]]
    print("f1 = %.3f, f3 = %.3f" % (f1_max_likelihood, f3_max_likelihood))


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
