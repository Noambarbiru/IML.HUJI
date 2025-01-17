from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_ = np.unique(y)
        self.mu_ = []
        self.pi_ = np.zeros(len(self.classes_))
        self.cov_ = np.zeros((len(X[0]), len(X[0])))
        for c in range(len(self.classes_)):
            x_index = []
            sum_of_x_k = np.zeros((len(X[0])))
            n_k = 0
            for x in range(len(X)):
                if y[x] == self.classes_[c]:
                    x_index.append(x)
                    sum_of_x_k += X[x]
                    n_k += 1
            self.pi_[c] = n_k / len(y)
            mu = sum_of_x_k / n_k
            self.mu_.append(mu)
            for x in x_index:
                val = np.reshape(X[x] - mu, (1, -1))
                self.cov_ += np.dot(val.T, val)
        self.cov_ /= len(y)
        self._cov_inv = inv(self.cov_)
        self.mu_ = np.asarray(self.mu_)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        a = []
        b = np.log(self.pi_)
        for i in range(len(self.classes_)):
            dot_product = np.dot(self._cov_inv, self.mu_[i])
            a.append(dot_product)
            dot = np.dot(self.mu_[i], dot_product)
            b[i] -= 0.5 * dot
        a = np.asarray(a)
        y = np.zeros(len(X))
        for i in range(len(X)):
            arr_of_all_k = np.dot(a, X[i]) + b
            class_index = np.argmax(arr_of_all_k)
            y[i] = self.classes_[class_index]
        return y

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        factor = 1 / np.sqrt(det(self.cov_) * (2 * np.pi) ** len(X[0]))
        likelihood = np.zeros((len(X), len(self.classes_)))
        for i in range(len(self.classes_)):
            c = self.classes_[i]
            mu_c = self.mu_[c]
            pi_c = self.pi_[c]
            for j in range(len(X)):
                x = X[j]
                x_minus_mu = np.reshape(x - mu_c, (1, -1))
                exp_input = -0.5 * np.dot(x_minus_mu, np.dot(self._cov_inv, x_minus_mu.T))
                likelihood[j][i] = factor * np.exp(exp_input) * pi_c
        return likelihood

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        return misclassification_error(y, self._predict(X))
