from typing import NoReturn
from ...base import BaseEstimator
import numpy as np


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """

    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

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
        self.vars_ = []
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
            var = np.zeros(len(X[0]))
            for x in x_index:
                var += (X[x] - mu) ** 2
            var /= n_k
            self.vars_.append(var)
        self.mu_ = np.asarray(self.mu_)
        self.vars_ = np.asarray(self.vars_)

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
        b = np.log(self.pi_)
        for i in range(len(self.classes_)):
            b[i] -= 0.5 * np.sum(np.log(self.vars_[i]))

        y = np.zeros(len(X))
        for i in range(len(X)):
            arr_of_all_k = np.array(b)
            for j in range(len(self.classes_)):
                arr_of_all_k[j] += -0.5 * np.sum((X[i] - self.mu_[j]) ** 2 / self.vars_[j])
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

        likelihood = np.zeros((len(X), len(self.classes_)))
        for i in range(len(self.classes_)):
            c = self.classes_[i]
            mu_c = self.mu_[c]
            var_c = self.vars_[c]
            pi_c = self.pi_[c]
            for j in range(len(X)):
                x = X[j]
                likelihood[j][i] = pi_c * np.sqrt(np.pi ** len(X[0]))
                for d in range(len(X[0])):
                    likelihood[j][i] *= np.exp((x[d] - mu_c[d]) ** 2 / (-2 * var_c[d])) / np.sqrt(var_c[d])
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
