import numpy as np

class NaiveBayes:
    """
    Gaussian Naive Bayes classifier implemented from scratch.

    Assumes features follow a Gaussian distribution within each class.
    """

    def __init__(self):
        """Initializes classifier attributes."""
        self._classes = None
        self._priors = {} 
        self._means = {}  
        self._vars = {}   
        # Small value to add to variance for numerical stability
        self._epsilon = 1e-9 

    def fit(self, X, y):
        """
        Train the Naive Bayes classifier.

        Calculates priors, means, and variances for each class based on the
        training data.

        Args:
            X (np.ndarray): Training data features (n_samples, n_features).
            y (np.ndarray): Training data labels (n_samples,).
        """
        n_samples, n_features = X.shape
        self._classes = np.unique(y) # Get unique class labels

        for idx, c in enumerate(self._classes):
            # Filter data for the current class
            X_c = X[y == c]

            # Calculate Prior P(C)
            self._priors[c] = len(X_c) / n_samples

            # Calculate Mean and Variance for each feature in this class
            self._means[c] = X_c.mean(axis=0)
            # Variance for each feature
            self._vars[c] = X_c.var(axis=0)

        print("Training complete.")
        print(f"  Classes found: {self._classes}")
        print(f"  Priors calculated: {self._priors}")
        # print(f"  Means calculated: {self._means}")
        # print(f"  Variances calculated: {self._vars}")


    def _gaussian_pdf(self, x, mean, var):
        """
        Calculate the Probability Density Function (PDF) of the Gaussian distribution.
        """
        # Add epsilon for numerical stability in case variance is near zero
        var_stable = var + self._epsilon

        numerator = np.exp(-((x - mean)**2) / (2 * var_stable))
        denominator = np.sqrt(2 * np.pi * var_stable)
        return numerator / denominator

    def _predict_log_proba(self, x):
        """
        Calculate the log posterior probability for a single sample x for each class.
        """
        log_posteriors = {}

        for c in self._classes:
            prior_c = self._priors[c]
            mean_c = self._means[c]
            var_c = self._vars[c]

            # Calculate log prior
            log_prior_c = np.log(prior_c)

            # Calculate log likelihood for each feature and sum them
            likelihoods_c = self._gaussian_pdf(x, mean_c, var_c)

            # Add epsilon before log in case PDF is zero
            log_likelihood_c = np.sum(np.log(likelihoods_c + self._epsilon)) 

            # Calculate log posterior (proportional)
            log_posteriors[c] = log_prior_c + log_likelihood_c

        return log_posteriors


    def predict(self, X):
        """
        Predict class labels for new data samples.

        Args:
            X (np.ndarray): Data samples to predict (n_samples, n_features).

        Returns:
            np.ndarray: Predicted class labels for each sample (n_samples,).
        """
        if self._classes is None:
            raise RuntimeError("You must train the classifier before predicting!")

        y_pred = []
        for x_sample in X:
            # Calculate log posteriors for the current sample
            log_posteriors = self._predict_log_proba(x_sample)

            # Choose the class with the highest log posterior probability
            best_class = max(log_posteriors, key=log_posteriors.get)
            y_pred.append(best_class)

        return np.array(y_pred)