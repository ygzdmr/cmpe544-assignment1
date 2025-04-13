import numpy as np

class NaiveBayes:
    """
    Gaussian Naive Bayes classifier implemented from scratch.

    Assumes features follow a Gaussian distribution within each class.
    """

    def __init__(self):
        """Initializes classifier attributes."""
        self._classes = None
        self._priors = {} # Dictionary to store prior probability P(C) for each class
        self._means = {}  # Dictionary to store mean of each feature for each class
        self._vars = {}   # Dictionary to store variance of each feature for each class
        # Small value to add to variance for numerical stability (avoids division by zero)
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
            X_c = X[y == c] # Get all samples belonging to class c

            # --- Calculate Prior P(C) ---
            # Probability of this class occurring in the dataset
            self._priors[c] = len(X_c) / n_samples

            # --- Calculate Mean and Variance for each feature in this class ---
            # Mean for each feature (column) for samples in class c
            self._means[c] = X_c.mean(axis=0)
            # Variance for each feature (column) for samples in class c
            self._vars[c] = X_c.var(axis=0)

        print("Training complete.")
        print(f"  Classes found: {self._classes}")
        print(f"  Priors calculated: {self._priors}")
        # print(f"  Means calculated: {self._means}") # Uncomment to see means
        # print(f"  Variances calculated: {self._vars}") # Uncomment to see variances


    def _gaussian_pdf(self, x, mean, var):
        """
        Calculate the Probability Density Function (PDF) of the Gaussian distribution.

        Args:
            x (float or np.ndarray): Value(s) for which to calculate the PDF.
            mean (float or np.ndarray): Mean(s) of the distribution.
            var (float or np.ndarray): Variance(s) of the distribution.

        Returns:
            float or np.ndarray: The PDF value(s).
        """
        # Add epsilon for numerical stability in case variance is near zero
        var_stable = var + self._epsilon

        numerator = np.exp(-((x - mean)**2) / (2 * var_stable))
        denominator = np.sqrt(2 * np.pi * var_stable)
        return numerator / denominator

    def _predict_log_proba(self, x):
        """
        Calculate the log posterior probability for a single sample x for each class.

        Uses log probabilities for numerical stability.
        log(P(C|x)) ∝ log(P(C)) + Σ log(P(xi|C))

        Args:
            x (np.ndarray): A single sample's features (n_features,).

        Returns:
            dict: A dictionary where keys are class labels and values are
                  proportional log posterior probabilities.
        """
        log_posteriors = {}

        for c in self._classes:
            prior_c = self._priors[c]
            mean_c = self._means[c]
            var_c = self._vars[c]

            # Calculate log prior
            log_prior_c = np.log(prior_c)

            # Calculate log likelihood for each feature and sum them
            # P(x | C) = P(x1|C) * P(x2|C) * ... * P(xn|C)
            # log(P(x|C)) = log(P(x1|C)) + log(P(x2|C)) + ... + log(P(xn|C))
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
            # Calculate log posteriors for the current sample for all classes
            log_posteriors = self._predict_log_proba(x_sample)

            # Choose the class with the highest log posterior probability
            best_class = max(log_posteriors, key=log_posteriors.get)
            y_pred.append(best_class)

        return np.array(y_pred)