import numpy as np

class MultinomialLogisticRegression:
    """
    Multinomial logistic regression.
    """
    def __init__(self, lr=0.05, n_iter=3000, l2=0.0, verbose=False):
        self.lr = lr
        self.n_iter = n_iter
        self.l2 = l2          # L2 regularisation strength (λ)
        self.verbose = verbose
        self.W = None         # Weight matrix
        self.b = None         # Bias vector
        self.loss_history = []

    # helpers 
    @staticmethod
    def _softmax(z):
        # z shape: (m, C)
        z -= z.max(axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / exp_z.sum(axis=1, keepdims=True)

    def _cross_entropy(self, y_true_onehot, y_pred):
        m = y_true_onehot.shape[0]
        ce = -np.sum(y_true_onehot * np.log(y_pred + 1e-9)) / m
        if self.l2 > 0:
            ce += (self.l2 / (2 * m)) * np.sum(self.W ** 2)
        return ce

    def fit(self, X, y):
        m, n = X.shape
        classes = np.unique(y)
        C = len(classes)

        # one‑hot encode labels
        Y = np.zeros((m, C))
        Y[np.arange(m), y] = 1

        # init params
        rng = np.random.default_rng(0)
        self.W = rng.normal(0, 0.01, size=(n, C))
        self.b = np.zeros((1, C))

        for i in range(self.n_iter):
            # forward
            scores = X @ self.W + self.b          # (m, C)
            probs  = self._softmax(scores)        # (m, C)

            # loss
            loss = self._cross_entropy(Y, probs)
            self.loss_history.append(loss)

            # gradient
            grad_scores = (probs - Y) / m         # (m, C)
            dW = X.T @ grad_scores + self.l2 * self.W / m
            db = grad_scores.sum(axis=0, keepdims=True)

            # update
            self.W -= self.lr * dW
            self.b -= self.lr * db

    def predict_proba(self, X):
        scores = X @ self.W + self.b
        return self._softmax(scores)              # (m, C)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def score(self, X, y):
        return (self.predict(X) == y).mean()