import numpy as np

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, max_epoch=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.max_epoch = max_epoch
        self.w = None
        self.b = 0.0
        self.best_weights = None
        self.best_bias = 0.0
        self.max_accuracy = 0.0
        self.best_epoch = 0
        self.support_vectors_ = None

    def fit(self, X, y, initial_w=None, initial_b=None, verbose=True):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)

        if initial_w is not None:
            self.w = np.copy(initial_w)
            if verbose: 
                print("Using Perceptron weights as initialization (warm-start)")
        else:
            self.w = np.zeros(n_features)
        
        if initial_b is not None:
            self.b = initial_b
        else:
            self.b = 0.0

        for epoch in range(1, self.max_epoch + 1):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) + self.b) >= 1

                if condition:
                    self.w -= self.lr * (self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (self.lambda_param * self.w - y_[idx] * x_i)
                    self.b -= self.lr * y_[idx]

            accuracy = self.score(X, y)
            if verbose and epoch % 50 == 0:
                print(f"Epoch {epoch}, Accuracy: {accuracy:.2f}%")

            if accuracy > self.max_accuracy:
                self.max_accuracy = accuracy
                self.best_epoch = epoch
                self.best_weights = np.copy(self.w)
                self.best_bias = self.b

            if accuracy == 100:
                if verbose:
                    print(f"Reached 100% accuracy at epoch {epoch}.")
                break

        if self.best_weights is not None:
            self.w = self.best_weights
            self.b = self.best_bias

        distances = y_ * (np.dot(X, self.w) + self.b)
        tolerance = 1e-2
        self.support_vectors_ = X[np.abs(distances - 1) <= tolerance]

        return self

    def predict(self, X):
        if self.w is None:
            raise ValueError("Model not trained. Call fit() first.")
        return (np.dot(X, self.w) + self.b >= 0).astype(int)

    def decision_function(self, X):
        return np.dot(X, self.w) + self.b

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y) * 100

    def get_params(self):
        return {
            'weights': self.w,
            'bias': self.b,
            'accuracy': self.max_accuracy,
            'epoch': self.best_epoch,
            'support_vectors': self.support_vectors_
        }
