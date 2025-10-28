import numpy as np


class Perceptron:
    def __init__(self, learning_rate=1.0, max_epoch=500):
        self.learning_rate = learning_rate
        self.max_epoch = max_epoch
        self.w = None
        self.b = 0.0
        self.best_weights = None
        self.best_bias = 0.0
        self.max_accuracy = 0.0
        self.best_epoch = 0
    
    def fit(self, X, y, verbose=True):
        num_features = X.shape[1]
        self.w = np.zeros(num_features)
        self.b = 0.0
        self.best_weights = np.copy(self.w)
        self.best_bias = self.b
        self.max_accuracy = 0.0
        self.best_epoch = 0
        
        for epoch in range(1, self.max_epoch + 1):
            misclassified_count = 0
            for xi, yi in zip(X, y):
                z = np.dot(self.w, xi) + self.b
                y_pred = 1 if z >= 0 else 0
                
                if y_pred != yi:
                    self.w += self.learning_rate * (yi - y_pred) * xi
                    self.b += self.learning_rate * (yi - y_pred)
                    misclassified_count += 1
            
            accuracy = (len(X) - misclassified_count) / len(X) * 100
            
            if verbose:
                print(f"Epoch {epoch}, Accuracy: {accuracy:.2f}%")
            
            if accuracy > self.max_accuracy:
                self.best_weights = np.copy(self.w)
                self.best_bias = self.b
                self.max_accuracy = accuracy
                self.best_epoch = epoch
            
            if accuracy == 100:
                if verbose:
                    print("All points classified correctly. Stopping early.")
                break
        
        self.w = self.best_weights
        self.b = self.best_bias
        
        return self
    
    def predict(self, X):
        
        if self.w is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")

        z = np.dot(X, self.w) + self.b
        return (z >= 0).astype(int)
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y) * 100
    
    def get_params(self):
        return {
            'weights': self.w,
            'bias': self.b,
            'accuracy': self.max_accuracy,
            'epoch': self.best_epoch
        }