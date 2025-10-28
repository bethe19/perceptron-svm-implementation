import numpy as np
class SVM:
    def __init__(self, learning_rate=1.0, reg_param=0.01, max_epoch=1000):
        self.learning_rate = learning_rate
        self.reg_param = reg_param
        self.max_epoch = max_epoch
        self.w = None
        self.b = 0.0
        self.best_weights = None
        self.best_bias = 0.0
        self.max_accuracy = 0.0
        self.best_epoch = 0
        self._t = 0
  
    def fit(self, X, y, init_w=None, init_b=None, verbose=True):
        num_features = X.shape[1]
        self.w = np.zeros(num_features) if init_w is None else np.copy(init_w)
        self.b = 0.0 if init_b is None else init_b
        self.best_weights = np.copy(self.w)
        self.best_bias = self.b
        self.max_accuracy = 0.0
        self.best_epoch = 0
        
        y_svm = 2 * y - 1
        
        self._t = 0
        
        for epoch in range(1, self.max_epoch + 1):
            for xi, yi_svm in zip(X, y_svm):
                self._t += 1
                eta = self.learning_rate / (self.reg_param * self._t)
                
                condition = yi_svm * (np.dot(self.w, xi) + self.b)
                
                if condition < 1:
                    self.w = (1 - eta * self.reg_param) * self.w + eta * yi_svm * xi
                    self.b += eta * yi_svm
                else:
                    self.w = (1 - eta * self.reg_param) * self.w
            
            accuracy = self.score(X, y)
            
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
    
    def decision_function(self, X):
        if self.w is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        return np.dot(X, self.w) + self.b
    
    def get_params(self):
        return {
            'weights': self.w,
            'bias': self.b,
            'accuracy': self.max_accuracy,
            'epoch': self.best_epoch
        }