import numpy as np

def train_svm(x, y, learning_rate=1.0, reg_param=0.01, max_epoch=1000, 
              init_w=None, init_b=None):

    num_features = x.shape[1]
    w = np.zeros(num_features) if init_w is None else np.copy(init_w)
    b = 0.0 if init_b is None else init_b
    best_weights = np.copy(w)
    best_bias = b
    max_accuracy = 0.0
    best_epoch = 0
    
    y_svm = 2 * y - 1  # -1/+1 labels
    
    t = 0
    
    for epoch in range(1, max_epoch + 1):
        for xi, yi_svm in zip(x, y_svm):
            t += 1
            eta = learning_rate / (reg_param * t)
            
            condition = yi_svm * (np.dot(w, xi) + b)
            
            if condition < 1:
                w = (1 - eta * reg_param) * w + eta * yi_svm * xi
                b += eta * yi_svm
            else:
                w = (1 - eta * reg_param) * w
        
        z = np.dot(x, w) + b
        y_pred = 1 * (z >= 0)  # 0 or 1
        accuracy = np.mean(y_pred == y) * 100
        
        print(f"Epoch {epoch}, Accuracy: {accuracy:.2f}%")
        
        if accuracy > max_accuracy:
            best_weights = np.copy(w)
            best_bias = b
            max_accuracy = accuracy
            best_epoch = epoch
        
        if accuracy == 100:
            print("All points classified correctly. Stopping early.")
            break
    
    return best_weights, best_bias, max_accuracy, best_epoch