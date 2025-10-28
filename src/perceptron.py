import numpy as np
def train_perceptron(x, y, learning_rate=1, max_epoch=500):
    num_features = x.shape[1]
    w = np.zeros(num_features)
    b = 0
    best_weights = np.copy(w)
    best_bias = b
    max_accuracy = 0
    best_epoch = 0

    for epoch in range(1, max_epoch + 1):
        missclassfied_count = 0
        for xi, yi in zip(x, y):
            z = np.dot(w, xi) + b
            y_pred = 1 if z >= 0 else 0
            if y_pred != yi:
                w += learning_rate * (yi - y_pred) * xi
                b += learning_rate * (yi - y_pred)
                missclassfied_count += 1
        accuracy = (len(x) - missclassfied_count) / len(x) * 100
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