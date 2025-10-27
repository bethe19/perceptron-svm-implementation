import numpy as np
import pandas as pd

data = pd.read_csv("hard_dataset.csv")
x = data.drop(columns='y').values
y = data['y'].values

learning_rate = 1
max_epoch = 500
num_features = x.shape[1]

w = np.zeros(num_features)
b = 0
best_weights = np.copy(w)
best_bias = b
max_accuracy = 0
e = 0

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
        e = epoch
    if accuracy == 100:
        print("All points classified correctly. Stopping early.")
        break

print("\nBest weights:", best_weights)
print("Best bias:", best_bias)
boundary_eq = " + ".join([f"{best_weights[i]}*x{i+1}" for i in range(len(best_weights))])
print(f"Decision boundary: {boundary_eq} + {best_bias} = 0")
print(f"max accuracy of among all the epoches at epoch {e} is :{max_accuracy}")

# visualization
import matplotlib.pyplot as plt

class0 = x[y == 0]
class1 = x[y == 1]

plt.figure(figsize=(8, 6))
plt.scatter(class0[:, 0], class0[:, 1], color='red', label='Class 0')
plt.scatter(class1[:, 0], class1[:, 1], color='blue', label='Class 1')

x1_vals = np.linspace(np.min(x[:, 0]), np.max(x[:, 0]), 100)
x2_vals = (-best_weights[0] * x1_vals - best_bias) / best_weights[1]
plt.plot(x1_vals, x2_vals, color='green', linewidth=2, label='Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title(f'Perceptron Decision Boundary (Max Accuracy: {max_accuracy:.2f}% at Epoch {e})')
plt.legend()
plt.grid(True)
plt.show()
