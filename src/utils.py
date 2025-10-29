import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
def compute_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred) * 100

def plot_decision_boundary(x_real, y, w, b, acc, epoch, save_path=None):
    class0 = x_real[y == 0]
    class1 = x_real[y == 1]

    plt.figure(figsize=(8, 6))
    plt.scatter(class0[:, 0], class0[:, 1], color='red', label='Negative-result')
    plt.scatter(class1[:, 0], class1[:, 1], color='blue', label='Positive-result')

    x1 = np.linspace(np.min(x_real[:, 0]), np.max(x_real[:, 0]), 100)
    x2 = (-w[0] * x1 - b) / w[1]
    plt.plot(x1, x2, color='green', linewidth=2, label='Decision Boundary')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'Perceptron Decision Boundary (Acc: {acc:.2f}% @ Epoch {epoch})')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_svm_boundary(X, y, w, b, accuracy, epoch, support_vectors=None, save_path=None):
    class0 = X[y == 0]
    class1 = X[y == 1]
    
    plt.figure(figsize=(8, 6))
    plt.scatter(class0[:, 0], class0[:, 1], color='red', label='Negative-result')
    plt.scatter(class1[:, 0], class1[:, 1], color='blue', label='Positive-result')
    
    x1 = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
    x2_decision = (-w[0] * x1 - b) / w[1]
    x2_margin_neg = (-w[0] * x1 - b - 1) / w[1]
    x2_margin_pos = (-w[0] * x1 - b + 1) / w[1]
    
    plt.plot(x1, x2_decision, color='green', linewidth=2, label='Decision Boundary')
    plt.plot(x1, x2_margin_neg, color='orange', linestyle='--', linewidth=2, label='Margin (-1)')
    plt.plot(x1, x2_margin_pos, color='purple', linestyle='--', linewidth=2, label='Margin (+1)')
    
    if support_vectors is not None and len(support_vectors) > 0:
        plt.scatter(support_vectors[:, 0], support_vectors[:, 1], 
                   s=200, facecolors='none', edgecolors='black', 
                   linewidths=3, label=f'Support Vectors ({len(support_vectors)})')
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'SVM Decision Boundary (Acc: {accuracy:.2f}% @ Epoch {epoch})')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()