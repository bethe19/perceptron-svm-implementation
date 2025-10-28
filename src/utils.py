import matplotlib.pyplot as plt
import numpy as np

def compute_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred) * 100

def plot_decision_boundary(x_real, y, w, b, acc, epoch, save_path=None):
    class0 = x_real[y == 0]
    class1 = x_real[y == 1]

    plt.figure(figsize=(8, 6))
    plt.scatter(class0[:, 0], class0[:, 1], color='red', label='Negative-result')
    plt.scatter(class1[:, 0], class1[:, 1], color='blue', label='Positive-result')

    # only first two features used for visualization
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


def plot_svm_boundary(x, y, w, b, acc, epoch, save_path=None):

    class0 = x[y == 0]
    class1 = x[y == 1]

    plt.figure(figsize=(10, 7))
    plt.scatter(class0[:, 0], class0[:, 1], color='red', s=60, label='Negative-result', alpha=0.7, edgecolors='white', linewidth=0.5)
    plt.scatter(class1[:, 0], class1[:, 1], color='blue', s=60, label='Positive-result', alpha=0.7, edgecolors='white', linewidth=0.5)

    padding = 0.1
    x_min, x_max = x[:, 0].min(), x[:, 0].max()
    y_min, y_max = x[:, 1].min(), x[:, 1].max()
    x_min -= padding * (x_max - x_min)
    x_max += padding * (x_max - x_min)
    y_min -= padding * (y_max - y_min)
    y_max += padding * (y_max - y_min)
    
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )
    
    zz = w[0] * xx + w[1] * yy + b
    
    plt.contour(xx, yy, zz, 
                levels=[-1.0, 0.0, 1.0], 
                colors=['#FF9500', '#00FF00', '#8E44AD'],  # Orange, Green, Purple
                linewidths=[2, 4, 2],
                linestyles=['--', '-', '--'])

    plt.plot([], [], color='#FF9500', ls='--', lw=2, label='Support Margin (-1)')
    plt.plot([], [], color='#00FF00', ls='-', lw=4, label='Decision Boundary (0)')
    plt.plot([], [], color='#8E44AD', ls='--', lw=2, label='Support Margin (+1)')

    plt.xlabel('Feature 1 (Scaled)', fontsize=12, fontweight='bold')
    plt.ylabel('Feature 2 (Scaled)', fontsize=12, fontweight='bold')
    plt.title(f'SVM Decision Boundary **+ Margins**\n(Acc: {acc:.2f}% @ Epoch {epoch})', 
              fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()