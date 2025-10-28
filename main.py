from src.data_loader import DataLoader
from src.perceptron import Perceptron
from src.svm import SVM
from src.utils import plot_decision_boundary, plot_svm_boundary
from sklearn.model_selection import train_test_split


def perceptron():
    data_loader = DataLoader(
        file_path="data/data.csv",
        target_column='Result',
        positive_label='Pass',
        negative_label='Fail'
    )
    x_scaled, y, x_real = data_loader.load()
    
    X_train, X_test, y_train, y_test = train_test_split(
        x_scaled, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {len(y_train)} samples")
    print(f"Test set: {len(y_test)} samples")
    
    model = Perceptron(learning_rate=1.0, max_epoch=50)
    model.fit(X_train, y_train)
    
    params = model.get_params()
    w = params['weights']
    b = params['bias']
    train_acc = params['accuracy']
    epoch = params['epoch']
    
    test_acc = model.score(X_test, y_test)
    
    print("\n" + "="*60)
    print("PERCEPTRON RESULTS")
    print("="*60)
    print(f"Training Accuracy:   {train_acc:.2f}%")
    print(f"Test Accuracy:       {test_acc:.2f}%")
    print(f"Best Epoch:          {epoch}")
    print("="*60)
    
    print("\nBest weights:", w)
    print("Best bias:", b)
    
    boundary_eq = " + ".join([f"{w[i]:.4f}*x{i+1}" for i in range(len(w))])
    print(f"Decision boundary: {boundary_eq} + {b:.4f} = 0")
    
    print("\nVisualizing...")
    plot_decision_boundary(
        x_scaled[:, :2], y, w[:2], b, test_acc, epoch,
        save_path="imgs/perceptron_diabetes.png"
    )
    
    return train_acc, test_acc


def svm(reg_param=0.01, max_epoch=50, save_plot=None):
    data_loader = DataLoader(
        file_path="data/data.csv",
        target_column='Result',
        positive_label='Pass',
        negative_label='Fail'
    )
    x_scaled, y, x_real = data_loader.load()
    
    X_train, X_test, y_train, y_test = train_test_split(
        x_scaled, y, test_size=0.2, random_state=42
    )
    
    print(f"Dataset: {len(y)} samples, {x_scaled.shape[1]} features")
    print(f"Training set: {len(y_train)} samples")
    print(f"Test set: {len(y_test)} samples")
    print(f"Hyperparams: reg_param={reg_param}, max_epoch={max_epoch}")
    
    print("\nTraining SVM...")
    model = SVM(learning_rate=1.0, reg_param=reg_param, max_epoch=max_epoch)
    model.fit(X_train, y_train, verbose=True)
    
    params = model.get_params()
    w = params['weights']
    b = params['bias']
    train_acc = params['accuracy']
    epoch = params['epoch']
    
    test_acc = model.score(X_test, y_test)
    
    print("\n" + "="*60)
    print("SVM RESULTS (Diabetes Dataset)")
    print("="*60)
    print(f"Training Accuracy:   {train_acc:.2f}%")
    print(f"Test Accuracy:       {test_acc:.2f}%")
    print(f"Best Epoch:          {epoch}")
    print(f"Status:              {'Perfect!' if test_acc >= 98 else 'Excellent!' if test_acc >= 90 else 'Good'}")
    print("="*60)
    
    print("\nFull decision boundary:")
    top5 = [f"{w[i]:+.4f}x{i+1}" for i in range(min(5, len(w)))]
    print("   " + " + ".join(top5) + f" + ... + {b:+.4f} = 0")
    
    print("\nVisualizing (Features 1-2, Scaled)...")
    plot_svm_boundary(
        x_scaled[:, :2], y, w[:2], b, test_acc, epoch,
        save_path=save_plot
    )
    
    return w, b, train_acc, test_acc


def main():
    print("="*60)
    print("PERCEPTRON CLASSIFICATION")
    print("="*60)
    perc_train_acc, perc_test_acc = perceptron()
    
    print("\n\n" + "="*60)
    print("SVM CLASSIFICATION")
    print("="*60)
    w_svm, b_svm, svm_train_acc, svm_test_acc = svm(
        reg_param=0.01,
        max_epoch=100,
        save_plot="imgs/svm_diabetes.png"
    )
    
    print("\n\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Model':<15} {'Train Acc':<12} {'Test Acc':<12}")
    print("-"*60)
    print(f"{'Perceptron':<15} {perc_train_acc:>10.2f}% {perc_test_acc:>10.2f}%")
    print(f"{'SVM':<15} {svm_train_acc:>10.2f}% {svm_test_acc:>10.2f}%")
    print("="*60)


if __name__ == "__main__":
    main()
