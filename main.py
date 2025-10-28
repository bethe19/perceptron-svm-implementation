from src.data_loader import load_data
from src.perceptron import train_perceptron
from src.svm import train_svm
from src.utils import plot_decision_boundary, plot_svm_boundary

def perceptron():
    x_scaled, y, x_real = load_data("data/wdbc_copy.csv", target_column='Diagnosis', drop_columns=['ID'], positive_label='M', negative_label='B')
    w, b, acc, epoch = train_perceptron(x_scaled, y, max_epoch=50)

    print("\nResults:")
    print("Best weights:", w)
    print("Best bias:", b)
    print(f"Max accuracy: {acc:.2f}% at epoch {epoch}")

    boundary_eq = " + ".join([f"{w[i]:.4f}*x{i+1}" for i in range(len(w))])
    print(f"Decision boundary: {boundary_eq} + {b:.4f} = 0")

    print("\n Visualizing...")
    plot_decision_boundary(x_scaled[:, :2], y, w[:2], b, acc, epoch, save_path="imgs/perceptron_wdbc.png")
def svm(reg_param=0.01, max_epoch=50, save_plot=None):
    x_scaled, y, x_real = load_data("data/wdbc_copy.csv", target_column='Diagnosis', 
                                    drop_columns=['ID'], positive_label='M', negative_label='B')
    
    print(f"Dataset: {len(y)} samples, {x_scaled.shape[1]} features")
    print(f"Hyperparams: reg_param={reg_param}, max_epoch={max_epoch}")
  
    
    print("Training SVM...")
    w, b, acc, epoch = train_svm(x_scaled, y, max_epoch=max_epoch, reg_param=reg_param)

    print("\n" + "="*60)
    print("SVM Results (WDBC Breast Cancer)")
    print("="*60)
    print(f"{'Accuracy':<12} {acc:>7.2f}%")
    print(f"{'Epoch':<12} {epoch:>7}")
    print(f"{'Status':<12} {'Perfect!' if acc >= 98 else 'Excellent!'}")
    print("="*60)
    
    print("\nFull decision boundary:")
    top5 = [f"{w[i]:+.4f}x{i+1}" for i in range(5)]
    print("   " + " + ".join(top5) + f" + ... + {b:+.4f} = 0")
    
    print("\n Visualizing (Features 1-2, Scaled)...")
    plot_svm_boundary(x_scaled[:, :2], y, w[:2], b, acc, epoch, save_path=save_plot)

    return w, b, acc, epoch
def main():
    perceptron()
    w_svm, b_svm, acc_svm, epoch_svm = svm(reg_param=0.01, save_plot="imgs/svm_wdbc.png")

if __name__ == "__main__":
    main()
