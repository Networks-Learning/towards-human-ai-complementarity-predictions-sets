import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import argparse

convert = {
    1.0: 0.9,
    0.7: 0.7,
    0.5: 0.5,
    0.3: 0.3
}

convert_pi = {
    1.0: 0.9,
    0.7778: 0.7,
    0.5556: 0.5,
    0.3334: 0.3
}

def command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", nargs="*", type=str, default="1000-10-100-False-False-prediction_sets.pickle")
    parser.add_argument("gt", nargs="*",
                        default="1000-10-100-False-False-ground_truth.pickle")
    parser.add_argument("--save", action="store_true", default=False)
    parser.add_argument("--difficulty", default=-1, type=int)
    parser.add_argument("--expressive-humans", default=-1, type=int)
    parser.add_argument("--expressive-mnl", help="Use a more expressive MNL based on difficulties", action="store_true", default=False)
    parser.add_argument("--use-g-score", help="Use the score given the by the optimization objective, rather than the empirical success probability.", action="store_true", default=False)
    return parser

def plot_calibration(y_true, y_pred):

    for y_gt in range(10):
        Y_test_cal = [1 if y==y_gt else 0 for y in y_true]
        predicted_probs = [x[y_gt] for x in y_pred.T]
        prob_true, prob_pred = calibration_curve(Y_test_cal, predicted_probs, n_bins=20)
        plt.plot(prob_pred, prob_true, marker='o', label=f'{y_gt}')

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve')
    plt.legend()
    plt.show()

def plot_confusion_matrix(user_cm, labels,
                          title="Confusion Matrix",
                          output_title="confusion_matrix_differences.png",
                          save=False,
                          ax_default = None):
    fig, ax = plt.subplots(figsize=(15,10)) if ax_default is None else None, ax_default
    sns.heatmap(user_cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax
                )
    ax.set_ylabel("Predicted Labels")
    ax.set_xlabel("True Labels")#, fontsize=20)
    if title is not None:
        ax.set_title(title)
    
    if save:
        plt.tight_layout()
        plt.savefig(output_title, dpi=400)