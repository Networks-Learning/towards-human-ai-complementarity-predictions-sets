import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

from sklearn.utils.random import sample_without_replacement

def generate_hypercube(samples, dimensions, rng):
    """Returns distinct binary samples of length dimensions."""
    if dimensions > 30:
        return np.hstack(
            [
                rng.randint(2, size=(samples, dimensions - 30)),
                generate_hypercube(samples, 30, rng),
            ]
        )
    out = sample_without_replacement(2**dimensions, samples, random_state=rng).astype(
        dtype=">u4", copy=False
    )
    out = np.unpackbits(out.view(">u1")).reshape((-1, 32))[:, -dimensions:]
    return out

def _generate_problem(N, num_classes, num_features, class_sep=1, seed=2024):

    generator = np.random.RandomState(seed)

    # Generate the means of the distributions
    means = generate_hypercube(num_classes,
                               num_features,
                               generator).astype(float, copy=False)
    means *= 2 * class_sep
    means -= class_sep

    # Generate the covariances
    covariances = [
        np.identity(num_features) for _ in range(num_classes)
    ]

    # Generate X and Y together. We assume P(Y) = 1/|classes|
    X_Y = []
    for k in range(num_classes):
        xes = generator.multivariate_normal(
            means[k], covariances[k], size=N//num_classes
        )
        X_Y += [(x,k) for x in xes]

    # Shuffle the array
    generator.shuffle(X_Y)

    # Split in X and Y
    X = np.array([x for x,_ in X_Y])
    Y = np.array([y for _,y in X_Y])

    # Return everything
    return means, covariances, X, Y

def P_y_given_x(X, means, covs):

    P_x_given_Ys = []
    denom = 0
    for mean, cov in zip(means, covs):
        P_x_given_Y = multivariate_normal(mean, cov).pdf(X)
        P_Y = 1/len(means)
        denom += P_x_given_Y * P_Y
        P_x_given_Ys.append(P_x_given_Y * P_Y)

    assert np.all(np.array(P_x_given_Y) / denom >= 0)

    return np.array(P_x_given_Ys) / denom

def generate_problem(N, num_classes, num_features, class_sep=1, seed=2024):

    means, covs, X, Y = _generate_problem(N, num_classes, num_features, class_sep, seed)

    # For each X, generate the softmaxes
    X_augmented = []
    for x in X:
        X_augmented.append(P_y_given_x(x, means, covs))

    return np.array(X_augmented).T, Y, X

if __name__ == "__main__":

    for sep in [0.30, 0.5, 0.7, 1.0]:
        X, Y, X_features = generate_problem(1000, 10, 10, class_sep=sep, seed=2024)
        total = 0
        for x, y in zip(X.T, Y):
            total += int(np.argmax(x) == y)
        print(sep, round(total / len(X.T), 2))    

    exit()

    Y_test_cal = [1 if y==1 else 0 for y in Y]
    predicted_probs = [x[1] for x in X.T]
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(Y_test_cal, predicted_probs, n_bins=20)

    plt.plot(prob_pred, prob_true, marker='o', label=f'Label')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve')
    plt.legend()
    plt.show()