import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import warnings

from scipy.special import softmax

class identity_vector():
    def predict_proba(self, x):
        return [x]
    def predict(self, x):
        return np.argmax(x, axis=1)
    
def get_sorted_probabilities_and_labels(model_cm, indices):
    sorted_softmaxes = [sorted(model_cm[:, idx], reverse=True) for idx in indices]
    sorted_softmaxes_keys = [
        sorted([
        (label, softmax) for label, softmax in enumerate(model_cm[:, idx])
        ], key=lambda x: x[1], reverse=True) for idx in indices
    ]

    # Check if everything is performed correctly
    assert sorted_softmaxes[0] == [x[1] for x in sorted_softmaxes_keys[0]]
    assert len(sorted_softmaxes_keys[0]) == len(model_cm[:, 0])
    assert len(sorted_softmaxes) == len(sorted_softmaxes_keys)

    sorted_softmaxes_keys = [
        [label for label, _ in softmax_labels] for softmax_labels in sorted_softmaxes_keys
    ]

    assert len(sorted_softmaxes[0]) == len(model_cm[:, 0])
    assert len(sorted_softmaxes) == len(indices) 

    return sorted_softmaxes, sorted_softmaxes_keys

def class_wise_calibration(cal_idxes, model_cm, Y, n_bins=10):

    N = len(model_cm[:, 0]) # Number of labels

    calibrators = []

    # For each label, we fit a binary calibrator
    for label in range(N):
        probabs_label = np.array([model_cm[label, idx] for idx in cal_idxes])
        y_binarized_label = np.array([1 if Y[idx] == label else 0 for idx in cal_idxes])

        calibrators.append(
            HB_binary(n_bins=n_bins).fit(probabs_label, y_binarized_label)
        )

    return calibrators

def top_k_label_calibration(cal_idxes, model_cm, Y, points_per_bin=30, ranks=3):

    assert len(model_cm[0, :]) == len(Y) # Assert dimensions are correct
    N = len(np.unique(Y)) # Get the number of labels

    # Extract the sorted probabilities and the labels
    sorted_softmaxes, sorted_softmaxes_keys = get_sorted_probabilities_and_labels(
        model_cm, cal_idxes
    )

    # List of calibrators for each rank
    # The default is just the identity
    calibrators = {
        label : { rank: identity_vector() for rank in range(ranks)} for label in range(N)
    }

    # For each label, train a calibrator for each rank
    for label in range(N):
        # For each rank, train a calibrator
        for rank in range(ranks):

            # Find all elements which, at rank k, have the corresponding label
            filter_rank_label = [
               softmaxes_keys[rank] == label for softmaxes_keys in sorted_softmaxes_keys
            ]

            n_l = np.sum(filter_rank_label)
            bins_l = np.floor(n_l/points_per_bin).astype('int')
            
            probabs_per_rank = [
                sorted_softmaxes[idx][rank] for idx in range(len(cal_idxes)) if filter_rank_label[idx]
            ] # Probabilities for this rank

            y_binarized = [
                int(Y[idx] == label) for k, idx in enumerate(cal_idxes) if filter_rank_label[k]
            ]

            assert len(probabs_per_rank) == n_l
            assert len(y_binarized) == n_l

            if bins_l == 0:
                #print(f"Avoiding {label} for rank {rank} since we have no enough points.")
                calibrators[label][rank] = identity_vector()
            else:
                calibrators[label][rank] = HB_binary(n_bins=bins_l).fit(
                            np.array(probabs_per_rank),
                            np.array(y_binarized)
                        )
            
    
    return calibrators

def top_k_confidence_calibration(cal_idxes, model_cm, Y, n_bins=10, ranks=3):

    assert len(model_cm[0, :]) == len(Y) # Assert dimensions are correct

    # Extract the sorted probabilities and the labels
    sorted_softmaxes, sorted_softmaxes_keys = get_sorted_probabilities_and_labels(
        model_cm, cal_idxes
    )

    # List of calibrators for each rank
    calibrators = {}

    Y_cal = np.array(Y)[cal_idxes]

    # For each rank, train a calibrator
    for rank in range(ranks):
        
        probabs_per_rank = [
            sorted_softmaxes[idx][rank] for idx in range(len(cal_idxes))
        ] # Probabilities for this rank

        y_binarized = [
            int(Y_cal[idx] == sorted_softmaxes_keys[idx][rank]) for idx, _ in enumerate(cal_idxes)
        ] # How many times the element at this rank position is correct

        calibrators[rank] = HB_binary(n_bins=n_bins).fit(
                    np.array(probabs_per_rank),
                    np.array(y_binarized)
        )    
    
    return calibrators

def calibration_error(target_label, y_true, y_predicted, n_bins=10, return_bins=False):
    """Compute the calibration error for a given label.
    """

    # Binarize the probabilities for the given label
    N = len(y_true)
    y_true_binarized = np.array([1 if y == target_label else 0 for y in y_true])
    y_predicted_for_label = np.array([y_predicted[target_label, x] for x in range(N)])
    bins = np.linspace(0.0, 1.0 + 1e-8, n_bins + 1)    
    binids = np.digitize(y_predicted_for_label, bins) -1

    bin_sums = np.bincount(binids, weights=y_predicted_for_label, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true_binarized, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))

    nonzero = bin_total != 0
    prob_true = bin_true[nonzero] / bin_total[nonzero]
    prob_pred = bin_sums[nonzero] / bin_total[nonzero]

    if not return_bins:
        return np.sum(np.abs(prob_true - prob_pred) * (bin_total[nonzero] / len(y_true_binarized)))
    else:
        return prob_true, prob_pred

def ECE(y_true, y_probabs, n_bins=10):
    """ Compute multiclass ECE by averaging single ECEs for each class.
    It takes the frequency of each label in the test set, so evaluating on larger sets
    might improve the accuracy of this metric.
    """
    unique, counts = np.unique(y_true, return_counts=True)
    label_frequency = { k: v/len(y_true) for k, v in dict(zip(unique, counts)).items()}
    return np.sum(
        [calibration_error(y, y_true, y_probabs, n_bins) * label_frequency.get(y) for y in unique]
    )

def temperature_scaling(softmaxes, T=1):
    """Given the model probabilities, apply temperature scaling."""
    return np.array([softmax(np.log(p) / T) for p in softmaxes])

def top_calibrated(softmaxes, generator):
    new_softmaxes = []
    for p in softmaxes:
        p = p.tolist()
        idx = np.argmax(p)
        max_val = p[idx]
        p.pop(idx)
        generator.shuffle(p)
        p.insert(idx, max_val)
        new_softmaxes.append(p)
    return np.array(new_softmaxes)

# All the code below is taken from https://github.com/AIgen/df-posthoc-calibration

def get_uniform_mass_bins(probs, n_bins):
    assert(probs.size >= n_bins), "Fewer points than bins"
    
    probs_sorted = np.sort(probs)

    # split probabilities into groups of approx equal size
    groups = np.array_split(probs_sorted, n_bins)
    bin_edges = list()
    bin_upper_edges = list()

    for cur_group in range(n_bins-1):
        bin_upper_edges += [max(groups[cur_group])]
    bin_upper_edges += [np.inf]

    return np.array(bin_upper_edges)

def bin_points(scores, bin_edges):
    assert(bin_edges is not None), "Bins have not been defined"
    scores = scores.squeeze()
    assert(np.size(scores.shape) < 2), "scores should be a 1D vector or singleton"
    scores = np.reshape(scores, (scores.size, 1))
    bin_edges = np.reshape(bin_edges, (1, bin_edges.size))
    return np.sum(scores > bin_edges, axis=1)

def bin_points_uniform(x, n_bins):
    x = x.squeeze()
    bin_upper_edges = get_uniform_mass_bins(x, n_bins)
    return np.sum(x.reshape((-1, 1)) > bin_upper_edges, axis=1)

def nudge(matrix, delta, generator):
    return((matrix + generator.uniform(low=0,
                                       high=delta,
                                       size=(matrix.shape)))/(1+delta))

class identity():
    def predict_proba(self, x):
        return x
    def predict(self, x):
        return np.argmax(x, axis=1)

class HB_binary(object):
    def __init__(self, n_bins=15):
        ### Hyperparameters
        self.delta = 1e-10
        self.n_bins = n_bins

        ### Parameters to be learnt 
        self.bin_upper_edges = None
        self.mean_pred_values = None
        self.num_calibration_examples_in_bin = None

        ### Internal variables
        self.fitted = False

        ### Randomness generator
        self.generator = np.random.default_rng(2024)
        
    def fit(self, y_score, y):
        assert(self.n_bins is not None), "Number of bins has to be specified"
        y_score = y_score.squeeze()
        y = y.squeeze()
        assert(y_score.size == y.size), "Check dimensions of input matrices"
        assert(y.size >= self.n_bins), "Number of bins should be less than the number of calibration points"
        
        ### All required (hyper-)parameters have been passed correctly
        ### Uniform-mass binning/histogram binning code starts below

        # delta-randomization
        y_score = nudge(y_score, self.delta, self.generator)

        # compute uniform-mass-bins using calibration data
        self.bin_upper_edges = get_uniform_mass_bins(y_score, self.n_bins)

        # assign calibration data to bins
        bin_assignment = bin_points(y_score, self.bin_upper_edges)

        # compute bias of each bin 
        self.num_calibration_examples_in_bin = np.zeros([self.n_bins, 1])
        self.mean_pred_values = np.empty(self.n_bins)
        for i in range(self.n_bins):
            bin_idx = (bin_assignment == i)
            self.num_calibration_examples_in_bin[i] = sum(bin_idx)

            # nudge performs delta-randomization
            if (sum(bin_idx) > 0):
                self.mean_pred_values[i] = nudge(y[bin_idx].mean(),
                                                 self.delta, self.generator)
            else:
                self.mean_pred_values[i] = nudge(0.5, self.delta, self.generator)
        # check that my code is correct
        assert(np.sum(self.num_calibration_examples_in_bin) == y.size)

        # histogram binning done
        self.fitted = True

        return self

    def predict_proba(self, y_score):
        assert(self.fitted is True), "Call HB_binary.fit() first"
        y_score = y_score.squeeze()

        # delta-randomization
        y_score = nudge(y_score, self.delta, self.generator)
        
        # assign test data to bins
        y_bins = bin_points(y_score, self.bin_upper_edges)
            
        # get calibrated predicted probabilities
        y_pred_prob = self.mean_pred_values[y_bins]
        return y_pred_prob

class HB_toplabel(object):
    def __init__(self, points_per_bin=50):
        ### Hyperparameters
        self.points_per_bin = points_per_bin

        ### Parameters to be learnt 
        self.hb_binary_list = []
        
        ### Internal variables
        self.num_classes = None
    
    def fit(self, pred_mat, y):
        assert(self.points_per_bin is not None), "Points per bins has to be specified"
        assert(np.size(pred_mat.shape) == 2), "Prediction matrix should be 2 dimensional"
        y = y.squeeze()
        assert(pred_mat.shape[0] == y.size), "Check dimensions of input matrices"
        self.num_classes = pred_mat.shape[1]
        assert(np.min(y) >= 1 and np.max(y) <= self.num_classes), "Labels should be numbered 1 ... L, where L is the number of columns in the prediction matrix"
        
        top_score = np.max(pred_mat, axis=1).squeeze()
        pred_class = (np.argmax(pred_mat, axis=1)+1).squeeze()

        for l in range(1, self.num_classes+1, 1):
            pred_l_indices = np.where(pred_class == l)
            n_l = np.size(pred_l_indices)

            bins_l = np.floor(n_l/self.points_per_bin).astype('int')
            if(bins_l == 0):
               self.hb_binary_list += [identity()]
               print("Predictions for class {:d} not recalibrated since fewer than {:d} calibration points were predicted as class {:d}.".format(l, self.points_per_bin, l))
            else:
                hb = HB_binary(n_bins = bins_l)
                hb.fit(top_score[pred_l_indices], y[pred_l_indices] == l)
                self.hb_binary_list += [hb]
        
        # top-label histogram binning done
        self.fitted = True

    def predict_proba(self, pred_mat):
        assert(self.fitted is True), "Call HB_binary.fit() first"
        assert(np.size(pred_mat.shape) == 2), "Prediction matrix should be 2 dimensional"
        assert(self.num_classes == pred_mat.shape[1]), "Number of columns of prediction matrix do not match number of labels"
        
        top_score = np.max(pred_mat, axis=1).squeeze()
        pred_class = (np.argmax(pred_mat, axis=1)+1).squeeze()

        n = pred_class.size
        pred_top_score = np.zeros((n))
        for i in range(n):
            pred_top_score[i] = self.hb_binary_list[pred_class[i]-1].predict_proba(top_score[i])

        return pred_top_score

    def fit_top(self, top_score, pred_class, y):
        assert(self.points_per_bin is not None), "Points per bins has to be specified"

        top_score = top_score.squeeze()
        pred_class = pred_class.squeeze()
        y = y.squeeze()

        assert(min(np.min(y), np.min(pred_class)) >= 1), "Labels should be numbered 1 ... L, use HB_binary for a binary problem"
        assert(top_score.size == y.size), "Check dimensions of input matrices"
        assert(pred_class.size == y.size), "Check dimensions of input matrices"
        assert(y.size >= self.n_bins), "Number of bins should be less than the number of calibration points"

        self.num_classes = max(np.max(y), np.max(pred_class))
        
        for l in range(1, self.num_classes+1, 1):
            pred_l_indices = np.where(pred_class == l)
            n_l = np.size(pred_l_indices)

            bins_l = np.floor(n_l/self.points_per_bin).astype('int')
            if(bins_l == 0):
               self.hb_binary_list += [identity()]
               print("Predictions for class {:d} not recalibrated since fewer than {:d} calibration points were predicted as class {:d}".format(self.points_per_bin, l))
            else:
                hb = HB_binary(n_bins = bins_l)
                hb.fit(top_score[pred_l_indices], y[pred_l_indices] == l)
                self.hb_binary_list += [hb]
        
        # top-label histogram binning done
        self.fitted = True

    def predict_proba_top(self, top_score, pred_class):
        assert(self.fitted is True), "Call HB_binary.fit() first"
        top_score = top_score.squeeze()
        pred_class = pred_class.squeeze()
        assert(top_score.size == pred_class.size), "Check dimensions of input matrices"
        assert(np.min(pred_class) >= 1 and np.min(pred_class) <= self.num_classes), "Some of the predicted labels are not in the range of labels seen while calibrating"
        n = pred_class.size
        pred_top_score = np.zeros((n))
        for i in range(n):
            pred_top_score[i] = self.hb_binary_list[pred_class[i]-1].predict_proba(top_score[i])

        return pred_top_score
        

if __name__ == "__main__":

    from utils.data_model import generate_problem
    import seaborn as sns
    import matplotlib.pyplot as plt
    from analytics.utils_plot import config_sns
    #config_sns()

    a = np.array([0.1, 0.1, 0.7, 0.05, 0.05])
    print(softmax(np.log(a) / 1000))
    exit()

    full_data = []

    for c in [1.0, 0.7, 0.5, 0.3]:
        break
        X, Y, X_features = generate_problem(1000, 10, 10, class_sep=c, seed=2024)

        for scal in np.linspace(0.1, 10):

            #X_new = temperature_scaling(X.T, scal).T

            X_new =  top_calibrated(X.T, np.random.RandomState(2024)).T

            total_ece = 0
            total_ll = 0
            for idx in range(10):
                accuracy_bins, confidence, ece = calibration_error(
                    idx,
                    Y,
                    X_new,
                    20
                )
                total_ece += ece * 1/10
                total_ll += - np.sum([X[idx, i]*np.log(X_new[idx, i]) for i in range(len(Y))]) * 1/10

            accuracy = np.mean([int(np.argmax(X_new[:, x]) == y) for y, x in zip(Y, range(len(Y)))])


            full_data.append(
                [str(c), scal, total_ece*100]
            )
            #full_data.append(
            #    [str(c), scal, "TCE", total_ece*100]
            #)

            print(c, '\t', scal, '\t', round(accuracy, 3), '\t', round(total_ece*100, 3), '\t', total_ll)
        print()

    #import pandas as pd

    #full_data = pd.DataFrame(full_data, columns=["Accuracy", "Temperature", "ECE"])

    #sns.lineplot(
    #    data=full_data,
    #    y="ECE",
    #   x="Temperature",
    #    hue="Accuracy"
    #)
    #plt.show()
        
    X, Y, X_features = generate_problem(20000, 3, 2, class_sep=1.0, seed=2024)
    for t in [0.1, 1.0, 10]:
        X_new =  temperature_scaling(X.T, t).T

        Y_test_cal = [1 if y==1 else 0 for y in Y]
        predicted_probs = [x[1] for x in X_new.T]
        #plt.hist(predicted_probs)
        #plt.show()
        #print(predicted_probs)
        import matplotlib.pyplot as plt
        from sklearn.calibration import calibration_curve
        prob_true, prob_pred = calibration_curve(Y_test_cal, predicted_probs, n_bins=20)
        plt.plot(prob_pred, prob_true, marker='o', label=f'Temp = {t}')

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve')
    plt.legend()
    plt.show()

