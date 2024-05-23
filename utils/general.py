import numpy as np
import itertools
from tqdm import tqdm

from itertools import chain, combinations

from copy import deepcopy

from sklearn.model_selection import train_test_split

def perturb_features(X, features, mixture, mean=0, variance=1,
                     generator = np.random.default_rng(2024)):
    X_tmp = deepcopy(X)
    N = len(X_tmp[:, 0])
    for label in features:
        X_tmp[:, label] = (1-mixture)*X_tmp[:, label] + mixture*generator.normal(mean, variance, N)
    return X_tmp

def multinomial_logit(original_prediction_set, true_label, user_cm, generator: np.random.RandomState = np.random.RandomState(2024)):
  """Use the multinomial logisitic model to return the user choice."""

  # If the prediction set is empty, then return -1 (which is never a valid label)
  if len(original_prediction_set) == 0:
    prediction_set = list(range(len(user_cm[:, 0])))
  else:
    prediction_set = original_prediction_set

  if true_label in prediction_set:

    denom = sum([user_cm[x, true_label] for x in prediction_set])
    new_probabs = [user_cm[k, true_label]/denom for k in prediction_set]

    return generator.choice(prediction_set, p=new_probabs)

  else:
    # If the true label is not present, then the user choses with uniform
    # probability, since it will be wrong anyway
    return generator.choice(prediction_set)

def top_k(x_test, model_cm, k=3):
    """Return the top-k prediction set given the model prediction."""

    # Get probabilities from the classifier and assign a label
    # to each of them
    probabilities = model_cm[:, x_test]
    probabilities = [(label, prob) for label, prob in enumerate(probabilities)]

    # Sort everything
    probabilities = sorted(probabilities, key= lambda x: x[1], reverse=True)

    # Get only the top-k elements and append it to the solution
    pset = [x[0] for x in probabilities[:k]]
    
    return pset

def brute_force(X_test, classifier, user_cms, k=16, difficulties=None, verbose=False):

    def calc(pset, cm, probabilities):
        total = 0
        for y in pset:
            denom = 0
            for i in pset:
                denom += cm[i,y]
            total += probabilities[y] * (cm[y,y] / denom)
        return total

    all_prediction_sets = []

    for x_test in tqdm(X_test, disable=not verbose):

        # Get the corresponding confusion matrix
        user_cm = user_cms.get(-1) if difficulties is None else user_cms.get(
            difficulties[x_test]
        )

        # Get how many labels we have, minus the true one
        n = len(user_cm[0])
        labels = list(range(n))

        # Get probabilities from the classifier
        probabilities = classifier[:, x_test]

        # All solutions for each k
        all_solutions_for_every_k = {}

        # Best solution found so far
        max_total = (None, -np.inf)

        # Loop over all the possible permutation (without the first label)
        for sequence in chain.from_iterable(combinations(labels, r) for r in range(len(labels)+1)):
            # Avoid the empty set
            if len(sequence) == 0:
                continue

            pset = list(sequence)
            total = calc(pset, user_cm, probabilities)

            if max_total[1] <= total:
                max_total = (deepcopy(pset), total)

            all_solutions_for_every_k[n] = deepcopy(max_total[0])

        all_prediction_sets.append(all_solutions_for_every_k)

    return all_prediction_sets

def greedy(x_test, model_cm, user_cm, k, generator=np.random.default_rng(2024)):

    probabilities = model_cm[:, x_test]

    def calc(pset, cm, probabilities):

        if len(pset) == 1:
            return probabilities[pset[0]]

        total = 0
        for y in pset:
            denom = 0
            for i in pset:
                denom += cm[i,y]
            total += probabilities[y] * (cm[y,y] / denom)
        return total

    S = []

    # Sort labels and probabilities
    full_labels = [label for label, _ in sorted([(x,p) for x,p in enumerate(probabilities)], key=lambda z: z[1], reverse=True)]

    # Iterate over all possible lengths
    for K in range(1, len(full_labels)+1):

        labels = deepcopy(full_labels[:K])
        S_k = []

        for _ in range(K):
            scores = []
            for y in labels:
                submodul = calc(S_k+[y], user_cm, probabilities) - calc(S_k, user_cm, probabilities)
                scores.append((y, submodul))

            # If there is no way to improve from the previous step, then stop
            if all(s[1] <= 0 for s in scores):
                break

            # Get all maximum values and pick one at random
            current_max_gain_values = max(scores, key=lambda x: x[1])[1]
            index = generator.choice([i for i in range(len(scores)) if scores[i][1] == current_max_gain_values])

            S_k.append(
                scores[index][0]
            )

            # Remove the element we found from the potential labels
            labels.pop(labels.index(S_k[-1]))
        
        # Assign the best solution
        if calc(S_k, user_cm, probabilities) > calc(S, user_cm, probabilities):
            S = deepcopy(S_k)

    assert len(S) <= len(full_labels), (len(S), len(full_labels))

    # If we get a singleton, it must be the maximum value of the classifier.
    if len(S) == 1:
        assert probabilities[S[0]] == np.max(probabilities), (S[0], np.max(probabilities), probabilities[S[0]])

    return S

def quantile(X_test, y_test, model_cm, alpha):
    """ Compute the quantile based only on frequencies and model confusion matrix"""

    n = len(X_test)
    quantile = min(1, np.ceil((1 - alpha)*(n+1))/n)
    scores = [1-model_cm[y_true, x_test] for x_test, y_true in zip(X_test, y_test)]
    return np.quantile(scores, quantile)

def quantile_APS(X_test, y_test, model_cm, alpha):
    """Compute the quantile following Adaptive Prediction Sets"""
    n = len(X_test)
    quantile = min(1, np.ceil((n+1)*(1-alpha))/n)
    scores = []
    for x_test, y_true in zip(X_test, y_test):
        temporary_score = 0
        sorted_scores = sorted(enumerate(model_cm[:, x_test]), key=lambda x: x[1], reverse=True)
        for label, value in sorted_scores:
            temporary_score += value
            if label == y_true:
                break
        scores.append(temporary_score)
    return np.quantile(scores, quantile)

def conformal_set_APS(x_test, model_cm, q_hat, return_empty=False):
    """Generate the conformal set for a given class"""
    n = len(model_cm[:, x_test])
    c_set = []

    temporary_score = 0
    sorted_scores = sorted(enumerate(model_cm[:, x_test]), key=lambda x: x[1], reverse=True)

    for label, value in sorted_scores:
        temporary_score += value
        if temporary_score <= q_hat:
            c_set.append(label)
        else:
            # We add this label anyway to avoid zero size prediction sets
            c_set.append(label)
            break

    return c_set if len(c_set) > 0 or return_empty else [int(np.argmax(model_cm[:, x_test]))]

def conformal_set(x_test, model_cm, q_hat, return_empty=False, return_full_set=False, verbose=False):
    """Generate the conformal set for a given class"""
    n = len(model_cm[:, x_test])
    c_set = [ k for k in range(n) if model_cm[k, x_test] >= 1-q_hat]
    if return_empty and len(c_set) == 0:
        return []
    if return_full_set and len(c_set) == 0:
        return list(range(n))
    return c_set if len(c_set) > 0 else [int(np.argmax(model_cm[:, x_test]))]

def user_accuracy(Y_test, prediction_sets, user_cms, image_difficulties: list=None):
    """Compute the user accuracy, givem the MNL model and prediction sets"""
    assert -1 in user_cms, "Warning! When computing the accuracy, we are missing the default MNL!"
    assert len(Y_test) == len(image_difficulties) if image_difficulties is not None else True, "Warning! The difficulties and the test set are misaligned!"
    acc_user = []
    for l, (y_true, pset) in enumerate(zip(Y_test, prediction_sets)):
        if y_true in pset:
            # Get the image level difficulty
            image_level = -1 if image_difficulties is None else image_difficulties[l]
            current_user_cm = user_cms.get(image_level)

            denom = sum([current_user_cm[x, y_true] for x in pset])
            acc_user.append(current_user_cm[y_true, y_true]/denom if len(pset) > 1 else 1)
        else:
            acc_user.append(0)
    return (np.mean(acc_user), np.std(acc_user))

def empirical_coverage(Y_test, prediction_sets):
    emp_cov_new = 0
    for y_true, pset in zip(Y_test, prediction_sets):
        emp_cov_new += int(y_true in pset)
    emp_cov_new /= len(Y_test)
    return emp_cov_new

def split_data_into_sets(full_data, Y, training_set_size, calibration_set_size, seed=2024):
    """Split the dataset indexes into training, calibration and testing"""

    total_indexes = list(range(full_data))

    # Get the training indexes only if we are requesting them
    # We keep the training splitting FIXED.
    training_set_indexes = []
    if training_set_size > 0:
        total_indexes, training_set_indexes, Y, _ = train_test_split(total_indexes, Y,
                                                                     test_size=training_set_size,
                                                                     stratify=Y,
                                                                     random_state=2024)
    
    test_set_indexes, calibration_set_indexes, Y, _ = train_test_split(total_indexes, Y, test_size=calibration_set_size,
                                                                       stratify=Y,
                                                                    random_state=seed)

    return training_set_indexes, test_set_indexes, calibration_set_indexes

def evaluate_prediction_sets(prediction_sets, Y_test, user_cm_difficulties, image_difficulties: list=None):
    """Evaluate the empirical success probability, prediction set size and coverage."""

    avg_len_psets = sum(map(len, prediction_sets))/len(prediction_sets)

    # Compute the human accuracy over our prediction sets
    acc_user, _ = user_accuracy(Y_test, prediction_sets, user_cms=user_cm_difficulties, image_difficulties=image_difficulties)

    # Empirical coverage
    emp_cov = empirical_coverage(Y_test, prediction_sets)

    return avg_len_psets, acc_user, emp_cov