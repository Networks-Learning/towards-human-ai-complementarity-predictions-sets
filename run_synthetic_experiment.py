import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.general import quantile, quantile_APS
from utils.general import brute_force, top_k, conformal_set_APS, conformal_set, greedy
from utils.general import user_accuracy, empirical_coverage
from utils.general import split_data_into_sets, perturb_features

from utils.custom_make_classification import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from utils.train_utils import train_model
from utils.calibration_utils import top_k_label_calibration, get_sorted_probabilities_and_labels, ECE

import random
from copy import deepcopy
import argparse
import pickle

from mpi4py import MPI
import os

# Initialize MPI
comm = MPI.COMM_WORLD
rank_mpi = comm.Get_rank()
size = comm.Get_size()

# Helper function to parallelize the bruteforce algorithm
def _parallel_bruteforce(X, start, end, original_model_cm, user_cms, total_labels=10, difficulties=None, verbose=True):
    return brute_force(X[start:end], original_model_cm, user_cms,
                       difficulties=difficulties,
                       k=total_labels, verbose=verbose)

np.random.seed(2024)
random.seed(2024)

# We hardcode the class separator parameters
class_separators_list = {
    10: [0.43334, 0.83334, 1.3, 2.0],
    25: [0.83334, 1.2333, 2.1, 2.6],
    50: [1.23, 1.7, 2.2, 3.2]
}

informative_features_list = {
    10: 4,
    25: 5,
    50: 6
}

calibration_best_points_per_bins = {
    10: 30,
    25: 20,
    50: 10
}

TOP_K_CALIBRATION_RANKS = 5

parser = argparse.ArgumentParser()
parser.add_argument("--labels", type=int, choices=[10,25,50], default=10)
parser.add_argument("--calibration-size", type=int, default=1000)
parser.add_argument("--training-size", type=int, default=8000)
parser.add_argument("--example-set-size", type=int, default=1000)
parser.add_argument("--number-of-features", type=int, default=20)
parser.add_argument("--skip-brute-force", action="store_true", default=False)
parser.add_argument("--calibrate", action="store_true", default=False)
parser.add_argument("--features-to-perturb", type=int, default=1)
parser.add_argument("--test", action="store_true", default=False)

args = parser.parse_args()

# Total labels we are going to use
total_labels = args.labels

# Number of features for each example
number_of_features = args.number_of_features

# How many examples we use
example_set_size = args.example_set_size

# Calibration size
calibration_size = args.calibration_size

# Training set size
training_size = args.training_size

# Since the calibration set has only m instances, we have only m potential
# quantiles. Thus we have
quantile_alphas = [1-(i/(calibration_size+1)) for i in range(1, calibration_size+1)]

# Keep all evaluations here
full_evals = []

# Full prediction sets
all_prediction_sets = {}

# All ground truth labels
all_ground_truth_labels = {}

# All the users confusion matrix for each run id
all_users_cms = {}

# All the users confusion matrix for each run id
all_users_cms_full = {}

# Add each human classifier for evaluation
all_human_classifiers = {}

# Add each test data for evaluation
all_test_data = {}

# All machine classifiers
all_machine_classifiers = {}

# Generate filename
filename = f"{example_set_size}-{total_labels}-{number_of_features}-{calibration_size}-{args.calibrate}-{args.skip_brute_force}-{args.features_to_perturb}"

# Dictionary containing all prediction sets for each model
all_prediction_sets_alpha_values = {
   "CP": {},
   "APS": {},
}

# Values of K we want to try
K = [total_labels]

# Top-K values for the experiments
Top_K_values = [2,3,5]

# Feature to perturb
features_to_perturb = list(range(informative_features_list.get(total_labels)))[-args.features_to_perturb:]
assert len(features_to_perturb) == args.features_to_perturb

# Load pretrained classifiers and humans
pre_trained_classifier = None
pre_trained_humans = None
if os.path.isfile(f'{filename}-machine_classifiers.pickle'):
    print("Loading pretrained models")
    pre_trained_classifier = pickle.load(open(f'{filename}-machine_classifiers.pickle', "rb"))
if os.path.isfile(f'{filename}-human_classifiers.pickle'):
    print("Loading pretrained humans")
    pre_trained_humans = pickle.load(open(f'{filename}-human_classifiers.pickle', "rb"))

# For every potential ground truth P(Y|X)
for epsilon in tqdm(class_separators_list.get(total_labels)):

    # Create the classification data task
    X_features_entire_task, Y_entire_task = make_classification(
            n_samples=example_set_size+calibration_size+training_size+8000,
            n_features=number_of_features, n_classes=total_labels,
            n_informative=informative_features_list.get(total_labels),
            n_redundant=0, n_clusters_per_class=1,
            class_sep=epsilon,
            random_state=2024,
            shuffle=False,
            random_covariance=True,
            ordered_hypercube_vertices=True
        )

    # Iterate over the mixture parameter
    for pi in [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]:
        
        # For each run id, create a different dataset split
        for run_id in tqdm(range(10), desc=f"Running P(Y|X)={epsilon}, P(Y'|Y,C(X))={pi}", disable=(rank_mpi!=0)):

            # Split the features into humans and machine samples
            X_features, X_features_humans, Y_train, Y_train_humans = train_test_split(X_features_entire_task, Y_entire_task, test_size=8000, stratify=Y_entire_task, random_state=2024+run_id)

            # Split again the model for training
            X_features_train, X_test_original, Y_train, Y_test_original = train_test_split(X_features, Y_train, test_size=0.2, stratify=Y_train, random_state=2024+run_id)
            
            if pre_trained_classifier is not None:
                classifier = pre_trained_classifier.get((epsilon, pi, run_id))
            else:
                # Train the model and remove features used to train the model
                classifier, _ = train_model(X_features_train, Y_train, None, return_model=True)
            
            # Delete some features
            del X_features
            del X_features_train
            del Y_train

            # Get all the model softmaxes for all the test instances
            model_cm_original = classifier.predict_proba(X_test_original).T
            
            if pre_trained_humans is not None:
                human_model = pre_trained_humans.get((epsilon, pi, run_id))
            else:
                # Perturb the features given the pi value
                X_features_humans_train = perturb_features(X_features_humans, features_to_perturb, pi, generator=np.random.default_rng(2024+run_id))

                # Train the human model and build confusion matrices 
                human_model, _ = train_model(X_features_humans_train, Y_train_humans, None, return_model=True)

            # Copy the classifier and the human
            all_machine_classifiers[(epsilon, pi, run_id)] = deepcopy(classifier)
            all_human_classifiers[(epsilon, pi, run_id)] = deepcopy(human_model)
            
            good_confusion_matrix = False
            good_test_confusion_matrix = False
            confusion_matrices_random_counter = 0

            while (not good_confusion_matrix or not good_test_confusion_matrix):
                
                # Reset these variables
                good_confusion_matrix = False
                good_test_confusion_matrix = False

                # Generate the indexes for training/validation in a stratified fashion
                # The training set has always the same value for each run_id
                # The stocasticity comes only from the training set.
                _, X_test_ids, X_cal_ids = split_data_into_sets(
                    len(X_test_original),
                    Y_test_original,
                    0,
                    calibration_size,
                    seed=2024+run_id+confusion_matrices_random_counter
                )

                # Sort the various indices to preserve the ordering
                X_test_ids = sorted(X_test_ids)
                X_cal_ids = sorted(X_cal_ids)

                # Split the dataset correctly
                X_test, Y_test = X_test_ids, Y_test_original[X_test_ids]
                X_cal, Y_cal = X_cal_ids, Y_test_original[X_cal_ids]

                # Perturb the human features and create the confusion matrix and
                # build the confusion matrix as requested.
                X_cal_human = perturb_features(X_test_original[X_cal_ids, :], features_to_perturb, pi,
                                               generator=np.random.default_rng(2024+run_id))
                user_cm = confusion_matrix(Y_cal, human_model.predict(X_cal_human))

                # We reshape it to get C_ij where i is predicted label and j the true one
                user_cm = np.array(user_cm).T
                user_cm = user_cm / user_cm.astype(float).sum(axis=0)

                # Check if the matrix is sound
                # We skip those split for which we do not have a good confusion matrix
                if not all([user_cm[x,x] > 0 for x in range(total_labels)]):
                    if args.test:
                        print("ERROR with ", run_id, epsilon, pi)
                else:
                    good_confusion_matrix = True

                # Generate a full confusion matrix for evaluation later
                # build the confusion matrix as requested.
                X_test_human = perturb_features(X_test_original[X_test_ids, :], features_to_perturb, pi, generator=np.random.default_rng(2024+run_id))
                user_cm_full = confusion_matrix(Y_test, human_model.predict(X_test_human))
                user_cm_full = np.array(user_cm_full).T
                user_cm_full = user_cm_full / user_cm_full.astype(float).sum(axis=0)
                all_users_cms_full[(run_id, epsilon, pi)] = deepcopy(user_cm_full)

                # Check if the test matrix is sound
                if not all([user_cm_full[x,x] > 0 for x in range(total_labels)]):
                    if args.test:
                        print("CM ERROR with test", run_id, epsilon, pi)
                else:
                    good_test_confusion_matrix = True
                
                # Increase the counter
                confusion_matrices_random_counter +=1
            
            assert all([user_cm[x,x] > 0 for x in range(total_labels)])
            assert all([user_cm_full[x,x] > 0 for x in range(total_labels)])

            # If we are testing, just skip the computation
            if args.test:
                continue

            # This is to accomodate potential different users
            user_cm_difficulties = {
                -1: user_cm
            }

            # If it is requested, calibrate the model
            model_cm = deepcopy(model_cm_original)
            if args.calibrate:
                
                model_cm_cal = classifier.predict_proba(X_test_original[X_cal_ids, :]).T

                calibrators_per_label = top_k_label_calibration(
                    range(len(model_cm_cal[0, :])), model_cm_cal, Y_cal,
                    calibration_best_points_per_bins.get(total_labels),
                    ranks=TOP_K_CALIBRATION_RANKS
                )

                _, sorted_softmaxes_keys = get_sorted_probabilities_and_labels(
                        model_cm, range(len(model_cm[0, :]))
                    )

                for idx in range(len(model_cm[0, :])):
                    for rank, label in enumerate(sorted_softmaxes_keys[idx]):
                        if rank < TOP_K_CALIBRATION_RANKS:
                            model_cm[label, idx] = calibrators_per_label[label][rank].predict_proba(model_cm[label, idx])[0]

            if rank_mpi == 0:
                # Copy the ground truth labels
                all_ground_truth_labels[(run_id, epsilon, pi)] = deepcopy(Y_test)

                # Copy the user confusion matrix
                all_users_cms[(run_id, epsilon, pi)] = deepcopy(user_cm)

                # All test data
                all_test_data[(run_id, epsilon, pi)] = deepcopy(X_test_original[X_test_ids, :])

            if rank_mpi == 0:
                # Define loop parameters
                total_iterations = len(X_test)
                chunk_size = total_iterations // size

                # Scatter iterations across processes
                for i in range(1, size):
                    start = i * chunk_size
                    end = start + chunk_size
                    if i == size-1 and end < total_iterations:
                        end = total_iterations
                    comm.send((start, end, X_test), dest=i)

                # Master process computation
                start = 0
                end = chunk_size
                prediction_sets_brute_force = _parallel_bruteforce(X_test, start, end, model_cm,
                                                                user_cms=user_cm_difficulties, verbose=False) if not args.skip_brute_force else []
                prediction_sets_greedy = { k: [greedy(x_test, model_cm, user_cm, k) for x_test in X_test[start:end]] for k in K}
                prediction_sets_top_k = {k: [top_k(x_test, model_cm, k=k) for x_test in X_test[start:end]] for k in Top_K_values}

                # Gather results from other processes
                for i in range(1, size):
                    pset_brute_worker, pset_greedy_worker, pset_topk_worker = comm.recv(source=i)
                    prediction_sets_brute_force += pset_brute_worker
                    for k in prediction_sets_greedy:
                        prediction_sets_greedy[k] += pset_greedy_worker[k]
                    for k in prediction_sets_top_k:
                        prediction_sets_top_k[k] += pset_topk_worker[k]
                                
                assert all(len(prediction_sets_greedy[y]) == len(X_test) for y in prediction_sets_greedy)

                # Convert bruteforce correctly
                prediction_sets_brute_force = {k: [element.get(k) for element in prediction_sets_brute_force] for k in K} if not args.skip_brute_force else {}

            else:
                start, end, X_test = comm.recv(source=0)
                # Compute all the prediction sets for both greedy and brute-force if needed
                prediction_sets_brute_force = _parallel_bruteforce(X_test, start, end, model_cm,
                                                                user_cms=user_cm_difficulties, verbose=False) if not args.skip_brute_force else []
                prediction_sets_greedy = { k: [greedy(x_test, model_cm, user_cm, k) for x_test in X_test[start:end]] for k in K}
                prediction_sets_top_k = {k: [top_k(x_test, model_cm, k=k) for x_test in X_test[start:end]] for k in Top_K_values}

                # Send everything to the main process
                comm.send((
                    prediction_sets_brute_force,
                    prediction_sets_greedy,
                    prediction_sets_top_k
                ), dest=0)


            if rank_mpi == 0:
                # Define loop parameters
                total_iterations = len(quantile_alphas)
                chunk_size = total_iterations // size

                # Scatter iterations across processes
                for i in range(1, size):
                    start = i * chunk_size
                    end = start + chunk_size
                    if i == size-1 and end < total_iterations:
                        end = total_iterations
                    comm.send((start, end), dest=i)
            
                start, end = 0, chunk_size

                # Build also the prediction sets for conformal prediction
                prediction_sets_conformal = {}
                for alpha in quantile_alphas[start:end]:

                        # Compute the quantile for the given alpha
                        q_hat = quantile(X_cal, Y_cal, model_cm, alpha)
                        
                        # Build the prediction sets for this alpha
                        # Since we correctly know P(Y|X), then, we can simply pick the top-k
                        # elements until we reach the 1-alpha cut-off, and that is our
                        # prediction set.
                        prediction_sets_conformal[alpha] = [conformal_set(x_test, model_cm, q_hat) for x_test in X_test]
                
                # Build also the prediction sets for adaptive conformal prediction
                prediction_sets_conformal_APS = {}
                for alpha in quantile_alphas[start:end]:

                        # Compute the quantile for the given alpha
                        q_hat = quantile_APS(X_cal, Y_cal, model_cm, alpha)
                        
                        # Build the prediction sets for this alpha
                        # Since we correctly know P(Y|X), then, we can simply pick the top-k
                        # elements until we reach the 1-alpha cut-off, and that is our
                        # prediction set.
                        prediction_sets_conformal_APS[alpha] = [conformal_set_APS(x_test, model_cm, q_hat) for x_test in X_test]
                
                # Gather results from other processes
                for i in range(1, size):
                    pset_cp_worker, pset_aps_worker = comm.recv(source=i)
                    prediction_sets_conformal.update(pset_cp_worker)
                    prediction_sets_conformal_APS.update(pset_aps_worker)

            else:
                start, end = comm.recv(source=0)

                # Build also the prediction sets for conformal prediction
                prediction_sets_conformal = {}
                for alpha in quantile_alphas[start:end]:

                        # Compute the quantile for the given alpha
                        q_hat = quantile(X_cal, Y_cal, model_cm, alpha)
                        
                        # Build the prediction sets for this alpha
                        # Since we correctly know P(Y|X), then, we can simply pick the top-k
                        # elements until we reach the 1-alpha cut-off, and that is our
                        # prediction set.
                        prediction_sets_conformal[alpha] = [conformal_set(x_test, model_cm, q_hat) for x_test in X_test]
                
                # Build also the prediction sets for adaptive conformal prediction
                prediction_sets_conformal_APS = {}
                for alpha in quantile_alphas[start:end]:

                        # Compute the quantile for the given alpha
                        q_hat = quantile_APS(X_cal, Y_cal, model_cm, alpha)
                        
                        # Build the prediction sets for this alpha
                        # Since we correctly know P(Y|X), then, we can simply pick the top-k
                        # elements until we reach the 1-alpha cut-off, and that is our
                        # prediction set.
                        prediction_sets_conformal_APS[alpha] = [conformal_set_APS(x_test, model_cm, q_hat) for x_test in X_test]
                
                # Send everything to the main process
                comm.send((
                    prediction_sets_conformal,
                    prediction_sets_conformal_APS
                ), dest=0)


            if rank_mpi == 0:

                # Human accuracy alone
                prediction_sets = [list(range(total_labels)) for _ in range(len(Y_test))]
                acc_user, std_user = user_accuracy(Y_test, prediction_sets, user_cm_difficulties)
                full_evals.append(
                    [pi, f"Human (Alone)", acc_user, std_user, total_labels, 1, 0, epsilon]
                )

                # Compute all prediction sets for bruteforce
                # Skip if we want only the greedy solution
                if not args.skip_brute_force:

                    for k in K:

                        # Extract only the prediction sets for the given k
                        prediction_sets = prediction_sets_brute_force.get(k)

                        # Create the new prediction sets and compute their average length
                        avg_len_psets = sum(map(len, prediction_sets))/len(prediction_sets)

                        # Compute the human accuracy over our prediction sets
                        acc_user, std_user = user_accuracy(Y_test, prediction_sets, user_cm_difficulties)

                        # Empirical coverage
                        emp_cov = empirical_coverage(Y_test, prediction_sets)

                        # For each alpha, add the same value since they do not change
                        full_evals.append(
                            [pi, f"BF (k={k})", acc_user, avg_len_psets, emp_cov, 0, epsilon]
                        )

                        # Add also the prediction sets
                        key = (pi, f"BF (k={k})", acc_user, std_user, avg_len_psets, emp_cov, 0, run_id, epsilon)
                        if key not in all_prediction_sets:
                            all_prediction_sets[key] =  deepcopy(prediction_sets)
            
                for k in K:

                    # Create the new prediction sets and compute their average length
                    prediction_sets = prediction_sets_greedy.get(k)
                    avg_len_psets = sum(map(len, prediction_sets))/len(prediction_sets)

                    # Compute the human accuracy over our prediction sets
                    acc_user, std_user = user_accuracy(Y_test, prediction_sets, user_cm_difficulties)

                    # Empirical coverage
                    emp_cov = empirical_coverage(Y_test, prediction_sets)

                    # For each alpha, add the same value since they do not change
                    full_evals.append(
                        [pi, f"Greedy (k={k})", acc_user, std_user, avg_len_psets, emp_cov, 0, epsilon]
                    )

                    # Add also the prediction sets
                    key = (pi, f"Greedy (k={k})", acc_user, std_user, avg_len_psets, emp_cov, 0, run_id, epsilon)
                    if key not in all_prediction_sets:
                        all_prediction_sets[key] =  deepcopy(prediction_sets)
                
                # Add Top-K evaluations
                for k in Top_K_values:

                    # Create the new prediction sets and compute their average length
                    prediction_sets = prediction_sets_top_k.get(k)
                    avg_len_psets = sum(map(len, prediction_sets))/len(prediction_sets)

                    # Compute the human accuracy over our prediction sets
                    acc_user, std_user = user_accuracy(Y_test, prediction_sets, user_cm_difficulties)

                    # Empirical coverage
                    emp_cov = empirical_coverage(Y_test, prediction_sets)

                    # For each alpha, add the same value since they do not change
                    full_evals.append(
                        [pi, f"Top-K (k={k})", acc_user, std_user, avg_len_psets, emp_cov, 0, epsilon]
                    )

                    # Add also the prediction sets
                    key = (pi, f"Top-K (k={k})", acc_user, std_user, avg_len_psets, emp_cov, 0, run_id, epsilon)
                    if key not in all_prediction_sets:
                        all_prediction_sets[key] =  deepcopy(prediction_sets)

                all_CP_evaluations = []
                all_CP_prediction_sets = []

                for alpha in quantile_alphas:
                    
                    # Extract the prediction set checking that it is not None
                    prediction_sets = prediction_sets_conformal.get(alpha, None)
                    assert prediction_sets != None
                    
                    # Compute the average length and empirical coverage of the sets
                    avg_len_psets = sum(map(len, prediction_sets))/len(prediction_sets)
                    emp_cov = empirical_coverage(Y_test, prediction_sets)

                    # Compute the human accuracy with the two users
                    user_prediction_bad, std_user = user_accuracy(Y_test, prediction_sets, user_cm_difficulties)

                    # Save the current prediction set
                    all_CP_prediction_sets.append(
                        deepcopy(prediction_sets)
                    )

                    # Append the results
                    all_CP_evaluations.append(
                        [pi, "CP", user_prediction_bad, std_user, avg_len_psets, emp_cov, q_hat, epsilon]
                    )

                    # Append the prediction sets for each alpha
                    if (run_id, epsilon, pi) not in all_prediction_sets_alpha_values["CP"]:
                        all_prediction_sets_alpha_values["CP"][(run_id, epsilon, pi)] = {}
                    all_prediction_sets_alpha_values["CP"][(run_id, epsilon, pi)][alpha] = (
                        user_prediction_bad, std_user
                    )
                
                # Append to the full evals only the best CP evaluations
                best_result = max(all_CP_evaluations, key=lambda x:x[2])
                best_result_index = all_CP_evaluations.index(best_result)
                full_evals.append(
                    deepcopy(best_result)
                )

                # Add also the prediction sets
                best_result.insert(-1, run_id)
                key = tuple(best_result)
                if key not in all_prediction_sets:
                    all_prediction_sets[key] =  deepcopy(all_CP_prediction_sets[best_result_index])
                
                all_CP_APS_evaluations = []
                all_CP_APS_prediction_sets = []

                for alpha in quantile_alphas:
                    
                    # Extract the prediction set checking that it is not None
                    prediction_sets = prediction_sets_conformal_APS.get(alpha, None)
                    assert prediction_sets != None
                    
                    # Compute the average length and empirical coverage of the sets
                    avg_len_psets = sum(map(len, prediction_sets))/len(prediction_sets)
                    emp_cov = empirical_coverage(Y_test, prediction_sets)

                    # Compute the human accuracy with the two users
                    user_prediction_bad, std_user = user_accuracy(Y_test, prediction_sets, user_cm_difficulties)

                    # Save the current prediction set
                    all_CP_APS_prediction_sets.append(
                        deepcopy(prediction_sets)
                    )

                    # Append the results
                    all_CP_APS_evaluations.append(
                        [pi, "APS", user_prediction_bad, std_user, avg_len_psets, emp_cov, q_hat, epsilon]
                    )

                    # Append the prediction sets for each alpha
                    if (run_id, epsilon, pi) not in all_prediction_sets_alpha_values["APS"]:
                        all_prediction_sets_alpha_values["APS"][(run_id, epsilon, pi)] = {}
                    all_prediction_sets_alpha_values["APS"][(run_id, epsilon, pi)][alpha] = (
                        user_prediction_bad, std_user
                    )
                
                # Append to the full evals only the best CP evaluations
                best_result = max(all_CP_APS_evaluations, key=lambda x:x[2])
                best_result_index = all_CP_APS_evaluations.index(best_result)
                full_evals.append(
                    deepcopy(best_result)
                )

                # Add also the prediction sets
                best_result.insert(-1, run_id)
                key = tuple(best_result)
                if key not in all_prediction_sets:
                    all_prediction_sets[key] =  deepcopy(all_CP_APS_prediction_sets[best_result_index])

if rank_mpi == 0 and not args.test:

    # Save the prediction sets
    pickle.dump(all_prediction_sets, open(f'{filename}-prediction_sets.pickle', 'wb'))
    pickle.dump(all_ground_truth_labels, open(f'{filename}-ground_truth.pickle', 'wb'))
    pickle.dump(all_users_cms, open(f'{filename}-confusion_matrices.pickle', 'wb'))
    #pickle.dump(all_users_cms_full, open(f'{filename}-confusion_matrices_full.pickle', 'wb'))
    pickle.dump(all_prediction_sets_alpha_values, open(f'{filename}-prediction_sets_alphas.pickle', "wb"))
    pickle.dump(all_human_classifiers, open(f'{filename}-human_classifiers.pickle', "wb"))
    pickle.dump(all_machine_classifiers, open(f'{filename}-machine_classifiers.pickle', "wb"))
    pickle.dump(all_test_data, open(f'{filename}-test_data.pickle', "wb"))

    # Convert the results to a dataframe and save it to disk
    full_evals = pd.DataFrame(full_evals, columns = ["pi", "method", "accuracy", "std", "length", "emp_cov", "quantile", "epsilon"])
    full_evals.to_csv(f"{filename}-results.csv", index=None)