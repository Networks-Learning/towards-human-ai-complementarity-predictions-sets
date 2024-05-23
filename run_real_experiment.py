import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy

from warnings import warn

from utils.general import greedy, brute_force, top_k
from utils.general import conformal_set, quantile, conformal_set_APS, quantile_APS
from utils.general import user_accuracy, evaluate_prediction_sets
from utils.general import split_data_into_sets

from utils.calibration_utils import ECE

from data.preprocess import extract_human_data, extract_model_data, extract_image_difficulties, extract_human_difficulties

from utils.calibration_utils import top_k_label_calibration, class_wise_calibration, top_k_confidence_calibration
from utils.calibration_utils import get_sorted_probabilities_and_labels

import random

import argparse

from mpi4py import MPI

# Initialize MPI
comm = MPI.COMM_WORLD
rank_mpi = comm.Get_rank()
size = comm.Get_size()

# Helper function to parallelize the bruteforce algorithm
def _parallel_bruteforce(X, start, end, original_model_cm, user_cms, total_labels=16, difficulties=None, verbose=True):
    return brute_force(X[start:end], original_model_cm, user_cms,
                       difficulties=difficulties,
                       k=total_labels, verbose=verbose)

np.random.seed(2024)
random.seed(2024)

MODELS = ["vgg19"]
NOISE_LEVELS = [80, 95, 110, 125]

parser = argparse.ArgumentParser()
parser.add_argument("--calibration-size", type=int, default=120)
parser.add_argument("--model-epochs", choices=["baseline", "epoch00", "epoch01", "epoch10"], default="epoch10")
parser.add_argument("--skip-brute-force", default=False, action="store_true")
parser.add_argument("--calibrate", default=None, choices=[None, "classwise", "top-k", "top-k-confidence"])
parser.add_argument("--ranks", default=1, type=int)
parser.add_argument("--calibration-bins", type=int, default=10)
parser.add_argument("--points-per-bins", type=int, default=25)
parser.add_argument("--expressive-mnl", default=False, action="store_true")
parser.add_argument("--expressive-humans", default=-1, type=int)
args = parser.parse_args()

# Finetune level of the model
model_finetune_map = f"data/ImageNet-16H/hai_{args.model_epochs}_model_preds_max_normalized.csv"

# Total labels in the ImageNet-16H task
total_labels = 16

# Top-K values
top_k_sizes = [2,3,5]

# How many calibration points have to be used
calibration_size = args.calibration_size

# List containing the full data for this evaluation
full_evals = []

# Dictionary containing all prediction sets for each model
all_prediction_sets = {}

# Dictionary containing all prediction sets for each model
all_prediction_sets_alpha_values = {
   "CP": {},
   "APS": {},
}

# All confusion matrices
all_confusion_matrices = {
   model: {
      noise : {
         run_id: [] for run_id in range(10)
      } for noise in NOISE_LEVELS
   } for model in MODELS
}

# All confusion matrix for testing
all_confusion_matrices_testing = {
   model: {
      noise : {
         run_id: [] for run_id in range(10)
      } for noise in NOISE_LEVELS
   } for model in MODELS
}

# Dictionary containing all the ground truth values
all_ground_truth = {}

for model in MODELS:
  for noise in NOISE_LEVELS:

    # Extract human data and model data for this configuration
    original_model_cm, Y, image_names, label_mapping, _ = extract_model_data(NOISE_LEVEL=noise, MODEL_NAME=model,
                                                                     model_data_path=model_finetune_map,
                                                                     random_generator=np.random.default_rng(2024))

    # Split everything for expressive humans
    human_ids = None
    if args.expressive_humans != -1:
       warn("Expressive humans do not work with multiprocessing yet!")
       human_ids, group_humans = extract_human_difficulties(NOISE_LEVEL=noise, labels_encoding=label_mapping)
       human_ids = [ x for k, x in enumerate(human_ids) if group_humans[k] == args.expressive_humans]

    # Check the consistency of the full set
    assert len(original_model_cm[0, :]) == 1200

    full_data_indeces = list(range(len(original_model_cm[0, :])))

    # The alphas needed for conformal prediction
    quantile_alphas = [1-(i/(calibration_size+1)) for i in range(1, calibration_size+1)]

    # Values of K we want to try
    K = [16]

    if rank_mpi == 0:

      if not args.calibrate:

        # Compute all the prediction sets for both greedy and brute-force if needed      
        prediction_sets_greedy = { k: [greedy(x_test, original_model_cm,
                                              user_cm_difficulties.get(difficulties[x_test] if difficulties is not None else -1),
                                              k,
                                              np.random.default_rng(2024)) for x_test in full_data_indeces] for k in K}
        prediction_sets_top_k = {k: [top_k(x_test,
                                          original_model_cm,
                                          k=k) for x_test in full_data_indeces] for k in top_k_sizes}

    # Add static values using our method. Here, we do not consider the alpha coverage,
    # but we instead maximize by bruteforcing everything
    for run_id in tqdm(range(10), desc=f"{model} (Noise={noise})", disable=rank_mpi != 0):

      # Split the set in calibration and testing (we do not have the training elements)
      _, test_indeces, cal_indeces = split_data_into_sets(
          len(Y), Y, 0, calibration_size, seed=2024+run_id
      )

      # Sort the test indeces and the calibration indeces
      # Otherwise, the filtering later will fail miserably (for greedy, brute-force and top-k)
      test_indeces = sorted(test_indeces)
      cal_indeces = sorted(cal_indeces)

      # Rename the indeces
      X_test, Y_test = test_indeces, np.array(Y)[test_indeces]
      X_cal, Y_cal = cal_indeces, np.array(Y)[cal_indeces]

      assert len(X_test) == 1200 - len(X_cal)

      # Run calibration if requested
      calibrators_per_label = []

      # Copy the original scores
      model_cm = deepcopy(original_model_cm)

      if args.calibrate == "classwise":
          calibrators_per_label = class_wise_calibration(
              cal_indeces, original_model_cm, Y, n_bins=args.calibration_bins
          )

          # Then, we calibrate everything using each time a calibrator
          model_cm = deepcopy(original_model_cm)
          for idx in full_data_indeces:
            for label in range(16):
              model_cm[label, idx] = calibrators_per_label[label].predict_proba(model_cm[label, idx])[0]

      elif args.calibrate == "top-k":
          
          # Use top-k-label calibration 
          calibrators_per_label = top_k_label_calibration(
              cal_indeces, original_model_cm, Y, args.points_per_bins, ranks=args.ranks
          )

          _, sorted_softmaxes_keys = get_sorted_probabilities_and_labels(
                original_model_cm, range(len(full_data_indeces))
            )

          for idx in full_data_indeces:
            for rank, label in enumerate(sorted_softmaxes_keys[idx]):
                if rank < args.ranks:
                    model_cm[label, idx] = calibrators_per_label[label][rank].predict_proba(model_cm[label, idx])[0]
      
      elif args.calibrate == "top-k-confidence":

        calibrators_per_label = top_k_confidence_calibration(
            cal_indeces, original_model_cm, Y, n_bins=args.calibration_bins, ranks=args.ranks
        )

        _, sorted_softmaxes_keys = get_sorted_probabilities_and_labels(
                original_model_cm, range(len(full_data_indeces))
              )

        for idx in full_data_indeces:
          for rank, label in enumerate(sorted_softmaxes_keys[idx]):
              if rank < args.ranks:
                  model_cm[label, idx] = calibrators_per_label[rank].predict_proba(model_cm[label, idx])[0]
  

      # We save the ground truth labels for this split
      key_gt = (run_id, model, noise)
      if key_gt not in all_ground_truth:
        Y_test_images = {image_names[image_idx]: y_true for image_idx, y_true in zip(test_indeces, Y_test)}
        all_ground_truth[key_gt] = deepcopy(Y_test_images)

      # Get the images from calibration we want to use to compute the user_cm
      calibration_images_names = set([i for k, i in enumerate(image_names) if k in cal_indeces])
      assert len(calibration_images_names) == len(Y_cal)

      # Add complete matrix
      user_cm_difficulties = {
          -1: extract_human_data(NOISE_LEVEL=noise,
                                 selected_images=calibration_images_names,
                                 label_mapping=label_mapping,
                                 humans_ids=human_ids)
      }

      # Add complete matrix
      testing_images_names = set([i for k, i in enumerate(image_names) if k in test_indeces])
      assert len(testing_images_names) == len(Y_test)
      user_cm_difficulties_testing = {
          -1: extract_human_data(NOISE_LEVEL=noise,
                                 selected_images=testing_images_names,
                                 label_mapping=label_mapping,
                                 humans_ids=human_ids)
      }

      # If we want to use a more expressive MNL, then we have to extract different
      # confusion matrices for each subset of images. Otherwise, the difficulties are
      # set as as None and we use the confusion matrix computed over all images.
      difficulties = None
      if args.expressive_mnl:
          difficulties = extract_image_difficulties(
              NOISE_LEVEL=noise,
              labels_encoding=label_mapping,
              image_labels=image_names
          )

          for difficulty in np.unique(difficulties):
              images_of_this_difficulty = set([i for k, i in enumerate(image_names) if difficulties[k] == difficulty])
              images_of_this_difficulty = images_of_this_difficulty.intersect(calibration_images_names) # keep only calibration images
              user_cm = extract_human_data(NOISE_LEVEL=noise, label_mapping=label_mapping,
                                          selected_images=images_of_this_difficulty,
                                          humans_ids=human_ids)
              user_cm_difficulties[difficulty] = deepcopy(user_cm)
              del user_cm
      
      # Append the generated confusion matrices
      if rank_mpi == 0:
        all_confusion_matrices[model][noise][run_id].append(
          deepcopy(user_cm_difficulties)
        )
        all_confusion_matrices_testing[model][noise][run_id].append(
          deepcopy(user_cm_difficulties_testing) 
        )

      if args.calibrate is not None:
        
        if not args.skip_brute_force:
          
          # Launch bruteforce if needed
          if rank_mpi == 0:
            
            # Define loop parameters
            total_iterations = len(test_indeces)
            chunk_size = total_iterations // size

            # Scatter iterations across processes
            for i in range(1, size):
                start = i * chunk_size
                end = start + chunk_size
                if i == size-1 and end < total_iterations:
                  end = total_iterations
                comm.send((start, end, test_indeces), dest=i)

            # Master process computation
            start = 0
            end = chunk_size
            prediction_sets_brute_force = _parallel_bruteforce(test_indeces, start, end, model_cm,
                                                              user_cms=user_cm_difficulties, difficulties=difficulties, verbose=True)

            # Gather results from other processes
            for i in range(1, size):
              prediction_sets_brute_force += comm.recv(source=i)

          else:
            start, end, test_indeces = comm.recv(source=0)
            result = _parallel_bruteforce(test_indeces, start, end, model_cm, user_cms=user_cm_difficulties, difficulties=difficulties, verbose=False)
            comm.send(result, dest=0)

        if rank_mpi == 0:
          # Compute all the prediction sets for both greedy and brute-force if needed
          # We do it again since we are calibrating
          prediction_sets_greedy = { k: [greedy(x_test, model_cm,
                                                user_cm_difficulties.get(difficulties[x_test] if difficulties is not None else -1),
                                                k,
                                                np.random.default_rng(2024)) for x_test in full_data_indeces] for k in K}
          prediction_sets_top_k = {k: [top_k(x_test,
                                            model_cm,
                                            k=k) for x_test in full_data_indeces] for k in top_k_sizes}
        
      if rank_mpi == 0:

        # Filter the greedy, brute-force and top-k data to use only the given indeces
        prediction_sets_brute_force_filtered = {k: [data.get(k) for _, data in enumerate(prediction_sets_brute_force)] for k in K} if not args.skip_brute_force else []
        prediction_sets_greedy_filtered = {k: [data for idx, data in enumerate(prediction_sets_greedy.get(k)) if 
                                              idx in X_test] for k in K}
        prediction_sets_top_k_filtered =  {k: [data for idx, data in enumerate(prediction_sets_top_k.get(k)) if idx in X_test] for k in top_k_sizes}



        # Build also the prediction sets for conformal prediction
        prediction_sets_conformal = {}
        for alpha in quantile_alphas:

          # Compute the quantile for the given alpha
          q_hat = quantile(X_cal, Y_cal, model_cm, alpha)
          
          # Build the prediction sets for this alpha
          prediction_sets_conformal[alpha] = [conformal_set(x_test, model_cm, q_hat) for x_test in X_test]

        # Build also the prediction sets for adaptive conformal prediction
        prediction_sets_conformal_APS = {}
        for alpha in quantile_alphas:

          # Compute the quantile for the given alpha
          q_hat = quantile_APS(X_cal, Y_cal, model_cm, alpha)
          
          # Build the prediction sets for this alpha
          # Since we correctly know P(Y|X), then, we can simply pick the top-k
          # elements until we reach the 1-alpha cut-off, and that is our
          # prediction set.
          prediction_sets_conformal_APS[alpha] = [conformal_set_APS(x_test, model_cm, q_hat) for x_test in X_test]

        # Compute the human accuracy
        prediction_sets_human_alone = [list(range(total_labels)) for _ in range(len(Y_test))]
        acc_user_alone, _ = user_accuracy(Y_test, prediction_sets_human_alone, user_cm_difficulties,
                                          difficulties[test_indeces] if difficulties is not None else None)    

        # Second, compute the model accuracy, again as a benchmark to check consistency
        model_accuracy_alone = 0
        for x,y in zip(X_test, Y_test):
          y_predicted = np.argmax(model_cm[:, x])
          model_accuracy_alone += int(y == y_predicted)
        model_accuracy_alone /= len(X_test)

        # Save the human, model and ECE results
        full_evals.append(
          [run_id, model, noise, "Human (Alone)", acc_user_alone, 0, 0]
        )
        full_evals.append(
          [run_id, model, noise, "Model", model_accuracy_alone, 0, 0]
        )
        full_evals.append(
          [run_id, model, noise, "ECE", ECE(np.array(Y), model_cm), 0, 0]
        )

        # Create dummy prediction sets for the human/model alone
        prediction_sets = [list(range(total_labels)) for _ in Y]

        key = (run_id, model, noise, f"Human (Alone)")
        if key not in all_prediction_sets:
            all_prediction_sets[key] = deepcopy(prediction_sets)
        
        key = (run_id, model, noise, f"Model")
        if key not in all_prediction_sets:
            all_prediction_sets[key] = deepcopy(prediction_sets)
        
        key = (run_id, model, noise, f"ECE")
        if key not in all_prediction_sets:
            all_prediction_sets[key] = deepcopy(prediction_sets)

        if not args.skip_brute_force:
          for k in K:

            # Create the new prediction sets and compute their average length
            prediction_sets = prediction_sets_brute_force_filtered.get(k)

            # Evaluate the prediction set
            avg_len_psets, acc_user, emp_cov = evaluate_prediction_sets(prediction_sets, Y_test, user_cm_difficulties, difficulties[test_indeces] if difficulties is not None else None)

            # For each alpha, add the same value since they do not change
            full_evals.append(
                [run_id, model, noise, f"BF (k={k})", acc_user, avg_len_psets, emp_cov]
            )

            # Add prediction sets to the full datasets
            key = (run_id, model, noise, f"BF (k={k})")
            if key not in all_prediction_sets:
              prediction_sets_images = {image_names[image_idx]: deepcopy(pset) for image_idx, pset in zip(test_indeces, prediction_sets)}
              all_prediction_sets[key] = deepcopy(prediction_sets_images)

        for k in K:

          # Create the new prediction sets and compute their average length
          prediction_sets = prediction_sets_greedy_filtered.get(k)
          
          # Evaluate the prediction set
          avg_len_psets, acc_user, emp_cov = evaluate_prediction_sets(prediction_sets, Y_test, user_cm_difficulties, difficulties[test_indeces] if difficulties is not None else None)

          # For each alpha, add the same value since they do not change
          full_evals.append(
              [run_id, model, noise, f"Greedy (k={k})", acc_user, avg_len_psets, emp_cov]
          )

          # Add prediction sets to the full datasets
          key = (run_id, model, noise, f"Greedy (k={k})")
          if key not in all_prediction_sets:
            prediction_sets_images = {image_names[image_idx]: deepcopy(pset) for image_idx, pset in zip(test_indeces, prediction_sets)}
            all_prediction_sets[key] = deepcopy(prediction_sets_images)

        for k in top_k_sizes:

          # Create the new prediction sets and compute their average length
          prediction_sets = prediction_sets_top_k_filtered.get(k)
          
          # Evaluate the prediction set
          avg_len_psets, acc_user, emp_cov = evaluate_prediction_sets(prediction_sets, Y_test, user_cm_difficulties, difficulties[test_indeces] if difficulties is not None else None)

          # For each alpha, add the same value since they do not change
          full_evals.append(
              [run_id, model, noise, f"Top-K (k={k})", acc_user, avg_len_psets, emp_cov]
          )

          # Add prediction sets to the full datasets
          key = (run_id, model, noise, f"Top-K (k={k})")
          if key not in all_prediction_sets:
            prediction_sets_images = {image_names[image_idx]: deepcopy(pset) for image_idx, pset in zip(test_indeces, prediction_sets)}
            all_prediction_sets[key] = deepcopy(prediction_sets_images)

        all_CP_evaluations = []
        all_CP_prediction_sets = []
        for alpha in quantile_alphas:
            
            prediction_sets = prediction_sets_conformal.get(alpha)
            
            # Evaluate the prediction set
            avg_len_psets, acc_user, emp_cov = evaluate_prediction_sets(prediction_sets, Y_test, user_cm_difficulties, difficulties[test_indeces] if difficulties is not None else None)

            # Save the current prediction set
            all_CP_prediction_sets.append(
                deepcopy(prediction_sets)
            )

            # Append the results
            all_CP_evaluations.append(
                [run_id, model, noise, "CP", acc_user, avg_len_psets, emp_cov]
            )

            # Append the prediction sets for each alpha
            prediction_sets_images = {image_names[image_idx]: deepcopy(pset) for image_idx, pset in zip(test_indeces, prediction_sets)}
            if (run_id, model, noise) not in all_prediction_sets_alpha_values["CP"]:
              all_prediction_sets_alpha_values["CP"][(run_id, model, noise)] = {}
            all_prediction_sets_alpha_values["CP"][(run_id, model, noise)][alpha] = deepcopy(prediction_sets_images)
        
        # Append to the full evals only the best CP evaluations
        best_result = max(all_CP_evaluations, key=lambda x:x[4])
        best_result_index = all_CP_evaluations.index(best_result)
        full_evals.append(
            deepcopy(best_result)
        )

        # Save the prediction set
        key = tuple(best_result[0:4])
        if key not in all_prediction_sets:
          prediction_sets_images = {image_names[image_idx]: deepcopy(pset) for image_idx, pset in zip(test_indeces, all_CP_prediction_sets[best_result_index])}
          all_prediction_sets[key] = deepcopy(prediction_sets_images)

        all_CP_APS_evaluations = []
        all_CP_APS_prediction_sets = []
        for alpha in quantile_alphas:
            
            # Extract the prediction set checking that it is not None
            prediction_sets = prediction_sets_conformal_APS.get(alpha, None)
            assert prediction_sets != None
            
            # Evaluate the prediction set
            avg_len_psets, acc_user, emp_cov = evaluate_prediction_sets(prediction_sets, Y_test, user_cm_difficulties, difficulties[test_indeces] if difficulties is not None else None)

            # Save the current prediction set
            all_CP_APS_prediction_sets.append(
                deepcopy(prediction_sets)
            )

            # Append the results
            all_CP_APS_evaluations.append(
                [run_id, model, noise, "APS", acc_user, avg_len_psets, emp_cov]
            )

            # Append the prediction sets for each alpha
            prediction_sets_images = {image_names[image_idx]: deepcopy(pset) for image_idx, pset in zip(test_indeces, prediction_sets)}
            if (run_id, model, noise) not in all_prediction_sets_alpha_values["APS"]:
              all_prediction_sets_alpha_values["APS"][(run_id, model, noise)] = {}
            all_prediction_sets_alpha_values["APS"][(run_id, model, noise)][alpha] = deepcopy(prediction_sets_images)
        
        # Append to the full evals only the best CP evaluations
        best_result = max(all_CP_APS_evaluations, key=lambda x:x[4])
        best_result_index = all_CP_APS_evaluations.index(best_result)
        full_evals.append(
            deepcopy(best_result)
        )

        # Add also the prediction sets
        key = tuple(best_result[0:4])
        if key not in all_prediction_sets:
          prediction_sets_images = {image_names[image_idx]: deepcopy(pset) for image_idx, pset in zip(test_indeces, all_CP_APS_prediction_sets[best_result_index])}
          all_prediction_sets[key] = deepcopy(prediction_sets_images)      

# Save everything to disk once it is done
if rank_mpi == 0:
  # Filename
  filename = f"real_data-{args.model_epochs}-{args.calibration_size}-{args.calibrate}-{args.points_per_bins if args.calibrate else '0'}-{args.calibration_bins if args.calibrate else '0'}-{args.ranks if args.calibrate else '0'}-{args.expressive_mnl}-{'all' if args.expressive_humans < 0 else args.expressive_humans}"

  # Save the prediction sets
  import pickle
  pickle.dump(all_prediction_sets, open(f'{filename}-prediction_sets.pickle', 'wb'))
  pickle.dump(all_ground_truth, open(f'{filename}-ground_truth.pickle', 'wb'))
  pickle.dump(all_confusion_matrices, open(f'{filename}-confusion_matrices.pickle', 'wb'))
  pickle.dump(all_confusion_matrices_testing, open(f'{filename}-confusion_matrices_full.pickle', 'wb'))
  pickle.dump(all_prediction_sets_alpha_values, open(f'{filename}-prediction_sets_alphas.pickle', "wb"))

  full_evals = pd.DataFrame(full_evals, columns = ["run_id", "model", "noise", "method", "accuracy", "length", "emp_cov"])
  full_evals.to_csv(f"{filename}-results.csv")