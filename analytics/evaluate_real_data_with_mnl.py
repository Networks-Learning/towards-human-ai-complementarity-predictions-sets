from utils.general import quantile, conformal_set
from analytics.utils_real_data import split_dataset
from utils.general import user_accuracy
import numpy as np

from copy import deepcopy

import json

from tqdm import tqdm

from data.preprocess import extract_human_data_new, extract_model_data

if __name__ == "__main__":

    # Get the best alpha
    with open("data/ImageNet-16H/real_human_eval/deploy_avg_acc_se_alphas_1.json") as fp:
        deploy_avg_results_dict = json.load(fp)

    # Convert alphas into list
    deploy_avg_results = [ (alpha, avg["avg"]) for alpha, avg in deploy_avg_results_dict.items()]

    # Hardcoded best alpha value 
    best_alpha = 0.15702479338842978

    assert best_alpha == float(max(deploy_avg_results, key=lambda x: x[1])[0])

    # Calibration run id which is attached to the best alpha
    run_id = 1

    # Path to the model predictions
    model_finetune_map = "data/ImageNet-16H/real_human_eval/vgg19_epoch10_preds.csv"

    # We extract the model data
    original_model_cm, Y, images_names, labels_mapping, inverse_mapping = extract_model_data(
                                          NOISE_LEVEL=110,
                                          MODEL_NAME="vgg19",
                                          model_data_path=model_finetune_map,
                                          random_generator=np.random.default_rng(2024))

    # Get the calibration and splitting data
    X_test, X_cal, _, _ = split_dataset(run_id, labels_mapping, 0.1)
    Y = np.array(Y)

    assert len(X_cal == 120)

    # We get user data
    selected_images = [img_name for img_name in images_names if img_name in X_test]
    user_cm_difficulties = {
        -1: extract_human_data_new(labels_encoding=labels_mapping,
                                   #selected_images=selected_images,
                                    image_names_and_labels={n:l for n,l in zip(images_names, Y)})
    }

    print("User accuracy alone: ", np.mean([user_cm_difficulties.get(-1)[x,x] for x in range(16)]))

    # Get the idx of the image names we need for calibration
    calibration_idx = [idx for idx, image_name in enumerate(images_names) if image_name in X_cal]
    testing_idx = [idx for idx, image_name in enumerate(images_names) if image_name in X_test]

    calibration_idx = sorted(calibration_idx)
    training_idx = sorted(testing_idx)

    # Get the corresponding quantile
    q_hat = quantile(calibration_idx, Y[calibration_idx], original_model_cm, best_alpha)

    # Generate conformal sets, given the current quantile
    prediction_sets = [conformal_set(x_test, original_model_cm, q_hat, return_full_set=True) for x_test in testing_idx]

    # Accuracy
    acc, std = user_accuracy(Y[testing_idx], prediction_sets, user_cm_difficulties)

    print("Best alpha:", best_alpha)
    print("CP (with best alpha):", acc, std)
    print("Percentage of >1 size psets: ", len([idx for idx, p in enumerate(prediction_sets) if len(p) > 1])/len(testing_idx))
    print()

    prediction_sets_with_alphas = {}

    best_cp_accuracy = (-np.inf, None, None)
    for best_alpha in deploy_avg_results_dict.keys():

        best_alpha = float(best_alpha)

        # Create the prediction sets for this alpha
        prediction_sets_with_alphas[best_alpha] = {}

        # Get the corresponding quantile
        q_hat = quantile(calibration_idx, Y[calibration_idx], original_model_cm, best_alpha)

        # Generate conformal sets, given the current quantile
        prediction_sets = [conformal_set(x_test, original_model_cm, q_hat, return_full_set=True) for x_test in testing_idx]

        for p in prediction_sets:
            assert len(p) > 0

        # Append each prediction sets to its image name
        prediction_sets_with_alphas[best_alpha]["images"] = {
            images_names[idx]: { "conformal": [inverse_mapping.get(l) for l in pset],
              "ground_truth": inverse_mapping.get(Y[idx])
            } for idx, pset in zip(testing_idx, prediction_sets)
        }
        prediction_sets_with_alphas[best_alpha]["quantile"] = q_hat

        # Accuracy
        acc, std = user_accuracy(Y[testing_idx], prediction_sets, user_cm_difficulties)
        
        prediction_sets_with_alphas[best_alpha]["accuracy"] = acc
        prediction_sets_with_alphas[best_alpha]["std"] = std

        if acc > best_cp_accuracy[0]:
            best_cp_accuracy = (acc, deepcopy(prediction_sets), best_alpha, std)
        
    print("Best Alpha: ", best_cp_accuracy[2])
    print("CP:", best_cp_accuracy[0], best_cp_accuracy[3])
    prediction_sets = best_cp_accuracy[1]


    with open("data/ImageNet-16H/real_human_eval/MNL_accuracy.json", "w") as fp:
        json.dump(
            prediction_sets_with_alphas,
            fp,
            indent=4,
            sort_keys=True
        )
