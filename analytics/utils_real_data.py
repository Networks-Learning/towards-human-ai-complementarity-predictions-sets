import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split

def split_dataset(run_no: int, label_mapping: dict, calibration_size=0.1):
    """Splitting function used by Straitouri et al. to generate the dataset for their experiments.

    See the original file at https://github.com/Networks-Learning/counterfactual-prediction-sets/blob/main/utils.py

    :param run_no: id the of run to split the dataset
    :type run_no: int
    :return: the splitted dataset (both for train and test)
    :rtype: tuple
    """

    model_path = f"data/ImageNet-16H/real_human_eval/vgg19_epoch10_preds.csv"
    model_predictions = pd.read_csv(model_path)

    # Image names and true labels
    x = model_predictions['image_name'].to_numpy()
    y_strings = model_predictions['category'].to_numpy()

    # Map label strings to ints
    label_to_int_mapping_path = f"submission/data/ImageNet-16H/real_human_eval/label_to_int_mapping.json"
    with open(label_to_int_mapping_path, 'rt') as f:
        label_to_int_mapping = json.load(f)
    y = np.array([label_to_int_mapping[label] for label in y_strings])
            
    # Get the calibration set used by them
    # See https://github.com/Networks-Learning/counterfactual-prediction-sets/blob/main/config.py
    X_train, X_cal, y_train, y_cal = train_test_split(
        x, y, test_size=calibration_size, random_state=42+run_no)
     
    return X_train, X_cal, y_train, y_cal