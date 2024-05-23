import pandas as pd
import numpy as np

def extract_human_difficulties(
    human_data_path: str="data/ImageNet-16H/human_only_classification_6per_img_export.csv",
    NOISE_LEVEL: int=125,
    labels_encoding: dict= None   
):
    # Read the human data
    human_data = pd.read_csv(human_data_path)

    # Filter the data based on this configuration
    # noise_level ([80  95 110 125])
    condition = human_data.noise_level == NOISE_LEVEL
    human_data = human_data[condition]

    # Convert the classification with the right index
    for column in ["image_category", "participant_classification"]:
        human_data[column] = human_data[column].apply(lambda x : labels_encoding.get(x, -1))

    # Pick only those instances for which we have a positive label
    assert len(human_data[human_data.image_category == -1]) == 0
    assert len(human_data[human_data.participant_classification == -1]) == 0
    human_data = human_data[human_data.image_category >= 0]
    human_data = human_data[human_data.participant_classification >= 0]

    humans_ids = human_data.worker_id.unique()

    # Iterate over all of them and compute the confusion matrix and the human classification
    cm_per_human = np.zeros(len(humans_ids))

    # Convert the labels to probabilities
    for idx_human, worker_id in enumerate(humans_ids):

        human_data_per_image = human_data[human_data.worker_id == worker_id]        

        for _, row in human_data_per_image.iterrows():
            current_true_label = row.image_category
            assert row.participant_classification < 16
            cm_per_human[idx_human] += (row.participant_classification == current_true_label)
        
        # Get probabilities
        cm_per_human[idx_human] /= len(human_data_per_image)

    # This snippet was adapted from:
    # https://github.com/Networks-Learning/improve-expert-predictions-conformal-prediction/blob/main/utils.py#L46
    y_groups = np.zeros(len(cm_per_human), dtype='int')

    # Thresholds defining the difficulty levels
    threshold_hard = np.quantile(cm_per_human, 0.25)
    threshold_easy = np.quantile(cm_per_human, 0.5)

    for i,acc in enumerate(cm_per_human):
        if acc > threshold_easy:
            # Very good humans
            y_groups[i] = 0
        elif acc <= threshold_hard:
            # Very bad humans
            y_groups[i] = 2
        else:
            y_groups[i] = 1
    
    return humans_ids, y_groups
    

def extract_image_difficulties(
    human_data_path: str="data/ImageNet-16H/human_only_classification_6per_img_export.csv",
    NOISE_LEVEL: int=125,
    image_labels: list=None,
    labels_encoding: dict= None
):
    # Read the human data
    human_data = pd.read_csv(human_data_path)

    # Filter the data based on this configuration
    # noise_level ([80  95 110 125])
    condition = human_data.noise_level == NOISE_LEVEL
    human_data = human_data[condition]

    # Convert the classification with the right index
    for column in ["image_category", "participant_classification"]:
        human_data[column] = human_data[column].apply(lambda x : labels_encoding.get(x, -1))

    # Pick only those instances for which we have a positive label
    assert len(human_data[human_data.image_category == -1]) == 0
    assert len(human_data[human_data.participant_classification == -1]) == 0
    human_data = human_data[human_data.image_category >= 0]
    human_data = human_data[human_data.participant_classification >= 0]

    # Iterate over all of them and compute the confusion matrix and the human classification
    cm_per_instance = np.zeros((len(image_labels), 16))

    # True labels
    y = []

    # Convert the labels to probabilities
    for idx_image, image_name in enumerate(image_labels):

        human_data_per_image = human_data[human_data.image_name == image_name]
        assert len(human_data_per_image) >= 6, human_data_per_image
        assert len(human_data_per_image.image_category.unique()) == 1
        
        current_true_label = human_data_per_image.image_category.unique()[0]

        for _, row in human_data_per_image.iterrows():
            assert row.participant_classification < 16
            cm_per_instance[idx_image, row.participant_classification] += 1
        
        # Get probabilities
        cm_per_instance[idx_image, :] /= np.sum(cm_per_instance[idx_image, :])

        y.append(current_true_label)

    # Extract the ground truth labels
    y = np.array(y)

    # This snippet was adapted from:
    # https://github.com/Networks-Learning/improve-expert-predictions-conformal-prediction/blob/main/utils.py#L46
    acc_per_sample = np.zeros(len(cm_per_instance))
    y_groups = np.zeros(len(cm_per_instance), dtype='int')
    
    # Average accuracy of experts for each sample
    for i,l in enumerate(y):
        acc_per_sample[i] = cm_per_instance[i][l]

    # Thresholds defining the difficulty levels
    threshold_hard = np.quantile(acc_per_sample, 0.25)
    threshold_easy = np.quantile(acc_per_sample, 0.5)

    for i,acc in enumerate(acc_per_sample):
        if acc > threshold_easy:
            # Easy samples
            y_groups[i] = 0
        elif acc <= threshold_hard:
            # Hard samples
            y_groups[i] = 2
        else:
            # Medium difficulty samples
            y_groups[i] = 1
    
    return y_groups
    

def extract_human_data_new(
        human_data_path: str="data/ImageNet-16H/human_only_classification_real.csv",
        labels_encoding: dict= None,
        selected_images: set=None,
        image_names_and_labels: list=None
):
    # Read the human data
    human_data = pd.read_csv(human_data_path)

    # Filter the data considering only the full set size
    human_data = human_data[human_data["set"] == 16]
    
    # Convert the prediction into a numeric form
    for column in ["prediction"]:
        human_data[column] = human_data[column].apply(lambda x : labels_encoding.get(x, -1))
    
    assert len(human_data[human_data.prediction == -1]) == 0
    human_data = human_data[human_data.prediction >= 0]

    # Get all the image names and sort them
    images_identifiers = human_data.image_name.unique().tolist()
    images_identifiers = sorted(images_identifiers)

    # Filter the image identifies
    # This will help us for the difficulties
    if selected_images is not None:
        images_identifiers = [img for img in images_identifiers if img in selected_images]

    # Iterate over all of them and compute the confusion matrix and the human classification
    cm_per_instance = np.zeros((len(images_identifiers), 16))

    # True labels
    y = []

    # Convert the labels to probabilities
    for idx_image, image_name in enumerate(images_identifiers):

        human_data_per_image = human_data[human_data.image_name == image_name]
        assert len(human_data_per_image) >= 5, human_data_per_image
        
        # Get true label
        current_true_label = image_names_and_labels.get(image_name)

        for _, row in human_data_per_image.iterrows():
            assert row.prediction < 16
            cm_per_instance[idx_image, row.prediction] += 1
        
        y.append(current_true_label)

    # Extract the ground truth labels
    y = np.array(y)

    # Generate the correct confusion matrix
    cm = np.zeros(shape=(16, 16))
    for i in range(16):
        idx = np.argwhere(y == i).flatten()

        # Accuracy for the current label
        label_accuracy = np.sum(cm_per_instance[idx, :], axis=0)

        # We normalize it
        label_accuracy /= np.sum(label_accuracy)

        # Set the label accuracy for the true label
        cm[:, i] = label_accuracy

    return cm

def extract_human_data(human_data_path: str="data/ImageNet-16H/human_only_classification_6per_img_export.csv",
                       NOISE_LEVEL: int=125,
                       label_mapping: dict= None,
                       selected_images: set=None,
                       humans_ids: list=None,
):


    # Read the human data
    human_data = pd.read_csv(human_data_path)

    # If the human ids are not none, then we just use those
    # for generating the confusion matrix
    if humans_ids is not None:
        human_data = human_data[human_data.worker_id.isin(humans_ids)]

    # Filter the data based on this configuration
    # noise_level ([80  95 110 125])
    condition = human_data.noise_level == NOISE_LEVEL
    human_data = human_data[condition]

    # Convert the labels into one-hot encoding
    image_labels = sorted(human_data.image_category.unique().tolist())
    assert len(image_labels) == 16, image_labels # Check if they are all correct
    labels_encoding = {label: k for k, label in enumerate(image_labels)}
    
    # Overwrite labels encoding if needed
    labels_encoding = labels_encoding if label_mapping is None else label_mapping

    # Convert the classification with the right index
    for column in ["image_category", "participant_classification"]:
        human_data[column] = human_data[column].apply(lambda x : labels_encoding.get(x, -1))

    # Pick only those instances for which we have a positive label
    assert len(human_data[human_data.image_category == -1]) == 0
    assert len(human_data[human_data.participant_classification == -1]) == 0
    human_data = human_data[human_data.image_category >= 0]
    human_data = human_data[human_data.participant_classification >= 0]

    # Get all the image names and sort them
    images_identifiers = human_data.image_name.unique().tolist()
    images_identifiers = sorted(images_identifiers)

    # Filter the image identifies
    # This will help us for the difficulties
    if selected_images is not None:
        images_identifiers = [img for img in images_identifiers if img in selected_images]

    # Iterate over all of them and compute the confusion matrix and the human classification
    cm_per_instance = np.zeros((len(images_identifiers), 16))

    # True labels
    y = []

    # Convert the labels to probabilities
    for idx_image, image_name in enumerate(images_identifiers):

        human_data_per_image = human_data[human_data.image_name == image_name]
        assert len(human_data_per_image) >= 6 or humans_ids is not None, human_data_per_image
        assert len(human_data_per_image.image_category.unique()) == 1
        
        current_true_label = human_data_per_image.image_category.unique()[0]

        for _, row in human_data_per_image.iterrows():
            assert row.participant_classification < 16
            cm_per_instance[idx_image, row.participant_classification] += 1
        
        y.append(current_true_label)

    # Extract the ground truth labels
    y = np.array(y)

    # Generate the correct confusion matrix
    cm = np.zeros(shape=(16, 16))
    for i in range(16):
        idx = np.argwhere(y == i).flatten()

        # Accuracy for the current label
        label_accuracy = np.sum(cm_per_instance[idx, :], axis=0)

        # We normalize it (if everything is zero, we keep zero)
        label_accuracy /= np.sum(label_accuracy)
        cm[:, i] = label_accuracy

    return cm

def extract_model_data(model_data_path: str="data/ImageNet-16H/hai_epoch10_model_preds_max_normalized.csv",
                       NOISE_LEVEL: int=125,
                       MODEL_NAME: str="vgg19",
                       random_generator: np.random.RandomState = np.random.RandomState(2024)):

    # Read the machine data
    machine_data = pd.read_csv(model_data_path)

    # Filter by noise level and model name
    condition = (machine_data.noise_level == NOISE_LEVEL) & (machine_data.model_name == MODEL_NAME)
    machine_data = machine_data[condition]

    # Full dataset classifier
    full_data_classifier = []
    Y = []

    # Convert the labels into one-hot encoding
    image_labels = sorted(machine_data.category.unique().tolist())
    assert len(image_labels) == 16, image_labels # Check if they are all correct
    labels_encoding = {label: k for k, label in enumerate(image_labels)}
    inverse_label_encoding = {k: label for label, k in labels_encoding.items()}

    # Convert the classification with the right index
    for column in ["category", "model_pred"]:
        machine_data[column] = machine_data[column].apply(lambda x : labels_encoding.get(x, -1))

    # Pick only those instances for which we have a positive label
    # Check if there are any which are missclassified
    assert len(machine_data[machine_data.category == -1]) == 0
    machine_data = machine_data[machine_data.category >= 0]

    # Generate the correct config file
    machine_image_identifiers = sorted(machine_data.image_name.unique().tolist())
    for _, image_name in enumerate(machine_image_identifiers):

        image_data = machine_data[machine_data.image_name == image_name]
        assert len(image_data == 1)

        # Extract ground truth data
        ground_truth_label = image_data.category.tolist()[0]

        labels_softmaxes = []
        for i in range(16):
            labels_softmaxes.append(image_data[inverse_label_encoding.get(i)].tolist()[0])
                
        full_data_classifier.append(
            [image_name] + [ground_truth_label] + labels_softmaxes
        )

    # Shuffle the machine dataset
    random_generator.shuffle(full_data_classifier)

    # Split again
    images_names = [x[0] for x in full_data_classifier]
    Y = [int(x[1]) for x in full_data_classifier]
    sofmaxes = np.array([x[2:] for x in full_data_classifier])

    return sofmaxes.T, Y, images_names, labels_encoding, inverse_label_encoding
