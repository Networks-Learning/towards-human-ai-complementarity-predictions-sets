import pandas as pd
import numpy as np
import pickle

from argparse import ArgumentParser

from copy import deepcopy

from data.preprocess import extract_human_data, extract_model_data, extract_image_difficulties, extract_human_difficulties
from utils.general import multinomial_logit, user_accuracy
from tqdm import tqdm

from analytics.utils_plot import command_line

def calc(pset, cm, probabilities):
    assert len(probabilities) == 16
    total = 0
    for y in pset:
        denom = 0
        for i in pset:
            denom += cm[i,y]
        total += probabilities[y] * (cm[y,y] / denom)
    return total

parser = command_line()
args = parser.parse_args()

data = pickle.load(open(args.file[0], 'rb'))
gt_filename = "-".join(args.file[0].split("-")[:-1]) + "-ground_truth.pickle"
cm_filename = "-".join(args.file[0].split("-")[:-1]) + "-confusion_matrices.pickle"
ground_truth = pickle.load(open(gt_filename, 'rb'))
users_cms = pickle.load(open(cm_filename, 'rb'))
epoch = args.file[0].split("-")[1].replace("epoch", "")

np.random.seed(2024)

MODELS = ["vgg19"]
NOISE_LEVELS = [80, 95, 110, 125]
METHODS = ['CP', 'APS', 'BF (k=16)', f'Greedy (k=16)']

complete_accuracy = []

for noise in NOISE_LEVELS:

    model_cm, _, images_name, labels_mapping, _ = extract_model_data(
        model_data_path=f"data/ImageNet-16H/hai_epoch{epoch}_model_preds_max_normalized.csv",
        NOISE_LEVEL=noise,
        MODEL_NAME="vgg19",
        random_generator=np.random.default_rng(2024))

    # Split everything for expressive humans
    human_ids = None
    if args.expressive_humans != -1:
       human_ids, group_humans = extract_human_difficulties(NOISE_LEVEL=noise, labels_encoding=labels_mapping)
       human_ids = [ x for k, x in enumerate(human_ids) if group_humans[k] == args.expressive_humans]
    
    # If we want to use a more expressive MNL, then we have to extract different
    # confusion matrices for each subset of images. Otherwise, the difficulties are
    # set as as None and we use the confusion matrix computed over all images.
    difficulties = None
    if args.expressive_mnl:
        difficulties = extract_image_difficulties(
            NOISE_LEVEL=noise,
            labels_encoding=labels_mapping,
            image_labels=images_name
        )     

    for model in MODELS:
        for run_id in tqdm(range(10), desc=f"{model} (Noise={noise})"):
            for model_name in METHODS:

                k = (run_id, model, noise, model_name)
                pset_key = (run_id, model, noise)

                # Add complete matrix
                user_cm_difficulties = users_cms.get(model).get(noise).get(run_id)[0]

                # Skip missing keys 
                if ground_truth.get(pset_key, None) is None or data.get(k, None) is None:
                    continue

                # Rebuilt the correct values, since they are now dictionaries
                X, Y, softmaxes = [], [], []
                for key in ground_truth.get(pset_key):
                    index_of_image = images_name.index(key)

                    if args.difficulty != -1:
                        if difficulties[index_of_image] != args.difficulty:
                            continue

                    softmaxes.append(index_of_image)
                    X.append(data[k].get(key))
                    Y.append(ground_truth.get(pset_key).get(key))
                    assert X[-1] is not None and Y[-1] is not None
                
                # If we want to use the score given by the optimization objective, 
                # the we use the full MNL mode, since we cannot know the difficulties at
                # test time.
                if args.use_g_score:
                    acc = []
                    for x,y, idx in zip(X, Y, softmaxes):
                        acc.append(calc(x, user_cm_difficulties.get(-1 if difficulties is None else difficulties[idx]), model_cm[:, idx]))
                    acc = np.mean(acc)
                else:
                    acc, std = user_accuracy(Y, X, user_cm_difficulties, difficulties[softmaxes] if difficulties is not None else None)

                # We append the accuracy to the complete values
                complete_accuracy.append([model, noise, model_name, acc])

# Generate a dataframe with all the results
complete_accuracy = pd.DataFrame(complete_accuracy, columns=["Model", "Noise", "Method", "Accuracy"])

# For each noise level, we return the accuracy
print()
for method in METHODS:
    vals = complete_accuracy[complete_accuracy.Method == method]
    value = f"{method}\t"
    for conf in sorted(complete_accuracy.Noise.unique()):
        vals2 = vals[(vals.Noise == conf)]
        for model in sorted(complete_accuracy.Model.unique()):
            vals3 = vals2[vals2.Model == model][["Method", "Accuracy"]]
            #assert(len(vals3) == 10), (conf, method, model)
            g = vals3.groupby("Method").mean("Accuracy")
            g = g.reset_index()
            g2 = vals3.groupby("Method").std()
            g2 = g2.reset_index()
            # Method  Human  Model  Accuracy
            for row, std in zip(g.values.tolist(), g2.values.tolist()):
                value += f"${round(row[1], 3):.3f} \,\scriptstyle\pm {round(std[1], 3):.3f}$ \t"
    print(value)