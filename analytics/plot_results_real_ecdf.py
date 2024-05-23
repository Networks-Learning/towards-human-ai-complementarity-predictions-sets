import pandas as pd
import numpy as np
import pickle

import seaborn as sns
import matplotlib.pyplot as plt

from data.preprocess import extract_model_data, extract_image_difficulties
from tqdm import tqdm

from analytics.utils_plot import command_line
sns.set(font_scale=1.0,
        style="ticks",
        rc={
        "text.usetex": True,
        'text.latex.preamble': r'\usepackage{amsfonts}',
        "font.family": "serif",
    })

def calc(pset, cm, probabilities):
    total = 0
    for y in pset:
        denom = 0
        for i in pset:
            denom += cm[i,y]
        total += probabilities[y] * (cm[y,y] / denom)
    return total

def get_ecdf(probabilities, bins=1000):
    grid_values = np.arange(0,1, 1/bins) 
    temp_accuracy = sorted(probabilities)
    accuracy_per_bin = np.digitize(temp_accuracy, grid_values, right=False)
    bin_counts = np.bincount(accuracy_per_bin)[:-1]
    events = np.cumsum(bin_counts)/len(temp_accuracy)
    return 1-events, grid_values

def export_legend(legend, filename="legend.pdf", expand=[-5,-5,5,5]):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, format="pdf", dpi=400, bbox_inches=bbox)

parser = command_line()
args = parser.parse_args()

CP_accuracies = []
greedy_accuracies = []
bf_accuracies = []

# Extract the label mapping once
_, _, images_names, labels_mapping, _ = extract_model_data(NOISE_LEVEL=110, MODEL_NAME="vgg19")

complete_accuracy = []

for file in args.file:

    data = pickle.load(open(file, 'rb'))
    gt_filename = "-".join(file.split("-")[:-1]) + "-ground_truth.pickle"
    cm_filename = "-".join(file.split("-")[:-1]) + "-confusion_matrices.pickle"
    ground_truth = pickle.load(open(gt_filename, 'rb'))
    users_cms = pickle.load(open(cm_filename, 'rb'))
    epoch = int(file.split("-")[1].replace("epoch", ""))

    np.random.seed(2024)

    MODELS = ["vgg19"]
    NOISE_LEVELS = [80, 95, 110, 125]
    METHODS = ['CP', 'APS', 'BF (k=16)', f'Greedy (k=16)']

    for noise in NOISE_LEVELS:
        
        model_cm, _, images_name, labels_mapping, _ = extract_model_data(
            model_data_path=f"data/ImageNet-16H/hai_{file.split('-')[1]}_model_preds_max_normalized.csv",
            NOISE_LEVEL=noise,
            MODEL_NAME="vgg19")

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

            greedy_accuracies_run = {
                "Greedy": [],
                "Naive": [],
                "Aps": [],
                "Brute Force": []
            }

            for run_id in tqdm(range(10), desc=f"{model} (Noise={noise}, Epoch={epoch})"):
                
                temp_accuracy = []
                
                probab_values = {
                    "Greedy": [],
                    "Naive": [],
                    "Aps": [],
                    "Brute Force": []
                }

                # Get all the accuracies for each method, for this run
                for model_name in METHODS:

                    k = (run_id, model, noise, model_name)
                    pset_key = (run_id, model, noise)

                    if ground_truth.get(pset_key, None) is None or data.get(k, None) is None:
                        continue

                    user_cm_difficulties = users_cms.get(model).get(noise).get(run_id)[0]

                    # Rebuilt the correct values, since they are now dictionaries
                    X, Y = [], []
                    for key in ground_truth.get(pset_key):
                        
                        if args.difficulty != -1:
                            if args.difficulty != difficulties[images_name.index(key)]:
                                continue

                        X.append((key, data[k].get(key)))
                        Y.append(ground_truth.get(pset_key).get(key))
                        assert X[-1] is not None and Y[-1] is not None

                    for idx, (y_true, (image_name,pset)) in enumerate(zip(Y, X)):

                        image_level = -1 if difficulties is None else difficulties[images_name.index(image_name)]
                        current_user_cm = user_cm_difficulties.get(image_level)

                        # We compute directly with the g-score if needed
                        if args.use_g_score:
                            probab = calc(
                                pset, current_user_cm, model_cm[:, images_name.index(image_name)]
                            )
                        else:
                            if y_true in pset:
                                denom = sum([current_user_cm[x, y_true] for x in pset])
                                probab = current_user_cm[y_true, y_true]/denom
                            else:
                                probab = 0                        
                        
                        if model_name == "Greedy (k=16)":
                            probab_values["Greedy"].append(probab)
                        elif model_name == "CP":
                            probab_values["Naive"].append(probab)
                        elif model_name == "BF (k=16)":
                            probab_values["Brute Force"].append(probab)
                        elif model_name == "APS":
                            probab_values["Aps"].append(probab)

                # Compute the eCDF given the probability values for this run
                # and append them to the list
                for method in ["Greedy", "Naive", "Aps", "Brute Force"]:
                    greedy_accuracies_run[method].append(
                        get_ecdf(probab_values[method])[0]
                    )
            
            # Check that, for each run id, we get the same results
            for method in ["Greedy", "Naive", "Aps", "Brute Force"]:
                assert len(greedy_accuracies_run[method]) == 10
                continue
        
        # For each method, append the final results for the complete data
        for method in ["Greedy", "Naive", "Aps", "Brute Force"]:
            grid_values = np.arange(0,1, 1/1000)
            for ecdf in greedy_accuracies_run[method]:
                for v, bin_id in zip(ecdf, grid_values):
                    complete_accuracy.append((epoch, noise, r"\textsc{"+f"{method}"+"}", v, bin_id))


complete_accuracy = pd.DataFrame(complete_accuracy, columns=["Epoch", r"$\omega$", "Method", "Empirical cCDF", "Empirical Success Probability"])

complete_accuracy = complete_accuracy[complete_accuracy.Method != "bf"]

counter = 0
for x in range(2):
    for j in range(2):

        tmp = complete_accuracy[complete_accuracy[r"$\omega$"] == NOISE_LEVELS[counter]]
        fig, ax = plt.subplots(1,1, figsize=(3, 2.5))

        g = sns.lineplot(
            data=tmp,
            x="Empirical Success Probability",
            y="Empirical cCDF",
            hue="Method",
            hue_order=[r"\textsc{Naive}", r"\textsc{Aps}", r"\textsc{Greedy}"],
            errorbar="se",
            linewidth=1,
            ax=ax,
            legend=False,
            zorder = -10
        )

        ax.set_rasterization_zorder(0)

        #legend = ax.legend(title=None, ncol=len(tmp["Method"].unique()), bbox_to_anchor=(0.5, 1.3))
        # Export the legend
        #export_legend(legend)

        if counter > 0:
            ax.set_ylabel(None)

        g.set(ylim=(0.5, 1))
        g.set(xlim=(0.4, 1))

        sns.despine()
        plt.tight_layout()
        plt.savefig(f"ecdf_real_data_{NOISE_LEVELS[counter]}.pdf", dpi=400, format='pdf', bbox_inches='tight')

        counter += 1