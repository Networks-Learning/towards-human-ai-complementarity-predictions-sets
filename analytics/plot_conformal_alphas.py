import pandas as pd
import numpy as np
import pickle
from utils.general import user_accuracy
from tqdm import tqdm

from analytics.utils_plot import command_line
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(font_scale=1.0,
        style="ticks",
        rc={
        "text.usetex": True,
        'text.latex.preamble': r'\usepackage{amsfonts}',
        "font.family": "serif",
    })

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
METHODS = ['CP', 'APS']

complete_accuracy = []

for noise in NOISE_LEVELS:
    for model in MODELS:
        for run_id in tqdm(range(10), desc=f"{model} (Noise={noise})"):
            for model_name in METHODS:

                if model_name != "CP" and model_name != "APS":
                    continue

                pset_key = (run_id, model, noise)

                # Add complete matrix
                user_cm_difficulties = users_cms.get(model).get(noise).get(run_id)[0]

                for alphas in data.get(model_name).get(pset_key):

                    # Skip missing keys 
                    if ground_truth.get(pset_key, None) is None or data.get(model_name).get(pset_key).get(alphas, None) is None:
                        continue

                    # Rebuilt the correct values, since they are now dictionaries
                    X, Y, softmaxes = [], [], []
                    for key in ground_truth.get(pset_key):
                        X.append(data.get(model_name).get(pset_key).get(alphas).get(key, None))
                        Y.append(ground_truth.get(pset_key).get(key))
                        assert X[-1] is not None and Y[-1] is not None
                
                    # If we want to use the score given by the optimization objective, 
                    # the we use the full MNL mode, since we cannot know the difficulties at
                    # test time.
                    acc, std = user_accuracy(Y, X, user_cm_difficulties)

                    # We append the accuracy to the complete values
                    complete_accuracy.append([model, noise, r"\textsc{Naive}" if model_name == "CP" else r"\textsc{Aps}", acc, alphas])

# Generate a dataframe with all the results
complete_accuracy = pd.DataFrame(complete_accuracy, columns=["Model", "Noise", "Method", "Empirical Success \n Probability", r"$\alpha$"])

# Subplots
fig, ax = plt.subplots(1,len(NOISE_LEVELS), figsize=(10,3))

for plot_idx, noise in enumerate(NOISE_LEVELS):

    quantile_alphas = complete_accuracy[r"$\alpha$"].unique().tolist()[::2] # for rendering reasons, we plot only half of the points
    tmp_full = complete_accuracy[(complete_accuracy.Noise == noise)] # we use it to compute the best
    tmp = complete_accuracy[(complete_accuracy.Noise == noise) & (complete_accuracy[r"$\alpha$"].isin(quantile_alphas))]
        
    fig, ax = plt.subplots(1,1, figsize=(5,3))

    for marker, method in zip(["^", "o"], [r"\textsc{Naive}", r"\textsc{Aps}"]):

        current_method_values = tmp[tmp.Method == method]
        avg_accuracy = current_method_values.groupby(r"$\alpha$")["Empirical Success \n Probability"].mean()
        std_accuracy = current_method_values.groupby(r"$\alpha$")["Empirical Success \n Probability"].std()*1.96/np.sqrt(10)
        ax.plot(sorted(current_method_values[r"$\alpha$"].unique()), avg_accuracy, f'-', label=method, zorder = -10)

        # Add std error
        y1 = avg_accuracy-std_accuracy
        y2 = avg_accuracy+std_accuracy
        ax.fill_between(sorted(current_method_values[r"$\alpha$"].unique()), y1=y1, y2=y2, alpha=.2)

        # Plot the best value so far
        best_alpha_real = tmp_full[tmp_full.Method == method].groupby(r"$\alpha$")["Empirical Success \n Probability"].mean().idxmax()
        best_alpha_real_success = tmp_full[tmp_full.Method == method].groupby(r"$\alpha$")["Empirical Success \n Probability"].mean().max()
        ax.plot([best_alpha_real], [best_alpha_real_success], f'r{marker}')

        ax.set_ylabel("Empirical Success \n Probability")
        ax.set_xlabel(r"$\alpha$")
    
    # Rasterize
    ax.set_rasterization_zorder(0)

    ax.legend(title=None, bbox_to_anchor=(0.7, 1.2), ncols=len(tmp.Method.unique()))
    plt.savefig(f"performance_alphas_{noise}_real.pdf", dpi=400, format="pdf", bbox_inches="tight")