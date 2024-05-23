import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from utils.general import user_accuracy
from tqdm import tqdm

from analytics.utils_plot import command_line
sns.set(font_scale=1.0,
        style="ticks",
        rc={
        "text.usetex": True,
        'text.latex.preamble': r'\usepackage{amsfonts}',
        "font.family": "serif",
    })

parser = command_line()
args = parser.parse_args()

data = pickle.load(open(args.file[0], 'rb'))
gt_filename = "-".join(args.file[0].split("-")[:-1]) + "-ground_truth.pickle"
cm_filename = "-".join(args.file[0].split("-")[:-1]) + "-confusion_matrices.pickle"
ground_truth = pickle.load(open(gt_filename, 'rb'))
users_cms = pickle.load(open(cm_filename, 'rb'))

total_labels = int(args.file[0].split("-")[1])

complete_accuracy = []

np.random.seed(2024)

# Convert prediction sets in lengths
for k in tqdm(data.keys()):

    user_mixture, run_id, model_accuracy = k[0], k[-2], k[-1]
    pset_key = (run_id, model_accuracy, user_mixture)

    user_cm = {-1: users_cms.get(pset_key)}
    acc, _ = user_accuracy(ground_truth.get(pset_key), data[k], user_cm)

    method_name = k[1]
    if k[1] == f"Greedy (k={total_labels})":
        method_name = r"\textsc{Greedy}"
    if k[1] == f"BF (k={total_labels})":
        method_name = r"\textsc{Brute force}"
    if k[1] == f"CP":
        method_name = r"\textsc{Naive}"
    if k[1] == f"APS":
        method_name = r"\textsc{Aps}"
    
    complete_accuracy.append([method_name, k[0], model_accuracy, acc])

    if method_name == r"\textsc{Naive}":
        complete_accuracy.append([r'\textsc{None}', k[0], model_accuracy, np.mean(
            [users_cms.get(pset_key)[x,x] for x in range(len(users_cms.get(pset_key)[:, 0]))]
        )])


df = pd.DataFrame(complete_accuracy, columns=["Method", r"$\gamma$", r"$\mathbb{P}[Y' = Y]$", r"$\mathbb{P}[\hat{Y} \mid \mathcal{S}, Y]$"])

complete_accuracy = pd.DataFrame(complete_accuracy, columns=["Method", "Human", "Model", "Accuracy"])

methods_order = [r'\textsc{Naive}', r'\textsc{Aps}', r'\textsc{Greedy}', r'\textsc{None}']

vals = complete_accuracy
for human_acc in sorted(complete_accuracy.Human.unique()):

    if human_acc not in [0.3, 0.5, 0.7, 1.0]:
        continue

    vals1 = vals[vals.Human == human_acc]
    for method in methods_order:
        vals2 = vals1[vals1.Method == method]
        if len(vals2) == 0:
            continue
        value = f"{method}\t"
        for model_acc in sorted(complete_accuracy.Model.unique()):
            vals3 = vals2[vals2.Model == model_acc]
            assert(len(vals3) == 10)
            g = vals3.groupby("Method").mean("Accuracy")
            g = g.reset_index()
            g2 = vals3.groupby("Method").std()
            g2 = g2.reset_index()
            # Method  Human  Model  Accuracy
            for row, std in zip(g.values.tolist(), g2.values.tolist()):
                if method == r'\textsc{Greedy}':
                    value += r"$\bm{"+f"{round(row[3], 3):.3f} \,\scriptstyle\pm {round(std[3], 3):.3f}"+"}$ \t"
                else:
                    value += f"${round(row[3], 3):.3f} \,\scriptstyle\pm {round(std[3], 3):.3f}$ \t"
        print(value)