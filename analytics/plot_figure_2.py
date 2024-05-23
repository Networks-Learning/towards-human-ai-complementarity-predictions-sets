import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from tqdm import tqdm
from analytics.utils_plot import command_line

sns.set(font_scale=1.0,
        style="ticks",
        rc={
        "text.usetex": True,
        'text.latex.preamble': r'\usepackage{amsfonts}',
        "font.family": "serif",
    })

from itertools import chain, combinations

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1,len(s)+1))

convert_model_accuracy = {
    0.43334: 0.3,
    0.83334: 0.5,
    1.3: 0.7,
    2.0: 0.9
}

parser = command_line()
parser.add_argument("--model", type=float, default=1.3, choices=[0.43334, 0.83334, 1.3, 2.0])
parser.add_argument("--human", type=float, default=0.7)
args = parser.parse_args()

data = pickle.load(open(args.file[0], 'rb'))
gt_filename = "-".join(args.file[0].split("-")[:-1]) + "-ground_truth.pickle"
cm_filename = "-".join(args.file[0].split("-")[:-1]) + "-confusion_matrices.pickle"
ground_truth = pickle.load(open(gt_filename, 'rb'))
users_cms = pickle.load(open(cm_filename, 'rb'))

total_labels = int(args.file[0].split("-")[1])

complete_accuracy = []

np.random.seed(2024)

confounding_set = {}

# Convert prediction sets in lengths
for k in tqdm(data.keys()):

    user_mixture, run_id, model_accuracy = k[0], k[-2], k[-1]
    pset_key = (run_id, model_accuracy, user_mixture)

    if "BF" in k[1] or "Top-K" in k[1]:
        continue

    # We inspect a single seed run
    if run_id != 9:
        continue

    if user_mixture != args.human or model_accuracy != args.model:
        continue

    user_cm = {-1: users_cms.get(pset_key)}

    if confounding_set.get((model_accuracy, user_mixture), None) is None:
        
        all_confounding_sets = {}
        for y in range(total_labels):
            worst_combination = [100, None]
            for pset in powerset(range(total_labels)):
                if len(list(pset)) != 2:
                    continue
                if y not in list(pset):
                    continue
                else:
                    denom = np.sum([user_cm.get(-1)[x, y] for x in pset])
                    acc = user_cm.get(-1)[y, y]/denom
                if acc < worst_combination[0]:
                    worst_combination = [acc, list(pset)]
            all_confounding_sets[y] = worst_combination
        
        # Get only the top-3 labels
        tmp = [(k, idx, all_confounding_sets.get(idx)[0]) for k, idx in enumerate(all_confounding_sets)]
        tmp = [idx for _, idx, _ in sorted(tmp, key=lambda x: x[2])[0:4]]
        all_confounding_sets = {y : all_confounding_sets.get(y)[1] for y in tmp}

        print(all_confounding_sets)
        
        confounding_set[(model_accuracy, user_mixture)] = all_confounding_sets
    else:
        all_confounding_sets = confounding_set[(model_accuracy, user_mixture)]

    confusion = {y: 0 for y in range(total_labels)}
    all_accuracies = {y: [] for y in range(total_labels)}
    total_labels_y = {y: 0 for y in range(total_labels)}
    for y_true, prediction_set in zip(ground_truth.get(pset_key), data[k]):
        total_labels_y[y_true] += 1
        if y_true in all_confounding_sets:

            if all([k in prediction_set for k in all_confounding_sets[y_true]]):
                confusion[y_true] += 1
        else:
            all_accuracies[y_true].append(0)

    method_name = k[1]
    if k[1] == f"Greedy (k={total_labels})":
        method_name = r"\textsc{Greedy}"
    elif "CP" in k[1]:
        method_name = r"\textsc{Naive}"
    else:
        method_name = r"\textsc{Aps}"
    
    for y in all_confounding_sets:
        complete_accuracy.append([method_name, k[0],
                                  model_accuracy, confusion.get(y)/total_labels_y[y_true],
                                  f"$y={y}, "+r"\bar{y}"+f"={[y_bar for y_bar in all_confounding_sets[y] if y_bar != y][0]}$"
        ])

complete_accuracy = sorted(complete_accuracy, key=lambda x: x[4])

df = pd.DataFrame(complete_accuracy, columns=["Method", "Noise", "Model", r"$P\left(\right\{Y,\bar{Y}\left\}\in \mathcal{S} \mid Y=y\right)$", "Labels"])

methods_order = [r"\textsc{Naive}", r"\textsc{Aps}", r"\textsc{Greedy}"]

# Get the model values and convert the empirical model accuracy to 0.3, 0.5, 0.7, 0.9
convert_values = { current:new_val for new_val, current in zip([0.3, 0.5, 0.7, 0.9], sorted(df["Model"].unique()))}

fig, ax = plt.subplots(1,1, figsize=(5,3))
g = sns.barplot(
    data=df[df.Method.isin(methods_order)],
    x="Labels",
    y=r"$P\left(\right\{Y,\bar{Y}\left\}\in \mathcal{S} \mid Y=y\right)$",
    hue="Method",
    hue_order=methods_order,
    ax=ax
)
g.set_xlabel(None)
ax.legend(loc="upper center", ncol=len(methods_order), bbox_to_anchor=(0.5, 1.2))
plt.savefig(f"errors_with_cp_{convert_model_accuracy.get(args.model)}_{args.human}.pdf", dpi=400, format="pdf", bbox_inches="tight")

fig, ax = plt.subplots(1,1, figsize=(7,3))

g = sns.heatmap(user_cm.get(-1), annot=True, fmt=".2f", cmap="Blues",
                xticklabels=range(total_labels), yticklabels=range(total_labels), ax=ax,
                square=True,  annot_kws={"size": 7}
                )
ax.set_ylabel("$\hat{Y}$")
ax.set_xlabel("$Y$")

from matplotlib.patches import Rectangle

for y in all_confounding_sets:
    id_pos, name_pos = all_confounding_sets.get(y)
    print(id_pos, name_pos)
    ax.add_patch(Rectangle((id_pos, name_pos), 1, 1, ec='red', fc='none', lw=2))
    plt.setp(ax.get_xticklabels()[id_pos], color='red', font="bold")

plt.savefig(f"confusion_matrix_{convert_model_accuracy.get(args.model)}_{args.human}.pdf", dpi=400, format="pdf", bbox_inches="tight")