import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(font_scale=1.0,
        style="ticks",
        rc={
        "text.usetex": True,
        'text.latex.preamble': r'\usepackage{amsfonts}',
        "font.family": "serif"
    })

def alpha_vs_acc(alphas_real_humans, alphas_greedy, show_full=False, save=True, show_markers=True, show_cf_se=True, no_baselines=False):
    """Plot alpha vs accuracy for the strict and the lenient implementation of our systems 
    adn highlight baselines"""

    fig, ax = plt.subplots(1,1, figsize=(5,3))

    path = alphas_real_humans
    with open(path, 'rt') as f:
        strict_alphas_avg_acc_se = json.load(f)
    
    df_strict = pd.DataFrame(strict_alphas_avg_acc_se).T
    df_strict.rename(columns={'avg':'avg_strict', 'se':'se_strict'}, inplace=True)
    df_strict.index.rename('alpha', inplace=True)
    print(df_strict)    

    path = alphas_greedy
    with open(path, 'rt') as f:
        lenient_alphas_avg_acc_se = json.load(f)
    
    for _, d in lenient_alphas_avg_acc_se.items():
        del d["images"]
        del d["quantile"]

        d["avg"] = d.pop("accuracy")
        d["se"] = d.pop("std")

    df_lenient = pd.DataFrame(lenient_alphas_avg_acc_se).T
    df_lenient.rename(columns={'avg':'avg_lenient', 'se':'se_lenient'}, inplace=True)
    print(df_lenient)

    df_lenient["se_lenient"] = df_lenient["se_lenient"] / np.sqrt(1080)# / np.sqrt(1080*5)

    df_lenient.index.rename('alpha', inplace=True)

    all_df = df_strict.merge(df_lenient, on='alpha', how='outer').sort_index()
    
    alphas_to_plot = all_df[all_df.index < '0.99']
    if show_full:
        alphas_to_plot = all_df
    
    alphas_to_plot.index = alphas_to_plot.index.astype(float)
    ci_radious = alphas_to_plot[['se_strict', 'se_lenient']]*1.96
    
    alphas_to_plot.plot(y=['avg_strict','avg_lenient'], style=['-o', '-x'], zorder=0,
                             ax=ax, color=sns.color_palette("colorblind")[2:4])
    
    # Plot the max values
    print(alphas_to_plot)
    best_alpha_real = alphas_to_plot['avg_strict'].idxmax()
    best_alpha_real_success = alphas_to_plot['avg_strict'].max()

    ax.plot([best_alpha_real], [best_alpha_real_success], 'ro')
    ax.annotate('Real human \n optimum',
                xy=(best_alpha_real, best_alpha_real_success),
                xytext=(best_alpha_real-0.1, best_alpha_real_success-0.1),
            arrowprops=dict(color='black', arrowstyle="->"),
            )

    best_alpha_real = alphas_to_plot['avg_lenient'].idxmax()
    best_alpha_real_success = alphas_to_plot['avg_lenient'].max()
    ax.plot([best_alpha_real], [best_alpha_real_success], 'rx')

    ax.annotate('MNL optimum',
                xy=(best_alpha_real, best_alpha_real_success),
                xytext=(best_alpha_real+0.2, best_alpha_real_success+0.01),
            arrowprops=dict(color='black', arrowstyle="->"),
            )

    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r"Empirical Success Probability")
    ax.spines[['right', 'top']].set_visible(False)
    
    ax.legend([ r'Real humans', r'MNL'])
        
    # Add 95% CI in lines strict and lenient
    counter = 2
    for setting in ['strict', 'lenient']:
        avg_k = f"avg_{setting}"
        se_k = f"se_{setting}"
        y1 = alphas_to_plot[avg_k].values-ci_radious[se_k].values
        y2 = alphas_to_plot[avg_k].values+ci_radious[se_k].values  
        ax.fill_between(alphas_to_plot.index, y1=y1, y2=y2, alpha=.2, color=sns.color_palette("colorblind")[counter])
        counter += 1

    plt.savefig("MNL_vs_human.pdf", format="pdf", dpi=400, bbox_inches="tight")

if __name__ == "__main__":
    
    alpha_vs_acc(
        "data/ImageNet-16H/real_human_eval/deploy_avg_acc_se_alphas_1.json",
        "data/ImageNet-16H/real_human_eval/MNL_accuracy.json"
    )
