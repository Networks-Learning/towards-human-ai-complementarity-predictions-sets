#!/bin/bash

# Generate figure 2 and appendix E
python analytics/plot_figure_2.py results/1000-10-20-1000-True-False-1-prediction_sets.pickle --model 0.83334 --human 0.5
python analytics/plot_figure_2.py results/1000-10-20-1000-True-False-1-prediction_sets.pickle --model 0.83334 --human 0.7
python analytics/plot_figure_2.py results/1000-10-20-1000-True-False-1-prediction_sets.pickle --model 0.83334 --human 1.0
python analytics/plot_figure_2.py results/1000-10-20-1000-True-False-1-prediction_sets.pickle --model 1.3 --human 0.5
python analytics/plot_figure_2.py results/1000-10-20-1000-True-False-1-prediction_sets.pickle --model 1.3 --human 0.7
python analytics/plot_figure_2.py results/1000-10-20-1000-True-False-1-prediction_sets.pickle --model 1.3 --human 1.0
python analytics/plot_figure_2.py results/1000-10-20-1000-True-False-1-prediction_sets.pickle --model 2.0  --human 0.5

# Generate Figure 3
python analytics/plot_results_real_ecdf.py results/real_data-epoch10-800-top-k-25-10-5-False-all-prediction_sets.pickle

# Generate Appendix F
python analytics/plot_MNL_comparison.py

# Generate Appendix G
python analytics/plot_conformal_alphas.py results/real_data-epoch10-800-top-k-25-10-5-False-all-prediction_sets_alphas.pickle