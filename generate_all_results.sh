#!/bin/bash

# Print latex for Table 1
python analytics/plot_results_synthetic.py results/1000-10-20-1000-True-False-1-prediction_sets.pickle

# Print latex for Table in Appendix D
python analytics/plot_results_synthetic.py results/1000-25-20-1000-True-True-1-prediction_sets.pickle
python analytics/plot_results_synthetic.py results/1000-50-20-1000-True-True-1-prediction_sets.pickle

# Print latex for Table 2
python analytics/plot_results_real.py results/real_data-epoch10-800-top-k-25-10-5-False-all-prediction_sets.pickle