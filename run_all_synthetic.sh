#!/bin/bash

# Section 5 experiments
mpirun python run_synthetic_experiment.py --calibrate --labels 10

# Appendix D experiments
mpirun python run_synthetic_experiment.py --calibrate --skip-brute-force --labels 25
mpirun python run_synthetic_experiment.py --calibrate --skip-brute-force --labels 50

