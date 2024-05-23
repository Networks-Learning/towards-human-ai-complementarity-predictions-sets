#!/bin/bash

# Run real experiment Section 6
mpirun python run_real_experiment.py --model-epochs epoch10 --calibrate top-k --ranks 5 --calibration-size 800
