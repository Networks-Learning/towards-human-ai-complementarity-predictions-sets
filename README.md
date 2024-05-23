# Towards Human-AI Complementarity with Predictions Sets

This is a repository for the code of the paper [Towards Human-AI Complementarity with Predictions Sets](#).

## Install

This code was tested on Python 3.8 and a Linux machine. To run the experiments, we use OpenMPI to parallelize the execution.
Please ensure to have its library installed locally. To install the various components, follows the commands:

```
conda create --name conformal python=3.8
conda activate conformal
pip install numpy scipy matplotlib scikit-learn pandas mpi4py tqdm
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

Moreover, in order to generate the plots we use Latex, so make sure to have a suitable Latex version installed locally.

## Experiments

Then, you can use the following bash script to replicate our results:
- submission/run_all_synthetic.sh: run the synthetic experiment detailed in Section 5 (it might take some time).
- submission/run_all_real_data.sh: run the real data experiment detailed in Section 6 (it might take some time).
- submission/generate_all_plots.sh: generate all plots presented in the paper
- submission/generale_all_results.sh: generate the latex code for the various tables presented in the paper.
Remember to run all these scripts from inside the `submission` directory (e.g., by doing `cd submission` and running also `export PYTHONPATH=.`). 

## Code structure

The directories have the following content:
- analytics/: it contains python scripts to generate the results of the main paper.
- data/: it contains the ImageNet-16H data needed for the experiments and some python utility functions to preprocess them.
- results/: it contains a series of pickle files with the experimental results in a raw format.
- utils/: it contains utility functions and code to run the experiments.