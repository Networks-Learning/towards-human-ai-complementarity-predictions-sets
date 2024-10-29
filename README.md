# Towards Human-AI Complementarity with Prediction Sets

This repository is the official implementation of [Towards Human-AI Complementarity with Prediction Sets](https://arxiv.org/abs/2405.17544).

## Requirements

This code was tested on Python 3.8 and on a Linux machine. The main prerequisite to run our experiments is to have a working MPI implementation (see [here](https://docs.open-mpi.org/en/v5.0.x/installing-open-mpi/quickstart.html) for additional instructions), since our scripts natively support multiprocessing.
You will need a working Latex installation to instead run the scripts generating the plots available in the manuscript.
On a Linux machine, please run the following to install the packages within a [conda](https://conda.io/projects/conda/en/latest/index.html) environment:

```bash
conda create --name hai-psets python=3.8
conda activate hai-psets
pip install -r requirements.txt
```

## Evaluation

In order to run our scripts, make sure to position yourself in the base directory and to activate the correct environment.

```bash
cd towards-human-ai-complementarity-predictions-sets
conda activate hai-psets
export PYTHONPATH=.
```

Please have a look to the following bash script running all of our experiments and analysis:
- `run_all_synthetic.sh`: run the synthetic experiment detailed in Section 5 (it might take some time).
- `run_all_real_data.sh`: run the real data experiment detailed in Section 6 (it might take some time).
- `generate_all_plots.sh`: generate all plots presented in the paper.
- `generale_all_results.sh`: generate the Latex code for the various tables presented in the paper.

### Running synthetic experiments

The script `run_synthetic_experiment.py` runs the synthetic experiments detailed in Section 5. Below you can find its invocation for all the prediction tasks with 10 labels where we apply calibration to the model. The `mpirun -n 2` invocation parallelizes the execution over 2 physical cores.

```bash
mpirun -n 2 python run_synthetic_experiment.py --calibrate --labels 10
```

### Running experiments with real data

The script `run_real_experiment.py` runs instead the experiments with the real data. For example, the following command generates the results for the experiment presented in Section 6.
```bash
mpirun - n 2 python run_real_experiment.py --model-epochs epoch10 --calibrate top-k --ranks 5 --calibration-size 800
```

## Repository Structure

The directories have the following content:
- `analytics/`: it contains python scripts to parse and analyze the results of the main paper.
- `data/`: it contains the ImageNet-16H data needed for the experiments and some python utility functions to preprocess them.
- `results/`: it contains a series of pickle files with our experimental results in a raw format.
- `utils/`: it contains utility functions and code to run the experiments.

## Citation
If you use parts of the code in this repository for your research purposes, please consider citing:
```
@article{detoni2024humanai,
      title={Towards Human-AI Complementarity with Prediction Sets}, 
      author={Giovanni De Toni and Nastaran Okati and Suhas Thejaswi and Eleni Straitouri and Manuel Gomez-Rodriguez},
      year={2024},
      journal={arXiv preprint arXiv:2405.17544}
}
```
