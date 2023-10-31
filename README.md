# GNN-SSL: Graph Neural Networks for Sound Source Localization

This repository contains the code for the papers 

["Graph neural networks for sound source localization on distributed microphone networks"](https://ieeexplore.ieee.org/abstract/document/10097211),
Eric Grinstein, Mike Brookes and Patrick Naylor, IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Rhodes Island, Greece, 2023

and 

"The Neural-SRP method for positional sound source localization" Eric Grinstein, Toon van Waterschoot, Mike Brookes and Patrick Naylor, IEEE Asilomar Conference, California, USA, 2023


E. Grinstein, M. Brookes and P. A. Naylor, "Graph Neural Networks for Sound Source Localization on Distributed Microphone Networks," ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Rhodes Island, Greece, 2023, pp. 1-5, doi: 10.1109/ICASSP49357.2023.10097211.


## Installation instructions

We suggest installing the required Python libraries in a virtual environment.
The instructions for doing so using Conda are listed below.

1. Clone this repository using the following command: 

`git clone https://github.com/SOUNDS-RESEARCH/gnn_ssl --recurse-submodules`

The `--recurse-submodules` flag is required as this project depends on other github projects, which are included as submodules.
The projects are "SYDRA", which is used to generate and manipulate synthetic acoustic data, and "Pysoundloc", which contains the implementations
of the Steered Response Power (SRP) and Least Squares localization methods used as baselines.

In case you forgot to use this flag, you can download the submodules by running the command `git submodule update --init` from the project's root. 

2. Create a virtual environment using `conda env create -f environment.yml`. Then activate it using `conda activate gnn_ssl`.

## Reproducing the experiments

1. Creating datasets

Synthetic datasets can be created using SYDRA. For example, by changing into SYDRA's directory (`cd sydra`) and running `python main.py dataset_dir="/path/to/dataset" n_samples=100`
will create a dataset containing 100 examples. For more information on customizing the dataset generation, read SYDRA's readme.

2. Create a `inputs_train.yaml` file

Copy the file `inputs_train_template.yaml` as a file called `ìnputs_train.yaml` file under `gnn_ssl/config`,
and edit the paths to refer to your training, validation and testing datasets.

Do the same for the `inputs_eval.yaml` file, which is used for testing the model.

3. Training the model

Run `python train.py` to start training. The outputs (trained weights, Tensorboard stats) will be saved in the `outputs/` directory
