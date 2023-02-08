# MARS: Masked Automatic Ranks Selection in Tensor Decompositions
This repository contains code for our paper [MARS: Masked Automatic Ranks Selection in Tensor Decompositions](https://arxiv.org/abs/2006.10859), AISTATS 2023.


The main files are:
* mars.py &mdash; the main module, containing realizations of the MARS wrapper over a tensorized model, the MARS loss and auxiliary functions;
* tensorized_models.py &mdash; module, containing realizations of several implemented tensorized models, the base class and auxiliary functions.

The notebooks are:
* MNIST-2FC-soft.ipynb &mdash; Jupyter Notebook, replicating the MNIST 2FC-Net experiment using soft compression mode;
* MNIST-2FC-hard.ipynb &mdash; Jupyter Notebook, replicating the MNIST 2FC-Net experiment using hard compression mode.

To run the notebooks, first, install the tt-pytorch library from https://github.com/KhrulkovV/tt-pytorch  
System requirements and dependencies are described in https://github.com/KhrulkovV/tt-pytorch/blob/master/README.md  
After installing all the dependencies, run the following command to install tt-pytorch from Git via pip: `pip install git+https://github.com/KhrulkovV/tt-pytorch.git`
