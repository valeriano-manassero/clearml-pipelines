# Machine Learning pipelines
Pipelines built on top of Allegro ClearML.

## General info
This repository contains machine learning pipelines mostly based on PyTorch.
Every pipeline is designed to be published on a Allegro ClearML Kubernetes cluster *on premise*.

Each folder contains needed code and README for pipeline usage.

Further pipelines are welcome via pull request.

## Pipelines:
* **[mushrooms](mushrooms)** - Complete pipeline for a simple Pytorch model on a tabular mushrooms dataset.
* **[inat-2019](inat-2019)** - Complete pipeline for a MobilenetV2 model on iNaturalist 2019 dataset [WIP].

## Prerequisites
Here some prerequisites needed to deploy this repo.

### Platform versions
* Allegro Trains >=1.6.4
* PyTorch >=1.7.1

### Kubernetes cluster
Kubernetes installation can be done using official ClearML chart at https://github.com/allegroai/trains-server-helm#deploying-trains-server

A (hopefully) good alternative is the usage of GitOps paradigm with declaration of the entire cluster at https://github.com/valeriano-manassero/mlops-k8s-infra

## Local development and building
Some python libraries are needed. it's possible to locally install them with:
```
pip install -r requirements.txt
```
requirements.txt files are on every pipeline folder. No need of manual install inside a ClearML task (instlllation should be automatic).

## Useful links
* [Allegro ClearML](https://allegro.ai/)
* [PyTorch](https://pytorch.org/)
