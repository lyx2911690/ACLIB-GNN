# ACLIB-GNN：Incorporating Adversarial Causal Learning with Information Bottlenecks for Interpretable Graph Neural Networks
This is the official code for the implementation of "ACLIB-GNN：Incorporating Adversarial Causal Learning with Information Bottlenecks for Interpretable Graph Neural Networks"
## Table of contents
* [Overview](#overview)
* [Installation](#installation)
* [Experimental Setup](#experimental-setup)
* [Run ACLIB-GNN](#run-aclib-gnn)

## Overview

In this work, we propose ACLIB-GNN, a framework unifying adversarial causal learning and the graph information bottleneck to address these gaps. By leveraging graph attention to filter redundant structural noise and adversarial training to maximize mutual information between explanatory subgraphs and labels, it explicitly disentangles causal features from shortcut signals, balancing transparency and performance.

## Installation
```shell
conda create -n ACLIB-GNN python=3.9
conda activate ACLIB-GNN
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121
pip install https://data.pyg.org/whl/torch-2.5.0%2Bcu121/torch_cluster-1.6.3%2Bpt25cu121-cp39-cp39-win_amd64.whl
pip install https://data.pyg.org/whl/torch-2.5.0%2Bcu121/torch_scatter-2.1.2%2Bpt25cu121-cp39-cp39-win_amd64.whl
pip install https://data.pyg.org/whl/torch-2.5.0%2Bcu121/torch_sparse-0.6.18%2Bpt25cu121-cp39-cp39-win_amd64.whl
pip install https://data.pyg.org/whl/torch-2.5.0%2Bcu121/torch_spline_conv-1.2.2%2Bpt25cu121-cp39-cp39-win_amd64.whl
pip install torch-geometric==2.6.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install pandas==2.2.3
pip install numpy==1.26.4
pip install scikit-learn==1.6.1
```

## Experimental Setup

Parameters | Cora | Citeseer | Pubmed | CS
--- | --- | --- | --- | ---
learning rate | 0.0001 | 0.001 | 0.001 | 0.001
weight_decay | 5e-4 | 5e-4 | 5e-4 | 5e-3
α | 0.4 | 0.4 | 0.7 | 0.5
β | 0.6 | 0.6 | 0.3 | 0.5
ε | 0.1 | 0.1 | 0.01 | 0.001

## Run ACLIB-GNN

For the Cora dataset, run &nbsp;
```Cora.py```

For the Citeseer dataset, run &nbsp;
```Citeseer.py```

For the Pubmed dataset, run &nbsp;
```Pubmed.py```

For the CS dataset, run &nbsp;
```CS.py```
