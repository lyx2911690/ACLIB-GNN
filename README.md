# ACLIB-GNN

Implementation of **Incorporating Adversarial Causal Learning into Information Bottlenecks for Interpretable Graph Neural Networks**.

Supports node classification on **Cora, Citeseer, Pubmed and Coauthor CS**.

## Installation

Run the following commands in a terminal, then run experiments from the repository root:

```bash
conda create -n aclib-gnn python=3.9 -y
conda activate aclib-gnn
python -m pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

For CUDA 12.1, replace the PyTorch installation command with:

```bash
python -m pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu121
```

## Training

```bash
python train.py --dataset Cora
python train.py --dataset Citeseer
python train.py --dataset Pubmed
python train.py --dataset CS
```

Datasets download automatically to `data/`. Defaults are 100 epochs, five-fold stratified cross-validation and seed 42. Checkpoints are selected using a separate validation set. 
For a single fold or multiple seeds:

```bash
python train.py --dataset Cora --fold 1 --output-dir results/cora-fold1
python train.py --dataset Cora --seeds 42 43 44 --output-dir results/cora-multiseed
```

Each run requires a new, empty output directory. Use `python train.py --help` for all options.

## Evaluation

After training, evaluate a saved checkpoint on its recorded test split:

```bash
python evaluate.py --run-dir results/Cora/paper --seed 42 --fold 1
```

Reports Accuracy, macro Precision, Recall, F1, AUC and AUPR. Training also saves these metrics automatically.

Results are saved to `results/<dataset>/<objective>/`, including configuration, runtime information, data splits, checkpoints, predictions, per-fold metrics and a summary of means and standard deviations.

## Software checks

```bash
python -m unittest discover -s tests -v
python train.py --smoke-test --output-dir results/smoke-check
```

Tests check code logic. The smoke test uses synthetic data to verify the training workflow; its scores are not experimental results.

## Directory structure

```text
aclib/          Model, losses, data processing and metrics
scripts/        Dataset shortcuts and source packaging
train.py        Training entry point
evaluate.py     Checkpoint evaluation entry point
```
