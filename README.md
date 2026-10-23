# ACLIB-GNN

ACLIB-GNN combines graph attention and adversarial learning for interpretable node classification. A two-layer graph attention encoder produces node representations, k-hop neighborhoods provide candidate explanatory subgraphs, and adversarial perturbations of pooled subgraph embeddings support joint node and subgraph learning.

The implementation includes three components:

* **Graph encoding:** learns node representations through multi-head graph attention.
* **Subgraph aggregation:** pools node-centered k-hop neighborhoods.
* **Adversarial learning:** perturbs subgraph embeddings using training labels and optimizes the node and subgraph objectives jointly.

[中文说明](README.zh-CN.md)

## Installation

Run all commands from the repository root. Create an environment and install the dependencies:

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

## Experimental settings

The default configuration uses 100 epochs, hidden dimension 64, four attention heads in the first layer, one-hop neighborhoods and at most 10 nodes per neighborhood. Dataset-specific hyperparameters are stored in [configs/datasets.json](configs/datasets.json).

Experiments use transductive five-fold stratified cross-validation with seed 42. Within each fold, 10% of the outer training nodes form a separate validation set for checkpoint selection. Test labels are excluded from training and perturbation generation. CPU and CUDA execution are supported.

## Data preparation

The supported datasets are **Cora, Citeseer, Pubmed and Coauthor CS**. PyTorch Geometric downloads them automatically on the first training run; an internet connection is required.

```text
data/
├── Cora/
├── Citeseer/
├── Pubmed/
└── CS/
```

Use `--data-dir` to choose another data location. Keep the PyG-generated subdirectories intact.

## Prediction evaluation

Train a model first, then evaluate the saved checkpoint on its recorded test split. Pretrained checkpoints are not included in the source package.

```bash
python evaluate.py --run-dir results/Cora/paper --seed 42 --fold 1
python evaluate.py --run-dir results/Citeseer/paper --seed 42 --fold 1
python evaluate.py --run-dir results/Pubmed/paper --seed 42 --fold 1
python evaluate.py --run-dir results/CS/paper --seed 42 --fold 1
```

Evaluation reports Accuracy, macro Precision, Recall, F1, one-vs-rest AUC and AUPR. Training also evaluates each test fold automatically and saves the results.

## Model training

```bash
python train.py --dataset Cora
python train.py --dataset Citeseer
python train.py --dataset Pubmed
python train.py --dataset CS
```

Checkpoints, splits, predictions and metrics are saved under `results/<dataset>/<objective>/`. Each seed/fold has its own directory. `all_metrics.csv` contains per-fold results; `summary.json` contains their means and sample standard deviations.

Run a single fold or multiple seeds:

```bash
python train.py --dataset Cora --fold 1 --output-dir results/cora-fold1
python train.py --dataset Cora --seeds 42 43 44 --output-dir results/cora-multiseed
```

Each run requires an empty output directory. Use `python train.py --help` for all options.

## Subgraph export

Export candidate neighborhoods for selected nodes during training:

```bash
python train.py --dataset Cora --fold 1 --export-nodes 0 10 42 --output-dir results/cora-subgraphs
```

The fold directory contains `subgraphs.json` with node IDs, induced edges and predicted labels. These are candidate supports, not verified causal explanations. Explanation-ranking AUC and fidelity evaluation are not implemented in this release.

## Code structure

| Path | Function |
| --- | --- |
| `train.py` | Model training and cross-validation |
| `evaluate.py` | Saved-checkpoint evaluation |
| `aclib/model.py` | Graph encoder, classifiers, perturbations and losses |
| `aclib/data.py` | Dataset loading, neighborhood construction and pooling |
| `aclib/metrics.py` | Classification metrics |
| `configs/datasets.json` | Dataset-specific hyperparameters |
| `scripts/` | Dataset shortcuts and source packaging |
| `tests/` | Code correctness checks |
| `docs/` | Implementation, validation and upload notes |
| `legacy/` | Original scripts preserved for reference |
| `.github/workflows/` | Automated software checks |

To check the implementation without downloading datasets:

```bash
python -m unittest discover -s tests -v
python train.py --smoke-test --output-dir results/smoke-check
```

## Reproducibility status

The default `--objective paper` follows manuscript equations (3)–(8). `--objective script` retains the original scripts' loss formula with training and evaluation corrections. Reported paper scores have not been verified. Graph classification and baseline experiments are not included.

See the [implementation audit](docs/REPRODUCIBILITY.md) and [verification record](docs/VALIDATION.md) for details. A release license has not yet been selected.

## Citation

If you use this implementation, please cite the manuscript:

YunXia Li, Chuya Wu, Mengjie Cai, Quan Lu and Zhenyu Yue. *ACLIB-GNN: Incorporating Adversarial Causal Learning into Information Bottlenecks for Interpretable Graph Neural Networks.*

## Contact

* **Zhenyu Yue:** [zhenyuyue@ahau.edu.cn](mailto:zhenyuyue@ahau.edu.cn)
