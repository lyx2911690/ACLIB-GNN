# ACLIB-GNN

**Incorporating Adversarial Causal Learning into Information Bottlenecks for Interpretable Graph Neural Networks**

YunXia Li, Chuya Wu, Mengjie Cai, Quan Lu and Zhenyu Yue  
School of Artificial Intelligence, Anhui Agricultural University

This repository provides a node-classification implementation based on the supplied manuscript and code. It supports Cora, Citeseer, Pubmed and Coauthor CS with saved splits, validation-based checkpoint selection, and per-fold predictions. The default objective implements manuscript equations (3)–(8). A second mode preserves the supplied scripts' loss formula with corrected training and evaluation.

**Reproduction status:** this is a corrected implementation, not a verified reproduction of the paper's reported numbers. Read [the manuscript audit](docs/REPRODUCIBILITY.md) for differences, unresolved pseudocode details and experimental artifacts still required. Original scripts are preserved in `legacy/`.

## Installation

The dependency specification retains Python 3.9, PyTorch 2.5.0 and PyG 2.6.1 from the supplied README. Use a clean environment:

```bash
conda create -n aclib-gnn python=3.9 -y
conda activate aclib-gnn
python -m pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

For CUDA 12.1, replace the PyTorch installation line with:

```bash
python -m pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu121
```

Alternatively, `conda env create -f environment.yml` installs the pip dependency specification. The model uses standard PyG layers without directly importing optional compiled scatter/cluster/sparse extensions. Do not reuse the old README's Windows-specific wheel URLs on other platforms. Dataset downloads require network access. CPU execution is supported; full-batch CS training may require substantial memory.

## Quick verification

```bash
python -m unittest discover -s tests -v
python train.py --smoke-test --device cpu --output-dir results/smoke-check --export-nodes 0 5
```

The smoke test runs up to two epochs on a synthetic graph without downloading datasets. Its metrics are only a software check. GitHub Actions runs the invariant tests and synthetic paper, script and ablation checks on CPU. Local verification details are in [VALIDATION.md](docs/VALIDATION.md).

## Run experiments

```bash
python train.py --dataset Cora
python train.py --dataset Citeseer
python train.py --dataset Pubmed
python train.py --dataset CS
```

Data are downloaded through PyG into `data/<dataset>/`. Each default run uses 100 epochs, five shuffled stratified outer folds, seed 42, hidden dimension 64, four first-layer attention heads, one hop, and at most ten nodes per candidate subgraph. The center is retained, followed by ascending node IDs. The original scripts supplied the following dataset defaults:

| Parameter | Cora | Citeseer | Pubmed | CS |
| --- | --- | --- | --- | --- |
| Learning rate | 0.0001 | 0.001 | 0.001 | 0.001 |
| Weight decay | 0.0005 | 0.0005 | 0.0005 | 0.005 |
| Alpha | 0.4 | 0.4 | 0.7 | 0.5 |
| Beta | 0.6 | 0.6 | 0.3 | 0.5 |
| Epsilon | 0.1 | 0.1 | 0.01 | 0.001 |

The outer training nodes are split into training and validation sets (90%/10%). Only validation accuracy selects the checkpoint. Test labels never enter training or FGSM generation. All node features/edges remain available: this is transductive cross-validation and does not use the Planetoid public masks.

```bash
# Multiple seeds; output folds and sample standard deviations
python train.py --dataset Cora --seeds 42 43 44 --output-dir results/cora-multiseed

# One fold, configurable data location and hyperparameters
python train.py --dataset Pubmed --fold 1 --data-dir data --alpha 0.7 --beta 0.3 --epsilon 0.01

# Supplied-script objective, with implementation corrections
python train.py --dataset Cora --objective script --output-dir results/cora-script

# Explicit engineering ablations (see manuscript audit)
python train.py --dataset Cora --no-adversarial --output-dir results/cora-no-adv
python train.py --dataset Cora --backbone gcn --output-dir results/cora-gcn

# Export candidate neighborhood nodes and actual induced edges
python train.py --dataset Cora --fold 1 --export-nodes 0 10 42 --output-dir results/cora-exports

# Evaluate a saved checkpoint and its recorded test fold
python evaluate.py --run-dir results/Cora/paper --seed 42 --fold 1
```

`python scripts/Cora.py` (and `scripts/Citeseer.py`, `scripts/Pubmed.py`, `scripts/CS.py`) forwards to the unified runner and accepts the same arguments. `python train.py --help` lists all options. `--max-subgraph-nodes 0` retains the full k-hop support. `--deterministic` requests strict deterministic operations; unsupported GPU operations raise an error. Seeds alone do not guarantee identical results across devices or dependency versions.

Every run requires an empty output directory to prevent mixing experiments. Select a new `--output-dir` when rerunning.

## Outputs

```text
results/<dataset>/<objective>/
  config.json                 # Resolved arguments and hyperparameters
  environment.json            # Runtime versions, device and graph fingerprint
  all_metrics.csv             # One row per seed/fold
  summary.json                # Mean, sample std and observation count
  seed42/fold1/
    splits.npz                # Explicit train/validation/test node IDs
    best.pt                   # Best validation checkpoint and epoch
    history.json              # Loss components and validation accuracy
    metrics.json              # Test classification metrics
    predictions.npz           # Test node IDs, labels and probabilities
    subgraphs.json            # Optional candidate supports and predicted labels
```

Accuracy, precision, recall and F1 are reported as fractions; precision/recall/F1 are macro averaged. AUC/AUPR use macro one-vs-rest probabilities. `Local_*` evaluates node embeddings; `Combined_*` evaluates pooled subgraphs in paper mode or concatenated features in script mode. Classification AUC is not explanation AUC. Exported neighborhoods are candidate supports and carry no verified causal ranking.

## Repository layout

- `aclib/`: encoder, losses, dataset loading, pooling and metrics
- `configs/datasets.json`: dataset-specific defaults
- `train.py` / `evaluate.py`: experiment and checkpoint evaluation entry points
- `scripts/`: dataset shortcuts and source packaging
- `tests/` / `.github/workflows/`: software checks
- `legacy/`: unchanged source scripts and original README
- `docs/REPRODUCIBILITY.md`: equation mapping, corrections and missing experiments

Local datasets belong in `data/`, experiment outputs in `results/`, and source archives in `dist/`. These generated directories are excluded from Git. See [the directory guide](docs/STRUCTURE.zh-CN.md) for the full layout.

Build the upload archive with `python scripts/package_source.py`; the output is `dist/ACLIB-GNN-GitHub-source.zip`.

## Citation and release information

Please cite the manuscript under the title and author names above. Venue, DOI and publication year have not been confirmed, so no publication metadata is fabricated here. The release license must be chosen by the copyright holder and added as `LICENSE` before redistribution terms can be stated. The manuscript's code-availability URL (`CGAMP`) also needs confirmation against the intended ACLIB-GNN repository.

For uploading, include source, configuration, tests and documentation. `.gitignore` excludes downloaded datasets, model checkpoints, experimental outputs and local verification dependencies. The supplied manuscript is not included in the source package.
