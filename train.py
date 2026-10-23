"""Run node classification without selecting checkpoints on the test set."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["Cora", "Citeseer", "Pubmed", "CS"], default="Cora")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--objective", choices=["paper", "script"], default="paper")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--fold", type=int, help="Run one fold, numbered from 1")
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--val-ratio", type=float, default=0.1, help="Fraction of outer training nodes")
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--hops", type=int, default=1)
    p.add_argument("--max-subgraph-nodes", type=int, default=10, help="0 means uncapped")
    p.add_argument("--gamma", type=float, default=0.1)
    for name in ["lr", "weight-decay", "alpha", "beta", "epsilon"]:
        p.add_argument("--" + name, type=float)
    p.add_argument("--backbone", choices=["gat", "gcn"], default="gat")
    p.add_argument("--no-adversarial", action="store_true")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--deterministic", action="store_true", help="Fail if an operation is nondeterministic")
    p.add_argument("--export-nodes", type=int, nargs="*", default=[])
    p.add_argument("--smoke-test", action="store_true", help="Synthetic graph; no download or scientific results")
    return p


def seed_everything(seed, deterministic):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(deterministic)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if (args.epochs < 1 or args.folds < 2 or args.hops < 0 or args.max_subgraph_nodes < 0
            or args.hidden_dim < 1 or args.heads < 1 or not 0 < args.val_ratio < 1
            or (args.fold is not None and not 1 <= args.fold <= args.folds)):
        p.error("Invalid epochs, folds, fold, dimensions, hops, cap or validation ratio")
    if args.no_adversarial and args.objective == "script":
        p.error("--no-adversarial requires --objective paper")
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 or s >= 2**32 for s in args.seeds):
        p.error("Seeds must be unique integers in [0, 2**32)")
    defaults = json.loads((Path(__file__).parent / "configs/datasets.json").read_text())[args.dataset]
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if min(args.lr, args.weight_decay, args.alpha, args.beta, args.epsilon, args.gamma) < 0 or args.lr == 0:
        p.error("Learning rate must be positive; loss weights, decay, epsilon and gamma nonnegative")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    import torch_geometric
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from tqdm import trange
    from aclib.data import load_dataset, neighborhoods, pool, pooling_index, synthetic_graph
    from aclib.metrics import classification_metrics
    from aclib.model import ACLIBGNN, adversarial_embeddings, objective

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        p.error("CUDA requested but unavailable")
    seed_everything(args.seeds[0], args.deterministic)
    if args.smoke_test:
        data = synthetic_graph(args.seeds[0])
        num_classes = 3
        args.epochs = min(args.epochs, 2)
        args.fold = args.fold or 1
    else:
        dataset = load_dataset(args.dataset, args.data_dir)
        data, num_classes = dataset[0], dataset.num_classes
    labels = data.y.numpy()
    if np.bincount(labels).min() < args.folds:
        p.error("Each class needs at least --folds examples")
    if any(n < 0 or n >= data.num_nodes for n in args.export_nodes):
        p.error("Export node ID outside graph")
    groups = neighborhoods(data.edge_index, data.num_nodes, args.hops, args.max_subgraph_nodes)
    index = pooling_index(groups, device)
    fingerprint = hashlib.sha256()
    for tensor in (data.x, data.y, data.edge_index):
        fingerprint.update(tensor.contiguous().numpy().tobytes())
    data = data.to(device)
    output = args.output_dir or Path("results") / ("smoke" if args.smoke_test else args.dataset) / args.objective
    # Never mix metrics from different configurations or overwrite a previous run.
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        p.error(f"Output directory is not empty: {output}. Choose a new --output-dir")
    write_json(output / "config.json", {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    write_json(output / "environment.json", {"python": sys.version, "platform": platform.platform(),
               "torch": str(torch.__version__), "torch_geometric": torch_geometric.__version__,
               "numpy": np.__version__, "device": str(device), "cuda": torch.version.cuda,
               "graph_sha256": fingerprint.hexdigest(), "synthetic": args.smoke_test})
    results = []

    for seed in args.seeds:
        splits = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=seed)
        for fold, (outer_train, test) in enumerate(splits.split(np.zeros(len(labels)), labels), 1):
            if args.fold is not None and fold != args.fold:
                continue
            train, val = train_test_split(outer_train, test_size=args.val_ratio,
                                         stratify=labels[outer_train], random_state=seed)
            seed_everything(seed + fold, args.deterministic)
            directory = output / f"seed{seed}" / f"fold{fold}"
            directory.mkdir(parents=True)
            np.savez(directory / "splits.npz", train=train, val=val, test=test)
            train_idx = torch.tensor(train, device=device)
            val_idx = torch.tensor(val, device=device)
            model = ACLIBGNN(data.num_features, args.hidden_dim, num_classes, args.heads, args.backbone).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

            def predict():
                model.eval()
                with torch.no_grad():
                    embeddings = model(data.x, data.edge_index)
                    pooled = pool(embeddings, index)
                    local = model.node_classifier(embeddings)
                    # No ground-truth label or FGSM perturbation is used at inference.
                    subgraph = model.node_classifier(pooled)
                    combined = model.combined_classifier(torch.cat([embeddings, pooled, pooled], dim=1))
                return local, subgraph if args.objective == "paper" else combined

            best, history = -1.0, []
            progress = trange(args.epochs, desc=f"Seed {seed} Fold {fold}")
            for epoch in progress:
                model.train()
                optimizer.zero_grad(set_to_none=True)
                embeddings = model(data.x, data.edge_index)
                pooled = pool(embeddings, index)
                adversarial = pooled if args.no_adversarial else adversarial_embeddings(
                    model, pooled, data.y, train_idx, args.epsilon)
                loss, parts = objective(model, embeddings, pooled, adversarial, data.y, train_idx,
                                        args.alpha, args.beta, args.gamma, args.objective, args.no_adversarial)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss")
                loss.backward()
                optimizer.step()
                _, predictions = predict()
                accuracy = (predictions[val_idx].argmax(1) == data.y[val_idx]).float().mean().item()
                history.append({"epoch": epoch + 1, "loss": loss.item(), "val_accuracy": accuracy, **parts})
                if accuracy > best:
                    best = accuracy
                    torch.save({"model": model.state_dict(), "epoch": epoch + 1,
                                "val_accuracy": best}, directory / "best.pt")
                progress.set_postfix(loss=loss.item(), val_accuracy=accuracy)
            write_json(directory / "history.json", history)
            checkpoint = torch.load(directory / "best.pt", map_location=device, weights_only=True)
            model.load_state_dict(checkpoint["model"])
            local, combined = predict()
            record = {"Seed": seed, "Fold": fold, "Best_epoch": checkpoint["epoch"]}
            arrays = {"node_ids": test, "labels": labels[test]}
            for name, logits in [("Local", local), ("Combined", combined)]:
                probabilities = logits[test].softmax(-1).cpu().numpy()
                arrays[name.lower() + "_probabilities"] = probabilities
                record.update({f"{name}_{k}": v for k, v in classification_metrics(labels[test], probabilities).items()})
            np.savez(directory / "predictions.npz", **arrays)
            write_json(directory / "metrics.json", record)
            if args.export_nodes:
                edges = data.edge_index.cpu().t().tolist()
                exports = []
                for node in args.export_nodes:
                    members = set(groups[node])
                    exports.append({"center": node, "nodes": groups[node], "edges": [
                        [u, v] for u, v in edges if u in members and v in members],
                        "predicted_label": int(combined[node].argmax().item()),
                        "selection": "center-first, then ascending node ID; candidate support, not causal ranking"})
                write_json(directory / "subgraphs.json", exports)
            results.append(record)
            print(json.dumps(record))
    with (output / "all_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    summary = {}
    for key in results[0]:
        if key.startswith(("Local_", "Combined_")):
            values = [r[key] for r in results if r[key] is not None]
            summary[key] = {"mean": float(np.mean(values)) if values else None,
                            "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                            "count": len(values)}
    write_json(output / "summary.json", summary)
    print(f"Saved results to {output.resolve()}")


if __name__ == "__main__":
    main()
