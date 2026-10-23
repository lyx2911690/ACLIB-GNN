"""Evaluate a saved fold with its recorded configuration and test split."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from aclib.data import load_dataset, neighborhoods, pool, pooling_index, synthetic_graph
from aclib.metrics import classification_metrics
from aclib.model import ACLIBGNN


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Directory containing config.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--data-dir", type=Path, help="Override dataset location")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    config = json.loads((args.run_dir / "config.json").read_text(encoding="utf-8"))
    if config["smoke_test"]:
        data, num_classes = synthetic_graph(config["seeds"][0]), 3
    else:
        dataset = load_dataset(config["dataset"], args.data_dir or Path(config["data_dir"]))
        data, num_classes = dataset[0], dataset.num_classes
    import hashlib
    fingerprint = hashlib.sha256()
    for tensor in (data.x, data.y, data.edge_index):
        fingerprint.update(tensor.contiguous().numpy().tobytes())
    environment = json.loads((args.run_dir / "environment.json").read_text(encoding="utf-8"))
    if fingerprint.hexdigest() != environment["graph_sha256"]:
        parser.error("Dataset fingerprint differs from training graph")
    groups = neighborhoods(data.edge_index, data.num_nodes, config["hops"], config["max_subgraph_nodes"])
    index = pooling_index(groups, args.device)
    data = data.to(args.device)
    model = ACLIBGNN(data.num_features, config["hidden_dim"], num_classes,
                     config["heads"], config["backbone"]).to(args.device)
    directory = args.run_dir / f"seed{args.seed}" / f"fold{args.fold}"
    checkpoint = torch.load(directory / "best.pt", map_location=args.device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    with np.load(directory / "splits.npz") as splits:
        test = splits["test"].copy()
    with torch.no_grad():
        embeddings = model(data.x, data.edge_index)
        pooled = pool(embeddings, index)
        local = model.node_classifier(embeddings)
        combined = (model.node_classifier(pooled) if config["objective"] == "paper" else
                    model.combined_classifier(torch.cat([embeddings, pooled, pooled], dim=1)))
        labels = data.y[test].cpu().numpy()
        result = {name: classification_metrics(labels, logits[test].softmax(-1).cpu().numpy())
                  for name, logits in [("Local", local), ("Combined", combined)]}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
