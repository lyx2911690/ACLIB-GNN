"""Dataset loading and deterministic, label-free neighborhood pooling."""
import torch
from torch_geometric.datasets import Coauthor, Planetoid
from torch_geometric.utils import k_hop_subgraph


def synthetic_graph(seed):
    from torch_geometric.data import Data
    generator = torch.Generator().manual_seed(seed)
    source = torch.arange(90)
    edges = torch.stack([source, (source + 1) % 90])
    return Data(x=torch.randn(90, 8, generator=generator), y=source % 3,
                edge_index=torch.cat([edges, edges.flip(0)], dim=1))


def load_dataset(name, root):
    if name == "CS":
        return Coauthor(root=str(root / name), name="CS")
    return Planetoid(root=str(root / name), name=name)


def neighborhoods(edge_index, num_nodes, hops=1, max_nodes=10):
    """Cache k-hop membership once; retain center, then ascending node IDs.

    max_nodes=0 disables the cap. No attention ranking is claimed here.
    """
    edge_index = edge_index.cpu()
    neighbors = [[] for _ in range(num_nodes)]
    if hops == 1:
        for source, target in edge_index.t().tolist():
            neighbors[target].append(source)
    groups = []
    for center in range(num_nodes):
        if hops == 1:
            others = sorted(set(neighbors[center]) - {center})
        else:
            nodes, _, _, _ = k_hop_subgraph(center, hops, edge_index, num_nodes=num_nodes)
            others = [n for n in nodes.tolist() if n != center]
        members = [center] + others
        groups.append(members[:max_nodes] if max_nodes else members)
    return groups


def pooling_index(groups, device):
    centers, members = [], []
    for center, group in enumerate(groups):
        centers.extend([center] * len(group))
        members.extend(group)
    return (torch.tensor(centers, device=device), torch.tensor(members, device=device),
            torch.tensor([len(g) for g in groups], device=device).unsqueeze(1))


def pool(embeddings, index):
    centers, members, counts = index
    result = embeddings.new_zeros(embeddings.shape)
    return result.index_add(0, centers, embeddings[members]) / counts
