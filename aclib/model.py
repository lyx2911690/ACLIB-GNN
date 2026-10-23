"""Two-layer encoder and shared node/subgraph prediction head."""
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATConv, GCNConv


class ACLIBGNN(nn.Module):
    def __init__(self, in_features, hidden_dim, num_classes, heads=4, backbone="gat"):
        super().__init__()
        if backbone == "gat":
            self.conv1 = GATConv(in_features, hidden_dim, heads=heads, add_self_loops=False)
            self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, add_self_loops=False)
        elif backbone == "gcn":
            self.conv1 = GCNConv(in_features, hidden_dim * heads, add_self_loops=False)
            self.conv2 = GCNConv(hidden_dim * heads, hidden_dim, add_self_loops=False)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        self.node_classifier = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(),
                                             nn.Linear(64, num_classes))
        self.combined_classifier = nn.Sequential(nn.Linear(3 * hidden_dim, 128), nn.ReLU(),
                                                 nn.Linear(128, num_classes))

    def forward(self, x, edge_index):
        return self.conv2(F.elu(self.conv1(x, edge_index)), edge_index)


def adversarial_embeddings(model, pooled, labels, train_idx, epsilon):
    """FGSM uses training labels only and never populates parameter .grad."""
    delta = torch.zeros_like(pooled, requires_grad=True)
    loss = F.cross_entropy(model.node_classifier(pooled.detach() + delta)[train_idx],
                           labels[train_idx])
    gradient, = torch.autograd.grad(loss, delta)
    return pooled + epsilon * gradient.sign().detach()


def objective(model, embeddings, pooled, adversarial, labels, train_idx,
              alpha, beta, gamma, mode="paper", no_adversarial=False):
    node = F.cross_entropy(model.node_classifier(embeddings)[train_idx], labels[train_idx])
    original = F.cross_entropy(model.node_classifier(pooled)[train_idx], labels[train_idx])
    attacked = F.cross_entropy(model.node_classifier(adversarial)[train_idx], labels[train_idx])
    diversity = F.relu(gamma - (attacked - original))
    if no_adversarial:
        total = node + alpha * original
    elif mode == "paper":
        total = node + alpha * original + beta * attacked + diversity
    else:
        combined = model.combined_classifier(torch.cat([embeddings, pooled, adversarial], dim=1))
        total = node + alpha * F.cross_entropy(combined[train_idx], labels[train_idx]) + beta * original
    return total, {"node": node.item(), "original": original.item(),
                   "adversarial": attacked.item(), "diversity": diversity.item()}
