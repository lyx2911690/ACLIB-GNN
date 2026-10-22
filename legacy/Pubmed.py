import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv
from torch_geometric.utils import k_hop_subgraph
from tqdm import tqdm
import os
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
np.random.seed(42)

hidden_dim = 64
num_heads = 4
lr = 0.001
weight_decay = 5e-4
epochs = 100
alpha = 0.7
beta = 0.3
epsilon = 0.01
k_hop = 1
max_subgraph_nodes = 10
results_dir = 'results_Pubmed'
os.makedirs(results_dir, exist_ok=True)

# 数据加载，将数据集名称改为Pubmed
dataset = Planetoid(root='/tmp/Pubmed', name='Pubmed')
data = dataset[0].to(device)
def save_triplets(fold, alpha, triplets):
    triplet_dir = f'{results_dir}/triplets/fold{fold}/alpha{alpha}'
    os.makedirs(triplet_dir, exist_ok=True)
    serializable_triplets = []
    for triplet in triplets:
        serializable_triplets.append([
            int(triplet[0]),
            str(triplet[1]),
            int(triplet[2]),
            int(triplet[3])
        ])

    with open(f'{triplet_dir}/triplets.json', 'w') as f:
        json.dump(serializable_triplets, f)

    df = pd.DataFrame(serializable_triplets,
                      columns=['node1', 'edge', 'node2', 'label'])
    df.to_csv(f'{triplet_dir}/triplets.csv', index=False)


def calculate_metrics(y_true, y_pred, y_score, num_classes):
    metrics = {
        'Accuracy': (y_pred == y_true).mean(),
        'Precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'Recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'F1': f1_score(y_true, y_pred, average='macro', zero_division=0),
    }

    y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))
    metrics['AUC'] = roc_auc_score(y_true_bin, y_score, multi_class='ovr')
    metrics['AUPR'] = average_precision_score(y_true_bin, y_score)
    return metrics


def generate_triplets(sub_nodes, sub_edges, center_node, label):
    triplets = []
    center_node = int(center_node)
    label = int(label)

    for i, (u, v) in enumerate(sub_edges):
        edge_type = f"edge_{i}"
        triplets.append([int(u), edge_type, int(v), label])

    for node in sub_nodes:
        node = int(node)
        if node != center_node:
            edge_type = f"center_to_{node}"
            triplets.append([center_node, edge_type, node, label])

    return triplets


class StructuredGNN(nn.Module):
    def __init__(self, in_feats, hidden_dim, num_classes, num_heads=4):
        super().__init__()
        self.gat1 = GATConv(in_feats, hidden_dim, heads=num_heads, add_self_loops=False)
        self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, add_self_loops=False)

        self.node_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes))

        self.combined_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes))

    def get_subgraph(self, node_idx, edge_index):
        if not isinstance(node_idx, torch.Tensor):
            node_idx = torch.tensor([node_idx], device=device)
        else:
            node_idx = node_idx.to(device)

        sub_nodes, sub_edge_index, _, _ = k_hop_subgraph(
            node_idx, k_hop, edge_index, num_nodes=data.num_nodes)

        if len(sub_nodes) > max_subgraph_nodes:
            sub_nodes = sub_nodes[:max_subgraph_nodes]
            sub_edge_mask = torch.isin(sub_edge_index[0], sub_nodes) & \
                            torch.isin(sub_edge_index[1], sub_nodes)
            sub_edge_index = sub_edge_index[:, sub_edge_mask]

        return sub_nodes, sub_edge_index

    def forward(self, x, edge_index, return_embeddings=False):
        x, _ = self.gat1(x, edge_index, return_attention_weights=True)
        x = F.elu(x)
        node_emb, _ = self.gat2(x, edge_index, return_attention_weights=True)

        if return_embeddings:
            return node_emb
        else:
            return self.node_classifier(node_emb)


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

print(f"\n=== Alpha: {alpha}, Beta: {beta} ===")

for fold, (train_idx, test_idx) in enumerate(skf.split(data.x.cpu(), data.y.cpu())):
    print(f"\n=== Fold {fold + 1}/5 ===")

    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    train_mask[train_idx] = True
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    test_mask[test_idx] = True

    model = StructuredGNN(dataset.num_features, hidden_dim, dataset.num_classes, num_heads).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_acc = 0
    pbar = tqdm(range(epochs), desc=f'Fold {fold + 1} Training')
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()

        node_emb = model(data.x, data.edge_index, return_embeddings=True)
        sub_emb = []
        adv_emb = []
        loss_adv_total = 0

        for n in range(node_emb.size(0)):
            node_idx = torch.tensor([n], device=device)
            sub_nodes, sub_edge_index = model.get_subgraph(node_idx, data.edge_index)
            base_sub_emb = node_emb[sub_nodes].mean(dim=0).unsqueeze(0)

            perturb = torch.zeros_like(base_sub_emb, requires_grad=True)
            adv_pred = model.node_classifier(base_sub_emb + perturb)

            if n in train_idx:
                target = data.y[n].unsqueeze(0).to(device)
                loss_adv = F.cross_entropy(adv_pred, target)
                loss_adv_total += loss_adv
                loss_adv.backward(retain_graph=True)

                with torch.no_grad():
                    perturb = epsilon * perturb.grad.sign()
                    current_adv_emb = base_sub_emb + perturb
            else:
                current_adv_emb = base_sub_emb

            sub_emb.append(base_sub_emb.squeeze(0))
            adv_emb.append(current_adv_emb.squeeze(0))

        loss_adv_total = loss_adv_total / len(train_idx)

        sub_emb = torch.stack(sub_emb)
        adv_emb = torch.stack(adv_emb)
        combined_feat = torch.cat([node_emb, sub_emb, adv_emb], dim=1)
        combined_pred = model.combined_classifier(combined_feat)

        node_pred = model(data.x, data.edge_index)

        loss_node = F.cross_entropy(node_pred[train_mask], data.y[train_mask])
        loss_combined = F.cross_entropy(combined_pred[train_mask], data.y[train_mask])
        total_loss = loss_node + alpha * loss_combined + beta * loss_adv_total

        total_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            node_pred_val = model(data.x, data.edge_index)
            combined_pred_val = model.combined_classifier(torch.cat([
                model(data.x, data.edge_index, return_embeddings=True),
                sub_emb,
                adv_emb
            ], dim=1))

            val_acc = (combined_pred_val.argmax(1)[test_mask] == data.y[test_mask]).float().mean().item()
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), f'{results_dir}/best_fold{fold}_alpha{alpha}.pth')

            pbar.set_postfix({'loss': total_loss.item(), 'val_acc': val_acc})

    model.load_state_dict(torch.load(f'{results_dir}/best_fold{fold}_alpha{alpha}.pth'))
    model.eval()
    with torch.no_grad():
        node_pred = model(data.x, data.edge_index)
        combined_pred = model.combined_classifier(torch.cat([
            model(data.x, data.edge_index, return_embeddings=True),
            sub_emb,
            adv_emb
        ], dim=1))

        y_true = data.y[test_mask].cpu().numpy()

        node_metrics = calculate_metrics(
            y_true,
            node_pred.argmax(1)[test_mask].cpu().numpy(),
            node_pred[test_mask].cpu().numpy(),
            dataset.num_classes)

        combined_metrics = calculate_metrics(
            y_true,
            combined_pred.argmax(1)[test_mask].cpu().numpy(),
            combined_pred[test_mask].cpu().numpy(),
            dataset.num_classes)

        results.append({
            'Fold': fold + 1,
            'Alpha': alpha,
            'Beta': beta,
            'Local_Accuracy': node_metrics['Accuracy'],
            'Combined_Accuracy': combined_metrics['Accuracy'],
            'Local_Precision': node_metrics['Precision'],
            'Combined_Precision': combined_metrics['Precision'],
            'Local_Recall': node_metrics['Recall'],
            'Combined_Recall': combined_metrics['Recall'],
            'Local_F1': node_metrics['F1'],
            'Combined_F1': combined_metrics['F1'],
            'Local_AUC': node_metrics['AUC'],
            'Combined_AUC': combined_metrics['AUC'],
            'Local_AUPR': node_metrics['AUPR'],
            'Combined_AUPR': combined_metrics['AUPR']
        })

df_results = pd.DataFrame(results)
df_results.to_csv(f'{results_dir}/all_metrics.csv', index=False)
print(df_results.groupby(['Alpha', 'Beta']).mean().reset_index()[
          ['Alpha', 'Beta',
           'Local_Accuracy', 'Combined_Accuracy',
           'Local_F1', 'Combined_F1',
           'Local_AUC', 'Combined_AUC']].to_string(index=False))
