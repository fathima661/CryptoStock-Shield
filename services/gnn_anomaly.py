import torch
import torch.nn as nn
import numpy as np
import os
from functools import lru_cache
from torch_geometric.nn import TransformerConv
from torch_geometric.data import Data

FEATURE_DIM = 6

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "ml_models", "temporal_gnn_anomaly_model.pth")


class TemporalGNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.edge_encoder = nn.Linear(1, 64)

        self.conv1 = TransformerConv(6, 64, heads=2, edge_dim=64)
        self.conv2 = TransformerConv(128, 64, heads=2, edge_dim=64)

        self.fc = nn.Linear(128, 1)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, edge_attr):
        edge_emb = self.edge_encoder(edge_attr)

        x = self.relu(self.conv1(x, edge_index, edge_emb))
        x = self.relu(self.conv2(x, edge_index, edge_emb))

        return self.fc(x)


@lru_cache()
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None

    model = TemporalGNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


def build_graph(features_sequence):
    x = torch.tensor(features_sequence, dtype=torch.float32)

    edge_index = []
    edge_attr = []

    for i in range(len(x) - 1):
        edge_index.append([i, i + 1])

        # 🔥 NEW: weight edges by recency
        weight = (i + 1) / len(x)
        edge_attr.append([weight])

    if not edge_index:
        edge_index = [[0, 0]]
        edge_attr = [[1.0]]

    return Data(
        x=x,
        edge_index=torch.tensor(edge_index).t().contiguous(),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32)
    )

def compute_anomaly_score(sequence_features):
    model = load_model()
    # Safety: GNN needs at least 2 nodes to create an edge
    if model is None or len(sequence_features) < 2:
        return 0.0
    
    graph = build_graph(sequence_features)

    with torch.no_grad():
        out = model(graph.x, graph.edge_index, graph.edge_attr)

    score = out.max().item()
    # reduce extreme spikes
    return float(1 / (1 + np.exp(-score * 0.4)))