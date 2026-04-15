import os
from functools import lru_cache

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv
from torch_geometric.data import Data

from .market_data import get_crypto_data, get_stock_data

FEATURE_DIM = 6

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "ml_models", "temporal_gnn_anomaly_model.pth")


# ==========================================
# RSI
# ==========================================
def compute_rsi(series, period=14):
    delta = series.diff()

    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()

    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


# ==========================================
# MAIN FEATURE BUILDER (STATIC SCAN)
# ==========================================
def build_features_from_history(df: pd.DataFrame, symbol=None, market_type=None):
    """
    Used for normal scan page (historical scan).
    This keeps your current scan logic intact.
    """
    if df is None or df.empty or len(df) < 30:
        raise ValueError("Not enough history")

    df = df.copy()

    # ===============================
    # LIVE DATA INJECTION (only 1 point for scan freshness)
    # ===============================
    try:
        live = None

        if market_type == "crypto" and symbol:
            live = get_crypto_data(symbol.lower())
        elif market_type == "stock" and symbol:
            live = get_stock_data(symbol.upper())

        if live and live.get("price") is not None and live.get("volume") is not None:
            df.loc[len(df)] = {
                "price": float(live.get("price")),
                "volume": float(live.get("volume"))
            }

    except Exception:
        pass

    # ===============================
    # FEATURE ENGINEERING
    # ===============================
    df["return"] = df["price"].pct_change()
    df["ma7"] = df["price"].rolling(7).mean()
    df["ma30"] = df["price"].rolling(30).mean()
    df["volatility_7d"] = df["return"].rolling(7).std()
    df["volume_spike"] = df["volume"] / (df["volume"].rolling(7).mean() + 1e-9)
    df["momentum"] = df["price"] - df["price"].shift(5)

    # extra features (RF only)
    df["volume_change"] = df["volume"].pct_change()
    df["price_acceleration"] = df["return"].diff()

    # ===============================
    # NORMALIZATION
    # ===============================
    if market_type == "crypto":
        vol_scale = 1.0
        vol_ref = 0.05
    elif market_type == "stock":
        vol_scale = 3.0
        vol_ref = 0.015
    else:
        vol_scale = 1.0
        vol_ref = 0.03

    rolling_vol = df["volume"].rolling(30).mean()
    df["volume"] = df["volume"] / (rolling_vol + 1e-9)

    df["volatility_7d"] = df["volatility_7d"] / (vol_ref + 1e-9)
    df["return"] = df["return"] * vol_scale
    df["momentum"] = df["momentum"] * vol_scale

    df = df.bfill().ffill()

    # ===============================
    # RF FEATURES
    # ===============================
    latest = df.iloc[-1]

    rf_features = {
        "price": float(latest["price"]),
        "volume": float(latest["volume"]),
        "volatility_7d": float(latest["volatility_7d"]),
        "daily_return": float(latest["return"]),
        "price_ma7": float(latest["ma7"]),
        "price_ma30": float(latest["ma30"]),
        "post_count": float(latest.get("post_count", 0))
    }

    # ===============================
    # GNN SEQUENCE (STRICT 6 FEATURES)
    # ===============================
    sequence_df = df.tail(20)

    gnn_sequence = []
    for _, row in sequence_df.iterrows():
        gnn_sequence.append([
            float(row["return"]),
            float(row["volatility_7d"]),
            float(row["momentum"]),
            float(row["volume_spike"]),
            float(row["price"] - row["ma7"]),
            0.0
        ])

    return rf_features, gnn_sequence


# ==========================================
# LIVE FEATURE BUILDER (REAL FIX)
# ==========================================
def build_live_sequence_from_buffer(buffer, market_type=None):
    """
    Used ONLY for live preview.
    Builds features from rolling live points instead of static history.
    """
    if not buffer or len(buffer) < 6:
        raise ValueError("Not enough live points")

    df = pd.DataFrame(buffer).copy()

    if "price" not in df.columns or "volume" not in df.columns:
        raise ValueError("Live buffer missing required columns")

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df = df.dropna(subset=["price", "volume"])

    if len(df) < 6:
        raise ValueError("Not enough valid live points")

    # ===============================
    # FEATURE ENGINEERING
    # ===============================
    df["return"] = df["price"].pct_change()
    df["ma7"] = df["price"].rolling(7).mean()

    # shorter rolling windows for live behavior
    df["ma30"] = df["price"].rolling(10).mean()
    df["volatility_7d"] = df["return"].rolling(7).std()
    df["volume_spike"] = df["volume"] / (df["volume"].rolling(7).mean() + 1e-9)
    df["momentum"] = df["price"] - df["price"].shift(3)

    # ===============================
    # NORMALIZATION
    # ===============================
    if market_type == "crypto":
        vol_scale = 1.0
        vol_ref = 0.05
    elif market_type == "stock":
        vol_scale = 3.0
        vol_ref = 0.015
    else:
        vol_scale = 1.0
        vol_ref = 0.03

    rolling_vol = df["volume"].rolling(10).mean()
    df["volume"] = df["volume"] / (rolling_vol + 1e-9)
    df["volatility_7d"] = df["volatility_7d"] / (vol_ref + 1e-9)
    df["return"] = df["return"] * vol_scale
    df["momentum"] = df["momentum"] * vol_scale

    df = df.bfill().ffill()

    latest = df.iloc[-1]

    rf_features = {
        "price": float(latest["price"]),
        "volume": float(latest["volume"]),
        "volatility_7d": float(latest["volatility_7d"]),
        "daily_return": float(latest["return"]),
        "price_ma7": float(latest["ma7"]),
        "price_ma30": float(latest["ma30"]),
        "post_count": 0.0
    }

    sequence_df = df.tail(20)

    gnn_sequence = []
    for _, row in sequence_df.iterrows():
        gnn_sequence.append([
            float(row["return"]),
            float(row["volatility_7d"]),
            float(row["momentum"]),
            float(row["volume_spike"]),
            float(row["price"] - row["ma7"]),
            0.0
        ])

    return rf_features, gnn_sequence


# ==========================================
# GNN MODEL
# ==========================================
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


# ==========================================
# BUILD GRAPH
# ==========================================
def build_graph(features_sequence):
    x = torch.tensor(features_sequence, dtype=torch.float32)

    edge_index = []
    edge_attr = []

    for i in range(len(x) - 1):
        edge_index.append([i, i + 1])

        # weight edges by recency
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


# ==========================================
# ANOMALY SCORE
# ==========================================
def compute_anomaly_score(sequence_features):
    model = load_model()

    if model is None or len(sequence_features) < 2:
        return 0.0

    graph = build_graph(sequence_features)

    with torch.no_grad():
        out = model(graph.x, graph.edge_index, graph.edge_attr)

    score = out.max().item()

    # reduce extreme spikes
    return float(1 / (1 + np.exp(-score * 0.4)))