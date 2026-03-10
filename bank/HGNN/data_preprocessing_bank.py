# data_preprocessing_bank.py

import time
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# Modified:
# Modified:
def preprocess_node_features_bank(
    data_source,            # Can accept either a "file path" (str) or an existing DataFrame
    is_test: bool = False,
    transformer: ColumnTransformer = None
):
    """
    Load and preprocess the Bank dataset. If data_source is a str, it is treated as a CSV path;
    if it is already a pd.DataFrame, it is copied and used directly.
    """
    # ——— Column name definitions ———
    col_names = [
        "age", "job", "marital", "education", "default",
        "balance", "housing", "loan", "contact", "day",
        "month", "duration", "campaign", "pdays",
        "previous", "poutcome", "y"
    ]

    # —— Support either a str path or a DataFrame —— #
    if isinstance(data_source, pd.DataFrame):
        df = data_source.copy()
    else:
        df = pd.read_csv(
            data_source,
            sep=';',
            header=0,
            names=col_names,
            skipinitialspace=True
        )

    # ——— Fill missing categorical values ———
    categorical_cols = [
        "job", "marital", "education", "default",
        "housing", "loan", "contact", "month", "poutcome"
    ]
    df[categorical_cols] = df[categorical_cols].fillna("unknown")

    # ——— Separate labels & features ———
    raw_labels = df["y"].str.strip().values
    feature_df = df.drop(columns=["y"])

    # ——— Build or reuse transformer ———
    continuous_cols = [
        "age", "balance", "day",
        "duration", "campaign", "pdays", "previous"
    ]
    if transformer is None:
        transformer = ColumnTransformer(transformers=[
            ("discrete",  OneHotEncoder(sparse_output=False), categorical_cols),
            ("continuous", StandardScaler(),           continuous_cols)
        ])
        X = transformer.fit_transform(feature_df)
    else:
        X = transformer.transform(feature_df)

    # ——— Label mapping & distribution statistics ———
    label_map = {"no": 0, "yes": 1}
    y = [label_map[val] for val in raw_labels]
    num_pos = sum(y)
    num_neg = len(y) - num_pos
    print(f"Label distribution → no: {num_neg}, yes: {num_pos}")

    return X, y, df, transformer

def generate_hyperedge_dict_bank(
    df: pd.DataFrame,
    categorical_cols: list,
    continuous_cols: list,
    max_nodes_per_hyperedge: int,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    """
    Construct hyperedges for the Bank dataset:
      1. Categorical columns: group by value, and call GPU clustering for splitting when node count exceeds the limit
      2. Continuous columns: uniformly use 10 equal-width bins, and do the same when exceeding the limit
      Also print clustering time and average number of nodes.

    Returns:
      hyperedges (dict) keys are (col, val[, cluster_id]), values are lists of node indices
    """
    hyperedges = {}
    col_edge_count = {}
    total_cluster_time = 0.0

    # ——— 1) Hyperedges for categorical columns ———
    for col in categorical_cols:
        start_count = len(hyperedges)
        unique_vals = df[col].unique()
        for val in unique_vals:
            indices = df.index[df[col] == val].tolist()
            if len(indices) <= max_nodes_per_hyperedge:
                hyperedges[(col, str(val))] = indices
            else:
                t0 = time.time()
                clusters = cluster_nodes_by_similarity_gpu(indices, df, max_nodes_per_hyperedge, device)
                total_cluster_time += (time.time() - t0)
                for cid, cluster_idxs in enumerate(clusters):
                    hyperedges[(col, str(val), cid)] = cluster_idxs
        col_edge_count[col] = len(hyperedges) - start_count

    # ——— 2) Hyperedges for continuous columns ———
    # Uniformly use 10 equal-width bins
    for col in continuous_cols:
        start_count = len(hyperedges)
        min_v, max_v = df[col].min(), df[col].max()
        bins = [min_v + (max_v - min_v) * i / 10 for i in range(11)]
        for i in range(10):
            lower, upper = bins[i], bins[i+1]
            idxs = df.index[(df[col] >= lower) & (df[col] < upper)].tolist()
            label = f"{lower:.2f}-{upper:.2f}"
            if len(idxs) <= max_nodes_per_hyperedge:
                hyperedges[(col, label)] = idxs
            else:
                t0 = time.time()
                clusters = cluster_nodes_by_similarity_gpu(idxs, df, max_nodes_per_hyperedge, device)
                total_cluster_time += (time.time() - t0)
                for cid, cluster_idxs in enumerate(clusters):
                    hyperedges[(col, label, cid)] = cluster_idxs
        col_edge_count[col] = len(hyperedges) - start_count

    # ——— Summary print ———
    print(f"Subgraph (clustering) division total time (GPU): {total_cluster_time:.4f} sec.")
    total_edges = len(hyperedges)
    if total_edges:
        avg_nodes = sum(len(nodes) for nodes in hyperedges.values()) / total_edges
        print(f"Average nodes per hyperedge: {avg_nodes:.2f}")
    else:
        print("No hyperedges generated.")

    return hyperedges


# ——— GPU clustering helper function remains unchanged ———
def cluster_nodes_by_similarity_gpu(indices, df, max_nodes, device):
    """
    Same as the ACI version: perform greedy clustering on the specified indices in df
    based on row similarity (number of exactly matched columns).
    """
    sub_df = df.loc[indices].drop(columns=["y"])
    encoded = []
    for col in sub_df.columns:
        codes, _ = pd.factorize(sub_df[col])
        encoded.append(torch.tensor(codes, dtype=torch.long, device=device))
    X = torch.stack(encoded, dim=1)
    n = X.size(0)
    unassigned = torch.ones(n, dtype=torch.bool, device=device)
    orig_idx = torch.tensor(indices, dtype=torch.long)
    clusters = []
    while unassigned.any():
        rep = torch.nonzero(unassigned, as_tuple=False)[0].item()
        cluster = [orig_idx[rep].item()]
        unassigned[rep] = False
        rem = torch.nonzero(unassigned, as_tuple=False).squeeze(1)
        if rem.numel():
            rep_vec = X[rep].unsqueeze(0)
            cand   = X[rem]
            sim    = (cand == rep_vec).sum(dim=1)
            _, order = torch.sort(sim, descending=True)
            take = min(max_nodes - 1, order.numel())
            selected = rem[order[:take]]
            for pos in selected.tolist():
                cluster.append(orig_idx[pos].item())
                unassigned[pos] = False
        clusters.append(cluster)
    return clusters