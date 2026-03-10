"""
"""
import re
import time
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import numpy as np
import torch
from scipy.sparse import coo_matrix
from bank.HGCN.HGCN import laplacian


def preprocess_node_features(
    data_source,
    is_test: bool = False,
    transformer: ColumnTransformer = None
):
    """
    """
    col_names = [
        "age", "job", "marital", "education", "default",
        "balance", "housing", "loan", "contact", "day",
        "month", "duration", "campaign", "pdays",
        "previous", "poutcome", "y"
    ]

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
    print(f"old col: {df.columns.tolist()}")

    categorical_cols = [
        "job", "marital", "education", "default",
        "housing", "loan", "contact", "month", "poutcome"
    ]
    df[categorical_cols] = df[categorical_cols].fillna("unknown")

    if 'age' in df.columns:
        df = df.drop(columns=['age'])
        print("[Info] 'age' col deleted,，new col: {}".format(df.columns.tolist()))

    raw_labels = df["y"].astype(str).str.strip().values
    feature_df = df.drop(columns=["y"])

    continuous_cols = [
        "balance", "day",
        "duration", "campaign", "pdays", "previous"
    ]

    if transformer is None:
        transformer = ColumnTransformer(transformers=[
            ("discrete",  OneHotEncoder(sparse_output=False), categorical_cols),
            ("continuous", StandardScaler(), continuous_cols),
        ])
        X = transformer.fit_transform(feature_df)
    else:
        X = transformer.transform(feature_df)

    label_map = {"no": 0, "yes": 1}
    y = [label_map[val] for val in raw_labels]
    num_pos = sum(y)
    num_neg = len(y) - num_pos
    print(f"label → no: {num_neg}, yes: {num_pos}")

    return X, y, df, transformer


def cluster_nodes_by_similarity_gpu(indices, df, max_nodes, device):
    """
    """
    sub_df = df.loc[indices].drop(columns=["y"])
    n_samples = sub_df.shape[0]

    encoded_columns = []
    for col in sub_df.columns:
        codes, _ = pd.factorize(sub_df[col])
        encoded_columns.append(torch.tensor(codes, dtype=torch.long, device=device))

    X = torch.stack(encoded_columns, dim=1)
    n, d = X.shape

    unassigned = torch.ones(n, dtype=torch.bool, device=device)
    clusters = []
    orig_indices = torch.tensor(indices, dtype=torch.long)

    while unassigned.any():
        rep_pos = torch.nonzero(unassigned, as_tuple=False)[0].item()
        current_cluster = [orig_indices[rep_pos].item()]
        unassigned[rep_pos] = False

        remaining = torch.nonzero(unassigned, as_tuple=False).squeeze(1)
        if remaining.numel() > 0:
            rep_vector = X[rep_pos].unsqueeze(0)
            candidates = X[remaining]
            sim = (candidates == rep_vector).sum(dim=1)

            _, order = torch.sort(sim, descending=True)
            take = min(max_nodes - 1, order.numel())
            selected = remaining[order[:take]]

            for pos in selected.tolist():
                current_cluster.append(orig_indices[pos].item())
                unassigned[pos] = False

        clusters.append(current_cluster)

    return clusters


def generate_hyperedge_dict(
    df: pd.DataFrame,
    categorical_cols: list,
    continuous_cols: list,
    max_nodes_per_hyperedge: int,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    """
    """
    # 过滤 age
    categorical_cols = [c for c in categorical_cols if c != 'age']
    continuous_cols  = [c for c in continuous_cols  if c != 'age']

    hyperedges = {}
    total_cluster_time = 0.0

    for col in categorical_cols:
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

    for col in continuous_cols:
        min_v, max_v = df[col].min(), df[col].max()
        bins = [min_v + (max_v - min_v) * i / 10 for i in range(11)]
        for i in range(10):
            lower, upper = bins[i], bins[i+1]
            idxs = df.index[(df[col] >= lower) & (df[col] < upper)].tolist()
            if len(idxs) <= max_nodes_per_hyperedge:
                hyperedges[(col, f"{lower:.2f}-{upper:.2f}")] = idxs
            else:
                t0 = time.time()
                clusters = cluster_nodes_by_similarity_gpu(idxs, df, max_nodes_per_hyperedge, device)
                total_cluster_time += (time.time() - t0)
                for cid, cluster_idxs in enumerate(clusters):
                    hyperedges[(col, f"{lower:.2f}-{upper:.2f}", cid)] = cluster_idxs

    print(f"[Info] Total number of hyperedges: {len(hyperedges)}, clustering time: {total_cluster_time:.4f}s")
    avg_nodes = (sum(len(nodes) for nodes in hyperedges.values()) / len(hyperedges)) if hyperedges else 0
    print(f"[Info] Average number of nodes per hyperedge: {avg_nodes:.2f}")

    return hyperedges
