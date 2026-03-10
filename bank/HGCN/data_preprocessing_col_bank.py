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

    categorical_cols = [
        "job", "marital", "education", "default",
        "housing", "loan", "contact", "month", "poutcome"
    ]
    df[categorical_cols] = df[categorical_cols].fillna("unknown")

    raw_labels = df["y"].astype(str).str.strip().values
    feature_df = df.drop(columns=["y"])

    continuous_cols = [
        "age", "balance", "day",
        "duration", "campaign", "pdays", "previous"
    ]
    if transformer is None:
        transformer = ColumnTransformer(transformers=[
            ("discrete",  OneHotEncoder(sparse_output=False), categorical_cols),
            ("continuous", StandardScaler(),             continuous_cols),
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
    hyperedges = {}
    col_edge_count = {}
    total_cluster_time = 0.0

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

    # ——— 2) 连续列超边 ———
    # 统一使用 10 等分
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

    print(f"Subgraph (clustering) division total time (GPU): {total_cluster_time:.4f} sec.")
    total_edges = len(hyperedges)
    if total_edges:
        avg_nodes = sum(len(nodes) for nodes in hyperedges.values()) / total_edges
        print(f"Average nodes per hyperedge: {avg_nodes:.2f}")
    else:
        print("No hyperedges generated.")

    return hyperedges




def delete_feature_columns_hgcn(
    X_tensor: torch.Tensor,
    transformer,
    column_names: list,
    hyperedges: dict,
    mediators=True,
    use_cuda=False
):
    """
    """
    try:
        feat_names = transformer.get_feature_names_out()
    except AttributeError:
        feat_names = transformer.get_feature_names()
    feat_names = feat_names.tolist()

    del_idx = []
    for col in column_names:
        idx = [
            i for i, f in enumerate(feat_names)
            if col in re.split(r'__|_', str(f))
        ]
        del_idx.extend(idx)
        if idx:
            print(f"[zero] Zeroing out {len(idx)} features for column '{col}' → indices {idx}")
        else:
            print(f"[warning] Column '{col}' not found in feature names!")

    if del_idx:
        nonzero_before = (X_tensor[:, del_idx] != 0.0).sum().item()
        X_tensor[:, del_idx] = 0.0
        nonzero_after  = (X_tensor[:, del_idx] != 0.0).sum().item()
        print(f"[verify] Non-zero entries for these features before: {nonzero_before}, after: {nonzero_after}")
        if nonzero_after == 0:
            print(f"[verify] Features for {column_names} successfully zeroed.")
        else:
            print(f"[error] Some features for {column_names} failed to zero out!")

    total_before = len(hyperedges)
    new_hyper = {k: v for k, v in hyperedges.items() if k[0] not in column_names}
    removed = total_before - len(new_hyper)
    print(f"Remaining hyperedges: {len(new_hyper)}")
    print(f"Deleted {removed} hyperedges related to {column_names}")

    edge_list_new = list(new_hyper.values())
    if isinstance(X_tensor, torch.Tensor):
        X_init = X_tensor.cpu().numpy()
    else:
        X_init = X_tensor
    A_new = laplacian(edge_list_new, X_init, mediators)
    if use_cuda:
        A_new = A_new.cuda()
    total_before = len(hyperedges)
    new_hyper = {k: v for k, v in hyperedges.items() if k[0] not in column_names}
    removed = total_before - len(new_hyper)
    print(f"Remaining hyperedges: {len(new_hyper)}")
    print(f"Deleted {removed} hyperedges related to {column_names}")
    return X_tensor, new_hyper, A_new
