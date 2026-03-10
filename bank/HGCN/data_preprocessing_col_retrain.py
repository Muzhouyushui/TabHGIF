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
    data_source,            # Accepts either a file path (str) or an existing DataFrame
    is_test: bool = False,
    transformer: ColumnTransformer = None
):
    """
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
    print(f"Original columns: {df.columns.tolist()}")

    # ——— Fill missing categorical values ———
    categorical_cols = [
        "job", "marital", "education", "default",
        "housing", "loan", "contact", "month", "poutcome"
    ]
    df[categorical_cols] = df[categorical_cols].fillna("unknown")

    # ——— Remove the age column to simulate retraining ———
    if 'age' in df.columns:
        df = df.drop(columns=['age'])
        print("[Info] The 'age' column has been removed. New columns: {}".format(df.columns.tolist()))

    # ——— Separate labels & features ———
    raw_labels = df["y"].astype(str).str.strip().values
    feature_df = df.drop(columns=["y"])

    # ——— Build or reuse the transformer ———
    continuous_cols = [
        "balance", "day",
        "duration", "campaign", "pdays", "previous"
    ]  # 'age' has already been removed from the original list

    if transformer is None:
        transformer = ColumnTransformer(transformers=[
            ("discrete",  OneHotEncoder(sparse_output=False), categorical_cols),
            ("continuous", StandardScaler(), continuous_cols),
        ])
        X = transformer.fit_transform(feature_df)
    else:
        X = transformer.transform(feature_df)

    print(f"[Info] Feature matrix shape: {X.shape} (rows, columns) —— the age column has been ignored)")

    # ——— Label mapping & distribution statistics ———
    label_map = {"no": 0, "yes": 1}
    y = [label_map[val] for val in raw_labels]
    num_pos = sum(y)
    num_neg = len(y) - num_pos
    print(f"Label distribution → no: {num_neg}, yes: {num_pos}")

    return X, y, df, transformer


def cluster_nodes_by_similarity_gpu(indices, df, max_nodes, device):
    """
    Use vectorized operations on GPU to greedily cluster the given node indices (from df)
    into several clusters, where each cluster contains at most max_nodes nodes.
    Similarity is defined as the number of identical column values (excluding the label column y).

    Parameters:
      indices    (list): List of node indices in the original df.
      df         (DataFrame): Original data, containing column 'y' as the label column.
      max_nodes  (int): Maximum number of nodes in a single cluster.
      device     (torch.device): Device to use (GPU).

    Returns:
      clusters   (list): Each cluster is a list whose elements are row indices from the original df.
    """
    # Extract the relevant sub-DataFrame and drop the label column "y"
    sub_df = df.loc[indices].drop(columns=["y"])
    n_samples = sub_df.shape[0]

    # Factorize each column and convert it into integer encoding; keep the original order
    encoded_columns = []
    for col in sub_df.columns:
        codes, _ = pd.factorize(sub_df[col])
        # Convert to torch tensor and send to device
        encoded_columns.append(torch.tensor(codes, dtype=torch.long, device=device))

    # Stack to obtain shape (n_samples, n_features)
    X = torch.stack(encoded_columns, dim=1)
    n, d = X.shape

    # Use a boolean mask to record unassigned samples, initially all True
    unassigned = torch.ones(n, dtype=torch.bool, device=device)
    clusters = []
    # For convenient return of original indices, convert input indices to a tensor (CPU is enough)
    orig_indices = torch.tensor(indices, dtype=torch.long)

    while unassigned.any():
        # Select the first unassigned node as the cluster representative
        rep_pos = torch.nonzero(unassigned, as_tuple=False)[0].item()
        current_cluster = [orig_indices[rep_pos].item()]
        unassigned[rep_pos] = False

        # Find row indices of all unassigned nodes
        remaining = torch.nonzero(unassigned, as_tuple=False).squeeze(1)
        if remaining.numel() > 0:
            # Representative vector (1, d)
            rep_vector = X[rep_pos].unsqueeze(0)
            # Candidate vectors (r, d)
            candidates = X[remaining]
            # Compute number of identical features
            sim = (candidates == rep_vector).sum(dim=1)  # shape (r,)

            # Select in descending order of similarity
            _, order = torch.sort(sim, descending=True)
            take = min(max_nodes - 1, order.numel())
            selected = remaining[order[:take]]

            # Add to current cluster and mark
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
    Construct hyperedges for the Bank dataset:
      - Simulate retraining: **skip the 'age' column**, i.e., if categorical_cols/continuous_cols contain 'age', it will be filtered out.
    Return the hyperedge dictionary and print the total number of hyperedges.
    """
    # Filter out age
    categorical_cols = [c for c in categorical_cols if c != 'age']
    continuous_cols  = [c for c in continuous_cols  if c != 'age']

    hyperedges = {}
    total_cluster_time = 0.0

    # Hyperedges for discrete columns
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

    # Hyperedges for continuous columns
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

    # Print hyperedge statistics
    print(f"[Info] Total number of hyperedges: {len(hyperedges)}, clustering time: {total_cluster_time:.4f}s")
    avg_nodes = (sum(len(nodes) for nodes in hyperedges.values()) / len(hyperedges)) if hyperedges else 0
    print(f"[Info] Average number of nodes per hyperedge: {avg_nodes:.2f}")

    return hyperedges