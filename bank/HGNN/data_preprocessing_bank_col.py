"""
data_preprocessing.py

This module is mainly used to load data from CSV files and preprocess the data:
  - Apply One-Hot encoding to discrete attributes;
  - Apply standardization to continuous attributes;
  - Separate labels and features;
  - Construct the dictionary required for hyperedge generation.

The CSV file is expected to contain 15 columns in the following order:
 1. age              (continuous)
 2. workclass        (categorical)
 3. fnlwgt           (continuous)
 4. education        (categorical)
 5. education-num    (continuous)
 6. marital-status   (categorical)
 7. occupation       (categorical)
 8. relationship     (categorical)
 9. race             (categorical)
10. sex              (categorical)
11. capital-gain     (continuous)
12. capital-loss     (continuous)
13. hours-per-week   (continuous)
14. native-country   (categorical)
15. income           (label)

When the parameter is_test is True, the first row of the CSV file
(header or comment) is skipped by default.
"""
import re
import time
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

import numpy as np
import torch
from scipy.sparse import coo_matrix

def build_incidence_matrix(hyperedges, num_nodes):
    """
    Build the incidence matrix of the hypergraph.

    Parameters:
    hyperedges (dict): Hyperedge dictionary, where the keys are (attribute, value)
                       or (attribute, value, subcluster_id), and the values are
                       node indices associated with the hyperedge.
    num_nodes (int): Number of nodes

    Returns:
    H_sparse (scipy.sparse.coo_matrix): Incidence matrix of the hypergraph
    """
    row_indices = []
    col_indices = []
    data = []

    # Build the incidence matrix of the hypergraph
    for col_val, nodes in hyperedges.items():
        for node in nodes:
            row_indices.append(node)
            col_indices.append(list(hyperedges.keys()).index(col_val))  # hyperedge index
            data.append(1)

    # Use coo_matrix to create a sparse matrix
    H_sparse = coo_matrix((data, (row_indices, col_indices)), shape=(num_nodes, len(hyperedges)))

    return H_sparse

def preprocess_node_features_bank(
    data_source,            # Can accept either a "file path" (str) or an existing DataFrame
    is_test: bool = False,
    transformer: ColumnTransformer = None
):
    """
    Load and preprocess the Bank dataset. If data_source is a str, it will be treated as a CSV path;
    if it is already a pd.DataFrame, it will be copied and used directly.
    """
    # ——— Column name definitions ———
    col_names = [
        "age", "job", "marital", "education", "default",
        "balance", "housing", "loan", "contact", "day",
        "month", "duration", "campaign", "pdays",
        "previous", "poutcome", "y"
    ]

    # —— Support str path or DataFrame —— #
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




def cluster_nodes_by_similarity_gpu(indices, df, max_nodes, device):
    """
    Use vectorized operations on GPU to greedily cluster the given node indices
    (from df) into several clusters, where each cluster contains at most max_nodes nodes.
    Similarity is defined as the number of identical column values
    (excluding the label column y).

    Parameters:
      indices    (list): List of node indices in the original df.
      df         (DataFrame): Original data containing column 'y' as the label column.
      max_nodes  (int): Maximum number of nodes in a single cluster.
      device     (torch.device): Device to use (GPU).

    Returns:
      clusters   (list): Each cluster is a list whose elements are row indices of the original df.
    """
    # Extract the relevant sub-DataFrame and remove the label column "y"
    sub_df = df.loc[indices].drop(columns=["y"])
    n_samples = sub_df.shape[0]

    # Factorize each column into integer encoding while keeping the original order
    encoded_columns = []
    for col in sub_df.columns:
        codes, _ = pd.factorize(sub_df[col])
        # Convert to torch tensor and send to device
        encoded_columns.append(torch.tensor(codes, dtype=torch.long, device=device))

    # Stack into shape (n_samples, n_features)
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
            # Compute the number of identical features
            sim = (candidates == rep_vector).sum(dim=1)  # shape (r,)

            # Select in descending order of similarity
            _, order = torch.sort(sim, descending=True)
            take = min(max_nodes - 1, order.numel())
            selected = remaining[order[:take]]

            # Add into the current cluster and mark as assigned
            for pos in selected.tolist():
                current_cluster.append(orig_indices[pos].item())
                unassigned[pos] = False

        clusters.append(current_cluster)

    return clusters


def delete_feature_column(X_tensor, transformer, column_name,
                          H_tensor, hyperedges, continuous_cols=None):
    """
    • Set all feature dimensions corresponding to column_name to 0
    • Remove hyperedges related to this column and return the new sparse incidence tensor H_tensor
    • continuous_cols is passed in to locate the index range of continuous columns in the transformer
    """
    # ---------- 1. Find the column indices to be zeroed ----------
    idx_to_zero = []

    # 1-a Discrete columns (One-Hot)
    onehot = transformer.named_transformers_['discrete']
    disc_col_names = transformer.transformers_[0][2]        # order of discrete column names
    if column_name in disc_col_names:
        oh_names = onehot.get_feature_names_out(disc_col_names)
        idx_to_zero += [i for i, n in enumerate(oh_names)
                        if n.startswith(column_name + '_')]

    # 1-b Continuous columns (StandardScaler)
    if continuous_cols is None:
        continuous_cols = transformer.transformers_[1][2]
    if column_name in continuous_cols:
        # Starting position of continuous features in overall X = total discrete feature dimension
        disc_dim = onehot.get_feature_names_out(disc_col_names).shape[0]
        cont_idx = continuous_cols.index(column_name)
        idx_to_zero.append(disc_dim + cont_idx)

    if not idx_to_zero:
        raise ValueError(f"Column {column_name} not found in transformer.")

    # ---------- 2. Zero out ----------
    X_tensor[:, idx_to_zero] = 0.0

    # ---------- 3. Remove related hyperedges ----------
    new_hyper = {k: v for k, v in hyperedges.items()
                 if k[0] != column_name}

    # Output the number of remaining hyperedges after deletion
    print(f"Remaining hyperedges after deleting column '{column_name}': {len(new_hyper)}")

    H_sparse = build_incidence_matrix(new_hyper, X_tensor.shape[0])
    H_coo = H_sparse.tocoo()
    indices = torch.LongTensor(np.vstack((H_coo.row, H_coo.col)))
    values  = torch.FloatTensor(H_coo.data)
    H_tensor = torch.sparse_coo_tensor(indices, values,
                                       size=H_coo.shape,
                                       device=X_tensor.device)
    print(f"Deleted {H_tensor.shape[1]} hyperedges remain after removing column '{column_name}'.")

    return X_tensor, H_tensor, new_hyper



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
      hyperedges (dict) keys are (col, val[, cluster_id]), values are node index lists
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