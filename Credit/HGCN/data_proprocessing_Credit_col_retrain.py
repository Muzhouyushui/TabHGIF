import time
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from typing import Union, Tuple, List
from Credit.HGCN.HGCN import laplacian
import re


def preprocess_node_features(
    data: Union[str, pd.DataFrame],
    transformer: ColumnTransformer = None,
    drop_cols: List[str] = None
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, ColumnTransformer]:
    """
    Read and preprocess the UCI Credit Approval dataset.
    Args:
      data: File path or an existing DataFrame.
      transformer: A fitted ColumnTransformer.
      drop_cols: List of original feature columns to simulate deletion (e.g., ['A5']).
    Returns:
      X: (n_samples, n_feats) numeric feature matrix (numpy array).
      y: (n_samples,) 0/1 labels.
      df: (n_samples, original columns - deleted columns) DataFrame used for hyperedge construction.
      transformer: The fitted preprocessor.
    """
    # 1. Read the DataFrame and assign column names
    if isinstance(data, str):
        df = pd.read_csv(data, header=None, na_values='?')
    else:
        df = data.copy()
    df = df.reset_index(drop=True)
    df.columns = [
        "A1","A2","A3","A4","A5","A6","A7","A8","A9",
        "A10","A11","A12","A13","A14","A15","class"
    ]

    # 2. Delete specified columns
    if drop_cols:
        for col in drop_cols:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)
        print(f"[Info] Remaining columns after deletion: {df.columns.tolist()}")

    # 3. Split features/labels
    feature_cols = [c for c in df.columns if c != "class"]
    # Update numerical and categorical column lists
    num_cols = [c for c in ["A2","A3","A8","A11","A14","A15"] if c in feature_cols]
    cat_cols = [c for c in feature_cols if c not in num_cols]

    y = df["class"].map({"+":1, "-":0}).astype(int).values
    X_df = df[feature_cols]

    # 4. Feature preprocessing
    if transformer is None:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler())
        ])
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot",   OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        transformer = ColumnTransformer([
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ])
        X = transformer.fit_transform(X_df)
    else:
        X = transformer.transform(X_df)

    print(f"[Preprocess] Samples={df.shape[0]} Pos={y.sum()} Neg={(y==0).sum()} -> Feature dimension={X.shape[1]}")
    return X, y, df, transformer


def generate_hyperedge_dict(
    df: pd.DataFrame,
    feature_cols: List[str] = None,
    max_nodes_per_hyperedge: int = 50,
    device: torch.device = torch.device("cpu"),
    drop_cols: List[str] = None
) -> dict:
    """
    Generate hyperedges by column values/binning, automatically removing all-NaN
    and constant columns, while ignoring drop_cols.
    Returns {he_id: [node_idx,...], ...}.
    """
    # Filter out class and dropped columns
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != "class"]
    if drop_cols:
        feature_cols = [c for c in feature_cols if c not in drop_cols]
    hyperedges = {}
    he_id = 0
    for col in feature_cols:
        ser = df[col]
        if ser.isna().all():
            continue
        vals = ser.dropna().unique()
        if len(vals) < 2:
            continue
        # Categorical columns
        if ser.dtype == object or ser.dtype.name == "category":
            for v in vals:
                if pd.isna(v):
                    continue
                idxs = df.index[ser == v].tolist()
                for i in range(0, len(idxs), max_nodes_per_hyperedge):
                    group = idxs[i:i+max_nodes_per_hyperedge]
                    if len(group) >= 2:
                        hyperedges[he_id] = group
                        he_id += 1
        # Numerical columns
        else:
            arr = pd.to_numeric(ser, errors="coerce").values
            mask = ~np.isnan(arr)
            if mask.sum() < 2:
                continue
            cuts = np.quantile(arr[mask], [0.0,0.25,0.5,0.75,1.0])
            bins = np.digitize(arr, bins=cuts[1:-1], right=True)
            for b in np.unique(bins):
                idxs = df.index[bins == b].tolist()
                for i in range(0, len(idxs), max_nodes_per_hyperedge):
                    group = idxs[i:i+max_nodes_per_hyperedge]
                    if len(group) >= 2:
                        hyperedges[he_id] = group
                        he_id += 1
    avg_size = np.mean([len(v) for v in hyperedges.values()]) if hyperedges else 0
    print(f"[Hyperedges] Total {len(hyperedges)}, average {avg_size:.2f} nodes/hyperedge (ignoring {drop_cols})")
    return hyperedges

# cluster_nodes_by_similarity_gpu and delete_feature_columns_hgcn remain unchanged

def cluster_nodes_by_similarity_gpu(
    indices: list,
    df: pd.DataFrame,
    max_nodes: int,
    device: torch.device
):
    """
    Greedily partition nodes within the same hyperedge by attribute similarity,
    ensuring each cluster has at most max_nodes nodes.
    Similarity is measured as the total number of identical fields between two
    samples across all fields (excluding the class column).
    """
    # Extract the subtable and remove the label column
    sub = df.loc[indices].drop(columns=["class"])
    n, d = sub.shape

    # Factorize each column and convert to long tensor
    cols = []
    for c in sub.columns:
        codes, _ = pd.factorize(sub[c])
        cols.append(torch.tensor(codes, dtype=torch.long, device=device))
    X = torch.stack(cols, dim=1)  # (n, d)

    unassigned = torch.ones(n, dtype=torch.bool, device=device)
    orig_idx = torch.tensor(indices, dtype=torch.long)  # Corresponding original df indices
    clusters = []

    while unassigned.any():
        # Find a representative
        rep = torch.nonzero(unassigned, as_tuple=False)[0].item()
        unassigned[rep] = False
        current = [orig_idx[rep].item()]

        # Remaining candidates
        rest = torch.nonzero(unassigned, as_tuple=False).squeeze(1)
        if rest.numel() > 0:
            rep_vec = X[rep].unsqueeze(0)            # (1, d)
            cand   = X[rest]                          # (r, d)
            sim    = (cand == rep_vec).sum(dim=1)     # (r,)
            _, order = torch.sort(sim, descending=True)
            take = min(max_nodes - 1, order.numel())
            chosen = rest[order[:take]]
            for i in chosen.tolist():
                current.append(orig_idx[i].item())
                unassigned[i] = False

        clusters.append(current)
    return clusters