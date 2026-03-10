# data_preprocessing.py

import time
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from typing import Union, Tuple

def preprocess_node_features(
    data: Union[str, pd.DataFrame],
    transformer: ColumnTransformer = None
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, ColumnTransformer]:
    """
    Read and preprocess the UCI Credit Approval dataset.
    Args:
      data: File path (.data/.csv, no header, missing values represented by '?')
            or an already loaded pd.DataFrame.
      transformer: If provided, directly call transformer.transform;
                   otherwise fit a new one.
    Returns:
      X: (n_samples, n_feats) numeric feature matrix (numpy array).
      y: (n_samples,) binary labels, '+' -> 1, '-' -> 0.
      df: (n_samples, original number of columns) DataFrame used for hyperedge construction
          (index has been reset).
      transformer: ColumnTransformer for numeric/categorical columns, convenient for test-set reuse.
    """
    # 1. Read the DataFrame and assign column names
    if isinstance(data, str):
        # The UCI crx.data file has no header, and missing values are represented by '?'
        df = pd.read_csv(data, header=None, na_values='?')
    else:
        df = data.copy()
    df = df.reset_index(drop=True)
    # Original data has 16 columns in total: A1~A15 inputs + class output
    df.columns = [
        "A1","A2","A3","A4","A5","A6","A7","A8","A9",
        "A10","A11","A12","A13","A14","A15","class"
    ]

    # 2. Split features / labels
    feature_cols = [f"A{i}" for i in range(1,16)]
    num_cols = ["A2","A3","A8","A11","A14","A15"]   # 6 continuous features
    cat_cols = [c for c in feature_cols if c not in num_cols]  # 9 categorical features

    # 3. Label mapping
    y = df["class"].map({"+": 1, "-": 0}).astype(int).values

    # 4. Feature preprocessing: missing values, scaling, one-hot encoding
    X_df = df[feature_cols]
    if transformer is None:
        # Numeric pipeline: median imputation + standardization
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        # Categorical pipeline: constant imputation + one-hot encoding
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False  # sklearn>=1.3
            ))
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


def cluster_nodes_by_similarity_gpu(
    indices: list,
    df: pd.DataFrame,
    max_nodes: int,
    device: torch.device
):
    """
    Greedily partition nodes within the same hyperedge according to attribute similarity,
    ensuring each cluster has at most max_nodes nodes.
    Similarity metric: total number of identical fields between two samples
    (excluding the class column).
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
    orig_idx = torch.tensor(indices, dtype=torch.long)  # corresponding original df indices
    clusters = []

    while unassigned.any():
        # Find one representative
        rep = torch.nonzero(unassigned, as_tuple=False)[0].item()
        unassigned[rep] = False
        current = [orig_idx[rep].item()]

        # Remaining candidates
        rest = torch.nonzero(unassigned, as_tuple=False).squeeze(1)
        if rest.numel() > 0:
            rep_vec = X[rep].unsqueeze(0)            # (1, d)
            cand   = X[rest]                         # (r, d)
            sim    = (cand == rep_vec).sum(dim=1)    # (r,)
            _, order = torch.sort(sim, descending=True)
            take = min(max_nodes - 1, order.numel())
            chosen = rest[order[:take]]
            for i in chosen.tolist():
                current.append(orig_idx[i].item())
                unassigned[i] = False

        clusters.append(current)
    return clusters


def generate_hyperedge_dict(
    df: pd.DataFrame,
    feature_cols: list = None,
    max_nodes_per_hyperedge: int = 50,
    device: torch.device = torch.device("cpu")
) -> dict:
    """
    Same as the previous version: generate hyperedges by column values / binning,
    automatically removing all-NaN and constant columns.
    Returns {he_id: [node_idx, ...], ...}.
    """
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != "class"]
    hyperedges = {}
    he_id = 0

    for col in feature_cols:
        ser = df[col]
        # Remove all-NaN columns
        if ser.isna().all():
            continue
        # Remove constant columns
        vals = ser.dropna().unique()
        if len(vals) < 2:
            continue

        # Categorical columns
        if ser.dtype == object or ser.dtype.name == "category":
            for v in vals:
                if pd.isna(v):
                    continue
                idxs = df.index[ser == v].tolist()
                if len(idxs) < 2:
                    continue
                for i in range(0, len(idxs), max_nodes_per_hyperedge):
                    hyperedges[he_id] = idxs[i:i+max_nodes_per_hyperedge]
                    he_id += 1

        # Numeric columns
        else:
            arr = pd.to_numeric(ser, errors="coerce").values
            mask = ~np.isnan(arr)
            if mask.sum() < 2:
                continue
            arr_valid = arr[mask]
            # Quartile binning
            cuts = np.quantile(arr_valid, [0.0, 0.25, 0.5, 0.75, 1.0])
            bins = np.digitize(arr, bins=cuts[1:-1], right=True)
            for b in np.unique(bins):
                if b < 0:
                    continue
                idxs = df.index[bins == b].tolist()
                if len(idxs) < 2:
                    continue
                for i in range(0, len(idxs), max_nodes_per_hyperedge):
                    hyperedges[he_id] = idxs[i:i+max_nodes_per_hyperedge]
                    he_id += 1

    avg_size = np.mean([len(v) for v in hyperedges.values()]) if hyperedges else 0
    print(f"[Hyperedges] Total {len(hyperedges)}, average {avg_size:.2f} nodes/hyperedge")
    return hyperedges