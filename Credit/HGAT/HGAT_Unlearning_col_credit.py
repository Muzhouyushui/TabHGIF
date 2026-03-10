
"""
HGAT Unlearning (column-deletion) pipeline based on GIF,
aligned to Credit Approval dataset.
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score
# Config
from Credit.HGAT.config import get_args
# Data tools
from Credit.HGAT.data_preprocessing_credit import (
    preprocess_node_features,
    generate_hyperedge_dict
)
# HGAT model
from Credit.HGAT.HGAT_new import HGAT_JK
# GIF unlearning (in-place)
from Credit.HGAT.GIF_HGAT_COL import approx_gif_col, delete_feature_columns_hgat, train_model
import pandas as pd
from sklearn.model_selection import train_test_split


def build_incidence_matrix(hyperedges: dict, num_nodes: int, device=None) -> torch.Tensor:
    n_edges = len(hyperedges)
    H = torch.zeros((n_edges, num_nodes), dtype=torch.float32, device=device)
    for i, nodes in enumerate(hyperedges.values()):
        H[i, nodes] = 1.0
    return H.to_sparse()


def main():
    args   = get_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # ─── 1. Load Credit data ─────────────────────────────────────────
    col_names = [f"A{i}" for i in range(1, 16)] + ["A16"]
    df_full = pd.read_csv(
        args.data_csv,
        header=None,
        names=col_names,
        na_values="?",
        skipinitialspace=True
    )
    label_col = "A16"

    # Split train/test
    df_train, df_test = train_test_split(
        df_full,
        test_size=args.split_ratio,
        stratify=df_full[label_col],
        random_state=42
    )
    df_train = df_train.reset_index(drop=True)
    df_test  = df_test.reset_index(drop=True)
    print(f"Train: {len(df_train)} rows, Test: {len(df_test)} rows")

    # ─── 2. Preprocess ───────────────────────────────────────────────
    # preprocess_node_features internally renames cols to A1-A15 + class
    X_train, y_train, df_train_proc, transformer = preprocess_node_features(df_train)
    X_test,  y_test,  df_test_proc,  _           = preprocess_node_features(df_test, transformer=transformer)
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

    # ─── 3. Build hyperedges ─────────────────────────────────────────
    feature_cols = [f"A{i}" for i in range(1, 16)]
    train_edges = generate_hyperedge_dict(
        df_train_proc,
        feature_cols=feature_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        device=device
    )
    test_edges = generate_hyperedge_dict(
        df_test_proc,
        feature_cols=feature_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        device=device
    )
    H_train = build_incidence_matrix(train_edges, len(X_train), device)
    H_test  = build_incidence_matrix(test_edges,  len(X_test),  device)

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.long,    device=device)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32, device=device)
    y_test_t  = torch.tensor(y_test,  dtype=torch.long,    device=device)

    # ─── 4. Train original HGAT ──────────────────────────────────────
    num_classes = int(y_train_t.max().item() + 1)
    model = HGAT_JK(
        in_dim=X_train_t.size(1),
        hidden_dim=args.hidden_dim,
        out_dim=num_classes,
        dropout=args.dropout,
        alpha=0.5,
        num_layers=2,
        use_jk=False
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[args.epochs // 2, args.epochs // 4 * 3],
        gamma=0.1
    )

    model = train_model(
        model, criterion, optimizer, scheduler,
        X_train_t, y_train_t, H_train,
        num_epochs=args.epochs, print_freq=100
    )

    # ─── 5. Evaluate original ────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        logits = model(X_test_t, H_test)
        preds  = logits.argmax(dim=1)
        acc    = accuracy_score(y_test_t.cpu(), preds.cpu())
        f1     = f1_score(y_test_t.cpu(), preds.cpu(), average='micro')
    print(f"Before Unlearning — Acc: {acc:.4f}, F1: {f1:.4f}")

    # ─── 6. Column deletion via GIF ──────────────────────────────────
    deleted_names = list(args.columns_to_unlearn)
    try:
        feat_names = transformer.get_feature_names_out().tolist()
    except AttributeError:
        feat_names = transformer.get_feature_names()
    deleted_idxs = []
    for col in deleted_names:
        matches = [
            i for i, fn in enumerate(feat_names)
            if col in fn.split('__') or col in fn.split('_')
        ]
        if not matches:
            raise ValueError(f"在编码后特征中未找到原始列 '{col}'")
        deleted_idxs.extend(matches)
    deleted_idxs = sorted(set(deleted_idxs))
    print(f"\nDeleting columns {deleted_names} → zeroing dims {deleted_idxs}")

    unlearn_info = (deleted_names, deleted_idxs, train_edges)
    data_obj = {
        "x":      X_train_t,
        "y":      y_train_t,
        "H_orig": H_train.coalesce().to(device),
    }

    start = time.time()
    un_time, _ = approx_gif_col(
        model, data_obj, unlearn_info,
        iteration=args.gif_iters, damp=args.gif_damp, scale=args.gif_scale
    )
    print(f"GIF column unlearning time: {un_time:.2f}s  (total {time.time()-start:.2f}s)")

    # ─── 7. Evaluate unlearned ───────────────────────────────────────
    X_test_u, test_edges_u, H_test_u = delete_feature_columns_hgat(
        X_test_t.clone(), transformer, deleted_names, test_edges, device
    )

    model.eval()
    with torch.no_grad():
        logits_u = model(X_test_u, H_test_u)
        preds_u  = logits_u.argmax(dim=1)
        acc_u    = accuracy_score(y_test_t.cpu(), preds_u.cpu())
        f1_u     = f1_score(y_test_t.cpu(), preds_u.cpu(), average='micro')
    print(f"After Unlearning  — Acc: {acc_u:.4f}, F1: {f1_u:.4f}")


if __name__ == "__main__":
    for run in range(1, 4):
        print(f"=== Run {run} ===")
        main()
