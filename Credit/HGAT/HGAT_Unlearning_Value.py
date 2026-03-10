#!/usr/bin/env python
# coding: utf-8
"""
HGAT Unlearning (row-deletion) pipeline based on GIF,
aligned to your original HGAT training code style (no degree vectors).
"""
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score
# Data tools
from Credit.HGAT.data_preprocessing_credit import (
    preprocess_node_features,
    generate_hyperedge_dict
)
# HGAT model
from Credit.HGAT.HGAT_new import HGAT_JK
# Common utils
from utils.common_utils import evaluate_test_f1, evaluate_test_acc
# GIF unlearning (in-place)
from Credit.HGAT.GIF_HGAT_ROW_NEI import approx_gif, rebuild_structure_after_node_deletion, train_model
# MIA
from Credit.HGAT.MIA_HGAT import train_shadow_model, membership_inference_hgat
import pandas as pd
from sklearn.model_selection import train_test_split

# Local incidence builder (Torch sparse, from your training code)
def build_incidence_matrix(hyperedges: dict, num_nodes: int, device=None) -> torch.Tensor:
    n_edges = len(hyperedges)
    H = torch.zeros((n_edges, num_nodes), dtype=torch.float32, device=device)
    for i, nodes in enumerate(hyperedges.values()):
        H[i, nodes] = 1.0
    return H.to_sparse()

# Config parser
from Credit.HGAT.config import get_args


def main():
    args   = get_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # 1) Define Credit Approval column names (16 attributes + 1 label)
    col_names = [
        "A1",  "A2",  "A3",  "A4",  "A5",
        "A6",  "A7",  "A8",  "A9",  "A10",
        "A11", "A12", "A13", "A14", "A15",
        "A16",  # the last one is the label
    ]

    # 2) Read headerless CSV
    df_full = pd.read_csv(
        args.data_csv,
        header=None,
        names=col_names,
        na_values="?",
        skipinitialspace=True
    )
    print("Available columns:", list(df_full.columns))

    # 3) Determine and map the label column (usually A16 in the Credit dataset)
    label_col = getattr(args, "label_col", "A16")
    if label_col not in df_full.columns:
        raise ValueError(f"Label column '{label_col}' not found in DataFrame")

    # 4) Split train / test
    df_train, df_test = train_test_split(
        df_full,
        test_size=args.split_ratio,
        stratify=df_full[label_col],
        random_state=42
    )
    df_train = df_train.reset_index(drop=True)
    df_test  = df_test.reset_index(drop=True)
    print(f"Train: {len(df_train)} rows, Test: {len(df_test)} rows")

    # 5) Preprocessing: DataFrame interface, automatically handles missing values / OneHot / standardization
    X_train, y_train, df_train_proc, transformer = preprocess_node_features(
        df_train,
    )
    X_test,  y_test,  df_test_proc, _ = preprocess_node_features(
        df_test, transformer=transformer
    )

    # 5) Build hyperedges & sparse H
    train_edges = generate_hyperedge_dict(
        df_train_proc,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        device=device
    )
    test_edges = generate_hyperedge_dict(
        df_test_proc,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        device=device
    )
    H_train = build_incidence_matrix(train_edges, X_train.shape[0], device)
    H_test  = build_incidence_matrix(test_edges,  X_test.shape[0],  device)

    # 6) Convert to PyTorch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.long,    device=device)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32, device=device)
    y_test_t  = torch.tensor(y_test,  dtype=torch.long,    device=device)

    # Initialize HGAT model, optimizer, scheduler
    num_classes = int(y_train_t.max().item() + 1)
    model = HGAT_JK(
        in_dim=X_train_t.size(1),
        hidden_dim=args.hidden_dim,
        out_dim=num_classes,
        dropout=args.dropout,
        alpha=0.5,
        num_layers=2,
        use_jk=False  # whether to concatenate outputs from all layers
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[args.epochs//2, args.epochs//4*3],
        gamma=0.1
    )

    # Define loss, optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[args.epochs // 2, args.epochs // 4 * 3],
        gamma=0.1
    )

    # Call the HGAT-specific train_model for training and return the best-weight model
    model = train_model(
        model,
        criterion,
        optimizer,
        scheduler,
        X_train_t,  # node feature tensor
        y_train_t,  # label tensor
        H_train,    # sparse hypergraph structure tensor
        num_epochs=200,
        print_freq=100
    )
    # —— Evaluate the original model —— #
    model.eval()
    with torch.no_grad():
        # 1) Forward inference
        logits = model(X_test_t, H_test)  # [N_test, C]
        preds = logits.argmax(dim=1)      # [N_test]

        # 2) Compute metrics
        acc = accuracy_score(y_test_t.cpu(), preds.cpu())
        f1 = f1_score(y_test_t.cpu(), preds.cpu(), average='micro')

    print(f"Original test acc {acc:.4f}, f1 {f1:.4f}")
    # --- Start Unlearning Process (GIF) ---
    # This part of the code should be executed after the model training above is completed.
    # Assume that X_train_t, y_train_t, H_train, train_edges, device, and args are already loaded.

    # 1) Delete by value: find all nodes to unlearn from the original df_train
    delete_col = "A1"     # e.g. "A1"
    delete_val = "b"      # e.g. "b"
    matched_idx = df_train[df_train[delete_col] == delete_val].index.tolist()
    if len(matched_idx) == 0:
        raise ValueError(f"[Unlearn] No samples with {delete_col} == {delete_val} found in the training set")
    deleted_idx = torch.tensor(matched_idx, dtype=torch.long, device=device)
    deleted_idx_np = deleted_idx.cpu().numpy()
    print(f"[Unlearn by value] Delete samples where column {delete_col} has value {delete_val}, total {len(matched_idx)} samples")

    # 3) Construct unlearn_info
    K = getattr(args, 'neighbor_k', 12)
    unlearn_info = (
        deleted_idx_np,  # numpy array used for find_hyperneighbors
        train_edges,     # {he_id: [node_id,...]} hyperedge dictionary
        K                # neighbor threshold
    )

    # 4) Construct data_obj for GIF
    data_obj = {
        "x": X_train_t,  # [N, F], FloatTensor
        "y": y_train_t,  # [N], LongTensor
        "H_orig": H_train.coalesce().to(device),  # [E, N] torch.sparse_coo_tensor
        # Optional: if you have train_mask:
        # "train_mask": train_mask_t            # [N], BoolTensor
    }

    # Then you can call GIF
    un_time, _ = approx_gif(
        model,
        data_obj,
        unlearn_info,
        iteration=30,
        damp=0.01,
        scale=1e7
    )
    print(f"GIF unlearning time: {un_time:.2f}s")

    # Evaluate unlearned model
    model.eval()
    with torch.no_grad():
        # Use the original test hypergraph here, instead of the training-set hypergraph after deletion
        logits_un = model(X_test_t, H_test)
        preds_un = logits_un.argmax(dim=1)
        acc_u = accuracy_score(y_test_t.cpu(), preds_un.cpu())
        f1_u = f1_score(y_test_t.cpu(), preds_un.cpu(), average='micro')
    print(f"Unlearned test acc {acc_u:.4f} f1 {f1_u:.4f}")

    X_train_np = X_train_t.cpu().numpy()
    y_train_np = y_train_t.cpu().numpy()

    # 2) Construct member_mask: deleted nodes are marked as 1 (positive), others as 0 (negative)
    num_samples = X_train_np.shape[0]
    member_mask = np.zeros(num_samples, dtype=bool)
    member_mask[deleted_idx_np] = True

    # 3) Call the HGAT-specific MIA function
    attack_model, (auc_s, f1_s), (auc_t, f1_t) = membership_inference_hgat(
        X_train_np,
        y_train_np,
        train_edges,
        target_model=model,   # pass the already unlearned model here
        args=args,
        device=device,
        member_mask=member_mask
    )

    # 4) Print results
    print(f"[Shadow MIA]   AUC={auc_s:.4f}, F1={f1_s:.4f}")
    print(f"[Deletion MIA] AUC={auc_t:.4f}, F1={f1_t:.4f}")


if __name__ == "__main__":
    main()
    for run in range(1, 5):
        print("=== Run", run, "===")
        main()