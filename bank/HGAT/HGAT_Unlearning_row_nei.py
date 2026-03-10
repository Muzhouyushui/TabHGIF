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
from bank.HGAT.data_preprocessing_bank import (
    preprocess_node_features_bank,
    generate_hyperedge_dict_bank
)
# HGAT model
from bank.HGAT.HGAT_new import HGAT_JK
# Common utils
from utils.common_utils import evaluate_test_f1, evaluate_test_acc
# GIF unlearning (in-place)
from bank.HGAT.GIF_HGAT_ROW_NEI import approx_gif, rebuild_structure_after_node_deletion,train_model
# MIA
from bank.HGAT.MIA_HGAT import train_shadow_model, membership_inference_hgat
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
from bank.HGAT.config import get_args


def main():
    args   = get_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    df_full = pd.read_csv(args.data_csv, sep=';', skipinitialspace=True)
    print("Available columns:", list(df_full.columns))

    label_col = getattr(args, "label_col", "y")
    if label_col not in df_full.columns:
        raise ValueError(f"Label column '{label_col}' not found")

    df_train, df_test = train_test_split(
        df_full,
        test_size=args.split_ratio,
        stratify=df_full[label_col],
        random_state=21
    )
    df_train = df_train.reset_index(drop=True)
    df_test  = df_test.reset_index(drop=True)
    print(f"Train: {len(df_train)} samples, Test: {len(df_test)} samples")

    X_train, y_train, df_train_proc, transformer = preprocess_node_features_bank(
        df_train, is_test=False
    )
    X_test,  y_test,  df_test_proc, _ = preprocess_node_features_bank(
        df_test,  is_test=True, transformer=transformer
    )

    train_edges = generate_hyperedge_dict_bank(
        df_train_proc, args.cat_cols, args.cont_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        device=device
    )
    test_edges = generate_hyperedge_dict_bank(
        df_test_proc, args.cat_cols, args.cont_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        device=device
    )
    H_train = build_incidence_matrix(train_edges, X_train.shape[0], device)
    H_test  = build_incidence_matrix(test_edges,  X_test.shape[0],  device)

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.long,    device=device)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32, device=device)
    y_test_t  = torch.tensor(y_test,  dtype=torch.long,    device=device)



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
        milestones=[args.epochs//2, args.epochs//4*3],
        gamma=0.1
    )

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

    model = train_model(
        model,
        criterion,
        optimizer,
        scheduler,
        X_train_t,
        y_train_t,
        H_train,
        num_epochs=130,
        print_freq=100
    )
    model.eval()
    with torch.no_grad():
        logits = model(X_test_t, H_test)  # [N_test, C]
        preds = logits.argmax(dim=1)  # [N_test]

        acc = accuracy_score(y_test_t.cpu(), preds.cpu())
        f1 = f1_score(y_test_t.cpu(), preds.cpu(), average='micro')

    print(f"Original test acc {acc:.4f}, f1 {f1:.4f}")

    num_train = X_train_t.size(0)
    remove_ratio = getattr(args, 'remove_ratio', 0.30)
    num_del = int(remove_ratio * num_train)


    perm = torch.randperm(num_train, device=device)
    deleted_idx = perm[:num_del]  # LongTensor [num_del]
    deleted_idx_np = deleted_idx.cpu().numpy()  # numpy array


    print(f"\nRandomly deleting {num_del}/{num_train} nodes ({100.0 * num_del / num_train:.1f}%) for unlearning.")

    # 3)  unlearn_info
    K = getattr(args, 'neighbor_k', 12)
    unlearn_info = (
        deleted_idx_np,  #
        train_edges,  # {he_id: [node_id,...]}
        K  #
    )

    data_obj = {
        "x": X_train_t,  # [N, F], FloatTensor
        "y": y_train_t,  # [N], LongTensor
        "H_orig": H_train.coalesce().to(device),  # [E, N] torch.sparse_coo_tensor

    }

    un_time, _ = approx_gif(
        model,
        data_obj,
        unlearn_info,
        iteration=30,
        damp=0.01,
        scale=1e7
    )
    print(f"GIF unlearning time: {un_time:.2f}s")

    # Evaluate unlearned
    model.eval()
    with torch.no_grad():
        logits_un = model(X_test_t, H_test)
        preds_un = logits_un.argmax(dim=1)
        acc_u = accuracy_score(y_test_t.cpu(), preds_un.cpu())
        f1_u = f1_score(y_test_t.cpu(), preds_un.cpu(), average='micro')
    print(f"Unlearned test acc {acc_u:.4f} f1 {f1_u:.4f}")


    X_train_np = X_train_t.cpu().numpy()
    y_train_np = y_train_t.cpu().numpy()

    num_samples = X_train_np.shape[0]
    member_mask = np.zeros(num_samples, dtype=bool)
    member_mask[deleted_idx_np] = True

    attack_model, (auc_s, f1_s), (auc_t, f1_t) = membership_inference_hgat(
        X_train_np,
        y_train_np,
        train_edges,
        target_model=model,
        args=args,
        device=device,
        member_mask=member_mask
    )

    # 4) 打印结果
    print(f"[Shadow MIA]   AUC={auc_s:.4f}, F1={f1_s:.4f}")
    print(f"[Deletion MIA] AUC={auc_t:.4f}, F1={f1_t:.4f}")


if __name__ == "__main__":
    main()
    for run in range(1,5):
        print("=== Run",run,"===")
        main()