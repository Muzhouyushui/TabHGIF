#!/usr/bin/env python
# coding: utf-8
"""
HGAT Unlearning (column-deletion) pipeline based on GIF,
aligned to your original HGAT training code style.
"""
import time
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score

from database.data_preprocessing.data_preprocessing_K import (
    preprocess_node_features,
    generate_hyperedge_dict
)
from HGNNs_Model.HGAT import HGAT_JK

from GIF.GIF_HGAT_COL import (
    approx_gif_col,
    train_model,
    delete_feature_columns_hgat
)

from config import get_args


def build_incidence_matrix(hyperedges: dict, num_nodes: int, device=None) -> torch.Tensor:
    n_edges = len(hyperedges)
    H = torch.zeros((n_edges, num_nodes), dtype=torch.float32, device=device)
    for i, nodes in enumerate(hyperedges.values()):
        H[i, nodes] = 1.0
    return H.to_sparse()


def main():
    args   = get_args()
    device = torch.device("cuda:0")

    # ─── 1. Load & preprocess ─────────────────────────────────────────
    X_train, y_train, df_train, transformer = preprocess_node_features(
        args.train_csv, is_test=False)
    X_test,  y_test,  df_test,  _           = preprocess_node_features(
        args.test_csv, is_test=True, transformer=transformer)

    train_edges = generate_hyperedge_dict(
        df_train, args.cat_cols, max_nodes_per_hyperedge=args.max_nodes_per_hyperedge, device=device
    )
    test_edges  = generate_hyperedge_dict(
        df_test,  args.cat_cols, max_nodes_per_hyperedge=args.max_nodes_per_hyperedge, device=device
    )

    H_train = build_incidence_matrix(train_edges, len(X_train), device)
    H_test  = build_incidence_matrix(test_edges,  len(X_test),  device)

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.long,    device=device)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32, device=device)
    y_test_t  = torch.tensor(y_test,  dtype=torch.long,    device=device)

    # ─── 2. Train original HGAT ───────────────────────────────────────
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

    model = train_model(
        model, criterion, optimizer, scheduler,
        X_train_t, y_train_t, H_train,
        num_epochs=args.epochs, print_freq=100
    )

    # ─── 3. Evaluate original ─────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        logits = model(X_test_t, H_test)
        preds  = logits.argmax(dim=1)
        acc    = accuracy_score(y_test_t.cpu(), preds.cpu())
        f1     = f1_score(y_test_t.cpu(), preds.cpu(), average='micro')
    print(f"Original test acc {acc:.4f}, f1 {f1:.4f}")

    # ─── 4. Prepare column-deletion info ──────────────────────────────
    if not args.del_cols:
        raise ValueError("Please specify the list of columns to unlearn via --del_cols")
    deleted_names = args.del_cols
    # We still need the list of encoded-dimension indices for GIF
    # (we'll match against transformer.get_feature_names_out())
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
            raise ValueError(f"Original column '{col}' was not found in the encoded features")
        deleted_idxs.extend(matches)
    deleted_idxs = sorted(set(deleted_idxs))
    print(f"\nDeleting columns {deleted_names} → zeroing dims {deleted_idxs}")

    unlearn_info = (deleted_names, deleted_idxs, train_edges)
    data_obj = {
        "x":      X_train_t,
        "y":      y_train_t,
        "H_orig": H_train.coalesce().to(device),
    }

    # ─── 5. Run GIF column deletion ────────────────────────────────────
    start = time.time()
    un_time, _ = approx_gif_col(
        model, data_obj, unlearn_info,
        iteration=args.if_iters, damp=args.if_damp, scale=args.if_scale
    )
    print(f"GIF column unlearning time: {un_time:.2f}s  (total {time.time()-start:.2f}s)")

    # ─── 6. Evaluate unlearned with delete_feature_columns_hgat ───────
    X_test_u, test_edges_u, H_test_u = delete_feature_columns_hgat(
        X_test_t.clone(), transformer, deleted_names, test_edges, device
    )

    model.eval()
    with torch.no_grad():
        logits_u = model(X_test_u, H_test_u)
        preds_u  = logits_u.argmax(dim=1)
        acc_u    = accuracy_score(y_test_t.cpu(), preds_u.cpu())
        f1_u     = f1_score(   y_test_t.cpu(), preds_u.cpu(), average='micro')
    print(f"Unlearned test acc {acc_u:.4f}, f1 {f1_u:.4f}")


if __name__ == "__main__":
    main()
    for run in range(1,3):
        print("=== Run",run,"===")
        main()
