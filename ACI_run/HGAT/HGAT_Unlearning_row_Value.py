#!/usr/bin/env python
# coding: utf-8
"""
HGAT Unlearning (row-deletion) pipeline based on GIF,
aligned to your original HGAT training code style (no degree vectors).
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score
# Data tools
from database.data_preprocessing.data_preprocessing_K import (
    preprocess_node_features,
    generate_hyperedge_dict
)
# HGAT model
from HGNNs_Model.HGAT import HGAT_JK
# Common utils
# GIF unlearning (in-place)
from GIF.GIF_HGAT_ROW_NEI import approx_gif, train_model
# MIA
from MIA.MIA_HGAT import membership_inference_hgat

# Local incidence builder (Torch sparse, from your training code)
def build_incidence_matrix(hyperedges: dict, num_nodes: int, device=None) -> torch.Tensor:
    n_edges = len(hyperedges)
    H = torch.zeros((n_edges, num_nodes), dtype=torch.float32, device=device)
    for i, nodes in enumerate(hyperedges.values()):
        H[i, nodes] = 1.0
    return H.to_sparse()

# Config parser
from config import get_args


def main():
    args = get_args()
    device = torch.device("cuda:0")
    t2=time.time()
    # Preprocess data
    X_train, y_train, df_train, transformer = preprocess_node_features(
        args.train_csv, is_test=False)
    X_test,  y_test,  df_test,  _           = preprocess_node_features(
        args.test_csv, is_test=True, transformer=transformer)
    t3=time.time()
    # Preprocess data
    X_train, y_train, df_train, transformer = preprocess_node_features(
        args.train_csv, is_test=False)
    X_test, y_test, df_test, _ = preprocess_node_features(
        args.test_csv, is_test=True, transformer=transformer)


    t4=time.time()
    # Generate hyperedges and incidence matrices
    train_edges = generate_hyperedge_dict(
        df_train, args.cat_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        device=device)
    t5=time.time()
    test_edges  = generate_hyperedge_dict(
        df_test,  args.cat_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        device=device)
    H_train = build_incidence_matrix(train_edges, len(X_train), device)
    H_test  = build_incidence_matrix(test_edges,  len(X_test),  device)

    # Tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.long,    device=device)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32, device=device)
    y_test_t  = torch.tensor(y_test,  dtype=torch.long,    device=device)


    t6=time.time()
    # Initialize HGAT model, optimizer, scheduler
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
    t8=time.time()
    t=t3-t2+t5-t4+t8-t6
    print(f"Original test acc {acc:.4f}, f1 {f1:.4f}")
    print("retrain time=",t)

    mask = (df_train['sex'] == 'Male') & (df_train['relationship'] == 'Husband')
    deleted_idx_np = np.where(mask.values)[0]
    deleted_idx    = torch.tensor(deleted_idx_np,
                          dtype=torch.long,
                          device=device)
    num_del = len(deleted_idx_np)
    num_train = X_train_t.size(0)
    print(f"\n按值删除，共 {num_del}/{num_train} 个节点 ({100. * num_del/num_train:.1f}%)，前 5 个索引：{deleted_idx_np[:5]}")



    print(f"\nRandomly deleting {num_del}/{num_train} nodes ({100.0 * num_del / num_train:.1f}%) for unlearning.")

    K = getattr(args, 'neighbor_k', 12)
    unlearn_info = (
        deleted_idx_np,  # numpy array 用于 find_hyperneighbors
        train_edges,
        K
    )

    data_obj = {
        "x": X_train_t,  # [N, F], FloatTensor
        "y": y_train_t,  # [N], LongTensor
        "H_orig": H_train.coalesce().to(device),  # [E, N] torch.sparse_coo_tensor

    }
    t1=time.time()
    # 然后你就可以
    un_time, _ = approx_gif(
        model,
        data_obj,
        unlearn_info,
        iteration=args.if_iters,
        damp=args.if_damp,
        scale=1e7
    )
    print(f"GIF unlearning time: {un_time:.2f}s")
    t2=time.time()
    print(f"GIF training time: {t2-t1:.2f}s")
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

    print(f"[Shadow MIA]   AUC={auc_s:.4f}, F1={f1_s:.4f}")
    print(f"[Deletion MIA] AUC={auc_t:.4f}, F1={f1_t:.4f}")


if __name__ == "__main__":
    main()
    for run in range(1,3):
        print("=== Run",run,"===")
        main()
