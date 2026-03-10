#!/usr/bin/env python
# coding: utf-8
"""
train_unlearning_hgnn_refactored.py

Refactored version: separates retraining on a pruned dataset and GIF-based unlearning
into two clear functions, using the same deleted node indices for both parts.
"""
from collections import Counter

from torch_geometric.data import Data

import torch.nn as nn
import torch.optim as optim
import time
import torch.nn.functional as F

from utils.common_utils import evaluate_test_acc, evaluate_test_f1
from database.data_preprocessing.data_preprocessing_K import (
    preprocess_node_features,
    generate_hyperedge_dict,
)
from HGNNs_Model.HGNN.HGNN_2 import HGNN_implicit, build_incidence_matrix, compute_degree_vectors
from GIF.GIF_HGNN_ROW import (
    rebuild_structure_after_node_deletion,
    approx_gif,
    train_model,
)
from config import get_args

import numpy as np
import torch
from MIA.MIA_utils import train_shadow_model, membership_inference
from paths import ACI_TEST, ACI_TRAIN

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None, reduction: str = 'mean'):

        super().__init__()
        self.gamma    = gamma
        self.weight   = weight
        self.reduction= reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

        ce_loss = F.cross_entropy(logits, targets,
                                  weight=self.weight,
                                  reduction='none')

        pt = torch.exp(-ce_loss)

        loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
def mia_with_hgnn(X_train, y_train, hyperedges, args, device):

    N = X_train.shape[0]

    H_sp        = build_incidence_matrix(hyperedges, N)
    dv_inv, de_inv = compute_degree_vectors(H_sp)
    Hc = H_sp.tocoo()
    idx = torch.LongTensor([Hc.row, Hc.col])
    val = torch.FloatTensor(Hc.data)
    H_tensor = torch.sparse_coo_tensor(idx, val, Hc.shape).to(device)

    full_train_mask = torch.ones(N, dtype=torch.bool, device=device)
    data_full = Data(
        x = torch.from_numpy(X_train).float().to(device),
        y = torch.from_numpy(y_train).long().to(device),
        H         = H_tensor,
        dv_inv    = torch.from_numpy(dv_inv).to(device),
        de_inv    = torch.from_numpy(de_inv).to(device),
        train_mask    = full_train_mask,
        train_indices = list(range(N)),
        test_indices  = []
    )

    full_model = HGNN_implicit(
        in_ch   = X_train.shape[1],
        n_class = int(y_train.max()) + 1,
        n_hid   = args.full_hidden,
        dropout = args.full_dropout
    ).to(device)

    full_model = train_shadow_model(
        full_model, data_full,
        lr     = args.full_lr,
        epochs = args.full_epochs
    )


    _, (_, _), (auc_target, f1_target) = membership_inference(
        X_train = X_train,
        y_train = y_train,
        hyperedges   = hyperedges,
        target_model = full_model,
        args         = args,
        device       = device
    )
    return auc_target, f1_target

def retrain_on_pruned(X, y, df, transformer, deleted_idx, args, device):
    num_train = X.shape[0]
    keep_mask = np.ones(num_train, dtype=bool)
    keep_mask[deleted_idx] = False

    X_pruned = X[keep_mask]
    y_pruned = y[keep_mask]
    df_pruned = df.drop(index=deleted_idx).reset_index(drop=True)
    print(f"[Retraining] Number of training nodes after pruning: {X_pruned.shape[0]} / original {num_train}")

    t_start = time.time()

    cat_cols = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ]
    t0 = time.time()
    hyperedges_pruned = generate_hyperedge_dict(
        df_pruned,
        cat_cols,
        max_nodes_per_hyperedge=getattr(args, 'max_nodes_per_hyperedge_train', 10),
        device=device,
    )
    t_hg = time.time() - t0
    print(f"[Retraining] Hypergraph reconstruction time: {t_hg:.2f}s")
    print(f"[Retraining] Total number of hyperedges after pruning: {len(hyperedges_pruned)}")

    H_sp = build_incidence_matrix(hyperedges_pruned, X_pruned.shape[0])
    dv_np, de_np = compute_degree_vectors(H_sp)

    H_coo = H_sp.tocoo()
    idx = torch.LongTensor(np.vstack((H_coo.row, H_coo.col))).to(device)
    val = torch.FloatTensor(H_coo.data).to(device)
    H_pruned = torch.sparse_coo_tensor(idx, val, size=H_coo.shape).coalesce()
    dv = torch.FloatTensor(dv_np).to(device)
    de = torch.FloatTensor(de_np).to(device)
    fts = torch.FloatTensor(X_pruned).to(device)
    lbls = torch.LongTensor(y_pruned).to(device)

    t_init_start = time.time()
    model = HGNN_implicit(
        in_ch=fts.shape[1],
        n_class=len(np.unique(y)),
        n_hid=getattr(args, 'hidden_dim', 128),
        dropout=getattr(args, 'dropout', 0.1)
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=getattr(args, 'lr', 0.01), weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=getattr(args, 'milestones', [100,150]),
        gamma=getattr(args, 'gamma', 0.1)
    )
    criterion = FocalLoss(gamma=2.0, reduction='mean')
    t_init_end = time.time()
    t_train_start = time.time()
    model = train_model(
        model, criterion, optimizer, scheduler,
        fts, lbls, H_pruned, dv, de,
        num_epochs=getattr(args, 'epochs', 200), print_freq=10
    )
    t_train_end = time.time()
    t_total = time.time() - t_start

    print(
        f"[Retraining] Initialization: {t_init_end - t_init_start:.2f}s | "
        f"Training: {t_train_end - t_train_start:.2f}s | "
        f"Total: {t_total:.2f}s"
    )

    test_csv = ACI_TEST

    X_test, y_test, df_test, _ = preprocess_node_features(
        test_csv, is_test=True, transformer=transformer
    )
    t2 = time.time()
    hyperedges_test = generate_hyperedge_dict(
        df_test, cat_cols,
        max_nodes_per_hyperedge=getattr(args, 'max_nodes_per_hyperedge', 50),
        device=device
    )
    t_hg_te = time.time() - t2
    print(f"[Retraining] Test-set hypergraph reconstruction time: {t_hg_te:.2f}s")
    print(f"[Retraining] Total number of hyperedges in test set: {len(hyperedges_test)}")

    H_sp_te = build_incidence_matrix(hyperedges_test, X_test.shape[0])
    dv_te, de_te = compute_degree_vectors(H_sp_te)
    H_coo_te = H_sp_te.tocoo()
    idx_te = torch.LongTensor(np.vstack((H_coo_te.row, H_coo_te.col))).to(device)
    val_te = torch.FloatTensor(H_coo_te.data).to(device)
    H_test = torch.sparse_coo_tensor(idx_te, val_te, size=H_coo_te.shape).coalesce()
    dv_test = torch.FloatTensor(dv_te).to(device)
    de_test = torch.FloatTensor(de_te).to(device)
    fts_test = torch.FloatTensor(X_test).to(device)
    lbls_test = torch.LongTensor(y_test).to(device)

    test_obj = {"x": fts_test, "y": lbls_test, "H": H_test, "dv_inv": dv_test, "de_inv": de_test}
    f1 = evaluate_test_f1(model, test_obj)
    acc = evaluate_test_acc(model, test_obj)
    print(f"[Retraining] Test set F1: {f1:.4f}, Acc: {acc:.4f}")

    X_del = X[deleted_idx]
    y_del = y[deleted_idx]
    df_del = df.iloc[deleted_idx].reset_index(drop=True)

    hyperedges_del = generate_hyperedge_dict(
        df_del, cat_cols,
        max_nodes_per_hyperedge=getattr(args, 'max_nodes_per_hyperedge_train', 10),
        device=device
    )
    H_sp_del = build_incidence_matrix(hyperedges_del, len(deleted_idx))
    dv_np_del, de_np_del = compute_degree_vectors(H_sp_del)
    H_coo_del = H_sp_del.tocoo()
    idx_del = torch.LongTensor(np.vstack((H_coo_del.row, H_coo_del.col))).to(device)
    val_del = torch.FloatTensor(H_coo_del.data).to(device)
    H_del = torch.sparse_coo_tensor(idx_del, val_del, size=H_coo_del.shape).coalesce()

    fts_del = torch.FloatTensor(X_del).to(device)
    lbls_del = torch.LongTensor(y_del).to(device)
    dv_del = torch.FloatTensor(dv_np_del).to(device)
    de_del = torch.FloatTensor(de_np_del).to(device)
    test_obj_del = {
        "x": fts_del,
        "y": lbls_del,
        "H": H_del,
        "dv_inv": dv_del,
        "de_inv": de_del
    }
    num_hyperedges = len(hyperedges_del)

    has_nonzero_features = (fts_del.abs().sum().item() > 0)

    print(f"Number of hyperedges in deleted set: {num_hyperedges}, "
          f"Deleted-set node features not zeroed out? {has_nonzero_features}")

    model.eval()
    with torch.no_grad():
        f1_del = evaluate_test_f1(model, test_obj_del)
        acc_del = evaluate_test_acc(model, test_obj_del)

    print(f"[Deleted-Node Evaluation] Retrained model scores on deleted nodes: F1={f1_del:.4f}, Acc={acc_del:.4f}")

    all_idx = np.arange(num_train)
    keep_idx = np.setdiff1d(all_idx, deleted_idx)
    X_keep, y_keep = X[keep_idx], y[keep_idx]
    X_del, y_del = X[deleted_idx], y[deleted_idx]

    he_keep = hyperedges_pruned
    df_del = df.iloc[deleted_idx].reset_index(drop=True)
    he_del = generate_hyperedge_dict(
        df_del, cat_cols,
        max_nodes_per_hyperedge=getattr(args, 'max_nodes_per_hyperedge_train', 10),
        device=device
    )

    he_attack = {}
    cnt = 0
    for nodes in he_keep.values():
        he_attack[cnt] = nodes
        cnt += 1
    offset = len(X_keep)
    for nodes in he_del.values():
        he_attack[cnt] = [n + offset for n in nodes]
        cnt += 1

    X_attack = np.vstack([X_keep, X_del])
    y_attack = np.hstack([y_keep, y_del])
    member_mask = np.concatenate([
        np.ones(len(X_keep), dtype=bool),
        np.zeros(len(X_del), dtype=bool)
    ])

    # 5) 调用 membership_inference
    _, _, (auc_rt, f1_rt) = membership_inference(
        X_train=X_attack,
        y_train=y_attack,
        hyperedges=he_attack,
        target_model=model,
        args=args,
        device=device,
        member_mask=member_mask
    )
    print(f"[MIA-Retrain Keep/Del] AUC={auc_rt:.4f}, F1={f1_rt:.4f}")
    # ———————————————————————————————

    return model

def full_train_and_unlearn(X, y, df, transformer, deleted_idx, args, device):
    """
    Trains on the full dataset, evaluates before unlearning,
    performs GIF-based unlearning on deleted_idx, then evaluates again.
    """

    # Build full hypergraph
    cat_cols = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ]
    hyperedges = generate_hyperedge_dict(
        df, cat_cols,
        max_nodes_per_hyperedge=getattr(args, 'max_nodes_per_hyperedge_train', 50),
        device=device
    )
    H_sp = build_incidence_matrix(hyperedges, X.shape[0])
    dv_np, de_np = compute_degree_vectors(H_sp)

    # Convert to torch sparse + tensors
    H_coo = H_sp.tocoo()
    idx = torch.LongTensor(np.vstack((H_coo.row, H_coo.col))).to(device)
    val = torch.FloatTensor(H_coo.data).to(device)
    H = torch.sparse_coo_tensor(idx, val, size=H_coo.shape).coalesce()
    dv = torch.FloatTensor(dv_np).to(device)
    de = torch.FloatTensor(de_np).to(device)
    fts = torch.FloatTensor(X).to(device)
    lbls = torch.LongTensor(y).to(device)

    # Prepare train_mask for GIF
    train_mask = torch.ones(X.shape[0], dtype=torch.bool, device=device)

    # Train full model
    model = HGNN_implicit(
        in_ch=fts.shape[1],
        n_class=len(np.unique(y)),
        n_hid=getattr(args, 'hidden_dim', 128),
        dropout=getattr(args, 'dropout', 0.1)
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=getattr(args, 'lr', 0.01), weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=getattr(args, 'milestones', [100,150]),
        gamma=getattr(args, 'gamma', 0.1)
    )
    # criterion = nn.CrossEntropyLoss()
    criterion = FocalLoss(gamma=2.0
                          # , weight=weights
                          , reduction='mean')

    model = train_model(
        model, criterion, optimizer, scheduler,
        fts, lbls, H, dv, de,
        num_epochs=getattr(args, 'epochs', 200), print_freq=10
    )

    # Prepare test data
    test_csv = ACI_TRAIN

    X_test, y_test, df_test, _ = preprocess_node_features(
        test_csv, is_test=True, transformer=transformer
    )
    hyperedges_test = generate_hyperedge_dict(
        df_test, cat_cols,
        max_nodes_per_hyperedge=getattr(args, 'max_nodes_per_hyperedge', 50),
        device=device
    )

    # 统计测试集里正负样本数量
    test_counter = Counter(y_test)
    num_neg_test = test_counter[0]
    num_pos_test = test_counter[1]
    print(f"测试集 → <=50K: {num_neg_test}，>50K: {num_pos_test}")

    H_sp_te = build_incidence_matrix(hyperedges_test, X_test.shape[0])
    dv_te, de_te = compute_degree_vectors(H_sp_te)
    H_coo_te = H_sp_te.tocoo()
    idx_te = torch.LongTensor(np.vstack((H_coo_te.row, H_coo_te.col))).to(device)
    val_te = torch.FloatTensor(H_coo_te.data).to(device)
    H_test = torch.sparse_coo_tensor(idx_te, val_te, size=H_coo_te.shape).coalesce()
    dv_test = torch.FloatTensor(dv_te).to(device)
    de_test = torch.FloatTensor(de_te).to(device)
    fts_test = torch.FloatTensor(X_test).to(device)
    lbls_test = torch.LongTensor(y_test).to(device)
    test_obj = {"x": fts_test, "y": lbls_test, "H": H_test, "dv_inv": dv_test, "de_inv": de_test}

    # Evaluate before unlearning
    f1_before = evaluate_test_f1(model, test_obj)
    acc_before = evaluate_test_acc(model, test_obj)
    print(f"GIF UNlearning以前: F1={f1_before:.4f}, Acc={acc_before:.4f}")

    # Build data_obj for GIF
    data_obj = {"x": fts, "y": lbls, "H": H, "dv_inv": dv, "de_inv": de, "train_mask": train_mask}

    with torch.no_grad():
        logits_after = model(
            test_obj["x"],
            test_obj["H"],
            test_obj["dv_inv"],
            test_obj["de_inv"]
        )
        preds_after = logits_after.argmax(dim=1)
        mismatches = (preds_after != test_obj["y"])
        num_errors = mismatches.sum().item()
        total = test_obj["y"].shape[0]

    print(f"[Unlearning以后] Test errors: {num_errors}/{total}  (err_rate={num_errors / total:.4f})")

    # Attach deletion logic
    def reason_once(data):
        return model(data["x"], data["H"], data["dv_inv"], data["de_inv"])

    def reason_once_unlearn(data):
        x_mod = data["x"].clone()
        x_mod[deleted_idx] = 0.0
        H_new, dv_new, de_new, _ = rebuild_structure_after_node_deletion(
            hyperedges, deleted_idx, X.shape[0], device
        )
        return model(x_mod, H_new, dv_new, de_new)

    model.reason_once = reason_once
    model.reason_once_unlearn = reason_once_unlearn

    # Perform GIF-based unlearning
    unlearning_time, _ = approx_gif(
        model, data_obj, (deleted_idx, None, None),
        iteration=getattr(args, 'if_iters', 20),
        damp=getattr(args, 'if_damp', 0.01),
        scale=getattr(args, 'if_scale', 1e7)
    )
    print(f"[Unlearn] 花费了 {unlearning_time:.4f}s")

    # Evaluate after unlearning
    f1_after = evaluate_test_f1(model, test_obj)
    acc_after = evaluate_test_acc(model, test_obj)
    print(f"Unlearning之后的:  F1={f1_after:.4f}, Acc={acc_after:.4f}")

    with torch.no_grad():
        logits_after = model(
            test_obj["x"],
            test_obj["H"],
            test_obj["dv_inv"],
            test_obj["de_inv"]
        )
        preds_after = logits_after.argmax(dim=1)  #
        mismatches = (preds_after != test_obj["y"])  #
        num_errors = mismatches.sum().item()  #
        total = test_obj["y"].shape[0]  #


    X_del = X[deleted_idx]
    y_del = y[deleted_idx]
    df_del = df.iloc[deleted_idx].reset_index(drop=True)

    hyperedges_del = generate_hyperedge_dict(
        df_del, cat_cols,
        max_nodes_per_hyperedge=getattr(args, 'max_nodes_per_hyperedge_train', 10),
        device=device
    )
    H_sp_del = build_incidence_matrix(hyperedges_del, len(deleted_idx))
    dv_np_del, de_np_del = compute_degree_vectors(H_sp_del)
    H_coo_del = H_sp_del.tocoo()
    idx_del = torch.LongTensor(np.vstack((H_coo_del.row, H_coo_del.col))).to(device)
    val_del = torch.FloatTensor(H_coo_del.data).to(device)
    H_del = torch.sparse_coo_tensor(idx_del, val_del, size=H_coo_del.shape).coalesce()

    fts_del = torch.FloatTensor(X_del).to(device)
    lbls_del = torch.LongTensor(y_del).to(device)
    dv_del = torch.FloatTensor(dv_np_del).to(device)
    de_del = torch.FloatTensor(de_np_del).to(device)
    test_obj_del = {
        "x": fts_del,
        "y": lbls_del,
        "H": H_del,
        "dv_inv": dv_del,
        "de_inv": de_del
    }
    num_hyperedges = len(hyperedges_del)
    has_nonzero_features = (fts_del.abs().sum().item() > 0)
    print(f"Number of hyperedges in deleted set: {num_hyperedges}, "
          f"Deleted-set node features not zeroed out? {has_nonzero_features}")
    model.eval()
    with torch.no_grad():
        f1_del = evaluate_test_f1(model, test_obj_del)
        acc_del = evaluate_test_acc(model, test_obj_del)

    print(f"[Deleted-Node Evaluation] Retrained model scores on deleted nodes: F1={f1_del:.4f}, Acc={acc_del:.4f}")


    all_idx  = np.arange(X.shape[0])
    keep_idx = np.setdiff1d(all_idx, deleted_idx)
    X_keep, y_keep = X[keep_idx], y[keep_idx]
    X_del,  y_del  = X[deleted_idx], y[deleted_idx]

    df_keep = df.drop(index=deleted_idx).reset_index(drop=True)
    df_del  = df.iloc[deleted_idx].reset_index(drop=True)
    he_keep = generate_hyperedge_dict(
        df_keep, cat_cols,
        max_nodes_per_hyperedge=getattr(args, 'max_nodes_per_hyperedge_train', 50),
        device=device
    )
    he_del  = generate_hyperedge_dict(
        df_del, cat_cols,
        max_nodes_per_hyperedge=getattr(args, 'max_nodes_per_hyperedge_train', 50),
        device=device
    )

    he_attack = {}
    idx_he = 0
    for nodes in he_keep.values():
        he_attack[idx_he] = nodes
        idx_he += 1
    offset = len(X_keep)
    for nodes in he_del.values():
        he_attack[idx_he] = [n + offset for n in nodes]
        idx_he += 1

    X_attack = np.vstack([X_keep, X_del])
    y_attack = np.hstack([y_keep, y_del])
    member_mask = np.concatenate([
        np.ones(len(X_keep), dtype=bool),
        np.zeros(len(X_del),  dtype=bool)
    ])

    _, _, (auc_un, f1_un) = membership_inference(
        X_train      = X_attack,
        y_train      = y_attack,
        hyperedges   = he_attack,
        target_model = model,
        args         = args,
        device       = device,
        member_mask  = member_mask
    )
    print(f"[MIA-Unlearned Keep/Del] AUC={auc_un:.4f}, F1={f1_un:.4f}")
    # ————————————————————

    return model

def main():
    args = get_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(args)

    # Load full training data
    train_csv = ACI_TRAIN
    X_train, y_train, df_train, transformer = preprocess_node_features(
        train_csv, is_test=False
    )
    y_train = np.asarray(y_train)
    num_train = X_train.shape[0]
    print(f"Number of nodes in full training: {num_train}, feature dim = {X_train.shape[1]}")
    cat_cols = [
        "workclass", "education", "marital-status",
        "occupation", "relationship", "race",
        "sex", "native-country"
    ]
    hypers_full = generate_hyperedge_dict(
        df_train, cat_cols,
        max_nodes_per_hyperedge = args.max_nodes_per_hyperedge_train,
        device = device
    )

    auc_mia, f1_mia = mia_with_hgnn(
        X_train   = X_train,
        y_train   = y_train,
        hyperedges= hypers_full,
        args      = args,
        device    = device
    )
    print(f"[MIA-Full]      AUC={auc_mia:.4f}, F1={f1_mia:.4f}")

    train_counter = Counter(y_train)
    num_neg_train = train_counter[0]
    num_pos_train = train_counter[1]
    print(f"Training set → <=50K: {num_neg_train}, >50K: {num_pos_train}")

    df_train = df_train.reset_index(drop=True)

    mask = (df_train['sex'] == 'Male') & (df_train['relationship'] =='Husband')

    deleted_idx = np.where(mask)[0]

    del_count = len(deleted_idx)
    print(f"Value-based deletion: {del_count} nodes in total, first 5 indices: {deleted_idx[:5]}")

    # Part 1: retrain on pruned data
    retrain_on_pruned(X_train, y_train, df_train, transformer, deleted_idx, args, device)

    # Part 2: full training + GIF unlearning
    full_train_and_unlearn(X_train, y_train, df_train, transformer, deleted_idx, args, device)

if __name__ == "__main__":
    # main()
    for run in range(1,21):
        print("=== Run",run,"===")
        main()