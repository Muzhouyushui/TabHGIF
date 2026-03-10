#!/usr/bin/env python
# coding: utf-8
"""
HGNN_Unlearning_row_bank_Value.py

Bank dataset + HGNN + Value (Row-Group) Unlearning
Logic aligned with:
- HGNN_Unlearning_row_bank.py: two-stage pipeline with retrain_on_pruned + full_train_and_unlearn
- HGCN_Unlearning_row_bank_Value.py: construct mask by value -> deleted_idx
- HGNN_Unlearning_Value.py: MIA / keep-del attack concatenation and member_mask
"""

import argparse
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

# HGNN (Bank)
from bank.HGNN.HGNN_utils import evaluate_test_acc, evaluate_test_f1
from bank.HGNN.data_preprocessing_bank import (
    preprocess_node_features_bank,
    generate_hyperedge_dict_bank,
)
from bank.HGNN.HGNN import HGNN_implicit, build_incidence_matrix, compute_degree_vectors
from bank.HGNN.GIF_HGNN_ROW import (
    rebuild_structure_after_node_deletion,
    approx_gif,
    train_model,
)

# MIA (same import style as HGNN_Unlearning_row_bank.py)
from MIA_HGNN import train_shadow_model, membership_inference
from paths import BANK_DATA

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce_loss)
        loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

def mia_with_hgnn(X_train, y_train, hyperedges, args, device):
    """
    Consistent with your HGNN example:
    first train full_model, then call membership_inference to obtain target AUC/F1.
    """
    N = X_train.shape[0]

    H_sp = build_incidence_matrix(hyperedges, N)
    dv_inv, de_inv = compute_degree_vectors(H_sp)
    Hc = H_sp.tocoo()
    idx = torch.LongTensor([Hc.row, Hc.col])
    val = torch.FloatTensor(Hc.data)
    H_tensor = torch.sparse_coo_tensor(idx, val, Hc.shape).to(device)

    full_train_mask = torch.ones(N, dtype=torch.bool, device=device)
    data_full = Data(
        x=torch.from_numpy(X_train).float().to(device),
        y=torch.from_numpy(y_train).long().to(device),
        H=H_tensor,
        dv_inv=torch.from_numpy(dv_inv).to(device),
        de_inv=torch.from_numpy(de_inv).to(device),
        train_mask=full_train_mask,
        train_indices=list(range(N)),
        test_indices=[],
    )

    full_model = HGNN_implicit(
        in_ch=X_train.shape[1],
        n_class=int(y_train.max()) + 1,
        n_hid=args.full_hidden,
        dropout=args.full_dropout,
    ).to(device)

    full_model = train_shadow_model(full_model, data_full, lr=args.full_lr, epochs=args.full_epochs)

    _, (_, _), (auc_target, f1_target) = membership_inference(
        X_train=X_train,
        y_train=y_train,
        hyperedges=hyperedges,
        target_model=full_model,
        args=args,
        device=device,
    )
    return auc_target, f1_target

def retrain_on_pruned(X, y, df, transformer, deleted_idx, args, device):
    """
    Align with the retrain structure in HGNN_Unlearning_row_bank.py:
    - rebuild the hypergraph on the pruned training set -> train HGNN -> evaluate on the test set
    - additionally: deleted-set evaluation + Retrain keep/del MIA (same as the example)
    """
    num_train = X.shape[0]
    keep_mask = np.ones(num_train, dtype=bool)
    keep_mask[deleted_idx] = False

    y = np.asarray(y)

    X_pruned = X[keep_mask]
    y_pruned = y[keep_mask]
    df_pruned = df.loc[keep_mask].reset_index(drop=True)

    print(f"[Retrain] Number of training nodes after pruning: {X_pruned.shape[0]} / original {num_train}")

    t_start = time.time()

    cat_cols = args.cat_cols
    cont_cols = args.cont_cols

    t0 = time.time()
    hyperedges_pruned = generate_hyperedge_dict_bank(
        df_pruned,
        cat_cols,
        cont_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge_train,
        device=device,
    )
    t_hg = time.time() - t0
    print(f"[Retrain] Hypergraph rebuild time: {t_hg:.2f}s")
    print(f"[Retrain] Total number of hyperedges after pruning: {len(hyperedges_pruned)}")

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
        n_hid=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    criterion = FocalLoss(gamma=2.0, reduction="mean")
    t_init_end = time.time()

    t_train_start = time.time()
    model = train_model(
        model,
        criterion,
        optimizer,
        scheduler,
        fts,
        lbls,
        H_pruned,
        dv,
        de,
        num_epochs=args.epochs,
        print_freq=args.log_every,
    )
    t_train_end = time.time()
    t_total = time.time() - t_start
    print(
        f"[Retrain] Initialization: {t_init_end - t_init_start:.2f}s | "
        f"Training: {t_train_end - t_train_start:.2f}s | "
        f"Total: {t_total:.2f}s"
    )

    # ---- Test eval (use the same path as in your bank row script) ----
    test_csv = args.data_csv
    X_test, y_test, df_test, _ = preprocess_node_features_bank(test_csv, is_test=True, transformer=transformer)
    t2 = time.time()
    hyperedges_test = generate_hyperedge_dict_bank(
        df_test,
        cat_cols,
        cont_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        device=device,
    )
    print(f"[Retrain] Test-set hypergraph rebuild time: {time.time() - t2:.2f}s")
    print(f"[Retrain] Total number of hyperedges in the test set: {len(hyperedges_test)}")

    H_sp_te = build_incidence_matrix(hyperedges_test, X_test.shape[0])
    dv_te, de_te = compute_degree_vectors(H_sp_te)
    H_coo_te = H_sp_te.tocoo()
    idx_te = torch.LongTensor(np.vstack((H_coo_te.row, H_coo_te.col))).to(device)
    val_te = torch.FloatTensor(H_coo_te.data).to(device)
    H_test = torch.sparse_coo_tensor(idx_te, val_te, size=H_coo_te.shape).coalesce()

    test_obj = {
        "x": torch.FloatTensor(X_test).to(device),
        "y": torch.LongTensor(y_test).to(device),
        "H": H_test,
        "dv_inv": torch.FloatTensor(dv_te).to(device),
        "de_inv": torch.FloatTensor(de_te).to(device),
    }
    f1_res = evaluate_test_f1(model, test_obj)
    acc = evaluate_test_acc(model, test_obj)
    f1_val = f1_res[0] if isinstance(f1_res, tuple) else f1_res
    print(f"[Retrain] Test-set F1: {f1_val:.4f}, Acc: {acc:.4f}")

    # ---- Deleted set eval ----
    X_del = X[deleted_idx]
    y_del = y[deleted_idx]
    df_del = df.iloc[deleted_idx].reset_index(drop=True)

    hyperedges_del = generate_hyperedge_dict_bank(
        df_del, cat_cols, cont_cols,
        max_nodes_per_hyperedge=min(args.max_nodes_per_hyperedge_train, 50),
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
    test_obj_del = {
        "x": fts_del,
        "y": lbls_del,
        "H": H_del,
        "dv_inv": torch.FloatTensor(dv_np_del).to(device),
        "de_inv": torch.FloatTensor(de_np_del).to(device),
    }

    num_hyperedges = len(hyperedges_del)
    has_nonzero_features = (fts_del.abs().sum().item() > 0)
    print(f"Deleted-set hyperedge count: {num_hyperedges}, deleted-set node features are not zeroed out? {has_nonzero_features}")

    model.eval()
    with torch.no_grad():
        f1_del = evaluate_test_f1(model, test_obj_del)
        acc_del = evaluate_test_acc(model, test_obj_del)
    print(f"[Deleted-node Evaluation] Scores of the retrained model on the deleted nodes: F1={f1_del:.4f}, Acc={acc_del:.4f}")

    # ---- MIA on retrained model (keep vs del) ----
    all_idx = np.arange(num_train)
    keep_idx = np.setdiff1d(all_idx, deleted_idx)
    X_keep, y_keep = X[keep_idx], y[keep_idx]
    X_del2, y_del2 = X[deleted_idx], y[deleted_idx]

    he_keep = hyperedges_pruned  # dict
    he_del = generate_hyperedge_dict_bank(
        df_del, cat_cols, cont_cols,
        max_nodes_per_hyperedge=min(args.max_nodes_per_hyperedge_train, 50),
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

    X_attack = np.vstack([X_keep, X_del2])
    y_attack = np.hstack([y_keep, y_del2])
    member_mask = np.concatenate([
        np.ones(len(X_keep), dtype=bool),
        np.zeros(len(X_del2), dtype=bool),
    ])

    _, _, (auc_rt, f1_rt) = membership_inference(
        X_train=X_attack,
        y_train=y_attack,
        hyperedges=he_attack,
        target_model=model,
        args=args,
        device=device,
        member_mask=member_mask,
    )
    print(f"[MIA-Retrain Keep/Del] AUC={auc_rt:.4f}, F1={f1_rt:.4f}")

    return model

def full_train_and_unlearn(X, y, df, transformer, deleted_idx, args, device):
    """
    Align with full + GIF unlearning in HGNN_Unlearning_row_bank.py:
    - full train
    - test before
    - reason_once_unlearn: zero out features of deleted_idx nodes + rebuild structure
    - approx_gif
    - test after + deleted-set eval + MIA(unlearned)
    """
    cat_cols = args.cat_cols
    cont_cols = args.cont_cols

    hyperedges = generate_hyperedge_dict_bank(
        df, cat_cols, cont_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge_train,
        device=device,
    )
    H_sp = build_incidence_matrix(hyperedges, X.shape[0])
    dv_np, de_np = compute_degree_vectors(H_sp)

    H_coo = H_sp.tocoo()
    idx = torch.LongTensor(np.vstack((H_coo.row, H_coo.col))).to(device)
    val = torch.FloatTensor(H_coo.data).to(device)
    H = torch.sparse_coo_tensor(idx, val, size=H_coo.shape).coalesce()

    dv = torch.FloatTensor(dv_np).to(device)
    de = torch.FloatTensor(de_np).to(device)
    fts = torch.FloatTensor(X).to(device)
    lbls = torch.LongTensor(y).to(device)
    train_mask = torch.ones(X.shape[0], dtype=torch.bool, device=device)

    model = HGNN_implicit(
        in_ch=fts.shape[1],
        n_class=len(np.unique(y)),
        n_hid=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    criterion = FocalLoss(gamma=2.0, reduction="mean")

    # Attach reason_once / reason_once_unlearn for GIF (aligned with bank row example)
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

    model = train_model(
        model,
        criterion,
        optimizer,
        scheduler,
        fts,
        lbls,
        H,
        dv,
        de,
        num_epochs=args.epochs,
        print_freq=args.log_every,
    )

    # ---- test set build (follow your bank row style: test_csv=args.data_csv) ----
    test_csv = args.data_csv
    X_test, y_test, df_test, _ = preprocess_node_features_bank(test_csv, is_test=True, transformer=transformer)
    hyperedges_test = generate_hyperedge_dict_bank(
        df_test, cat_cols, cont_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        device=device
    )

    test_counter = Counter(y_test)
    print(f"Test set → 0: {test_counter[0]}, 1: {test_counter[1]}")

    H_sp_te = build_incidence_matrix(hyperedges_test, X_test.shape[0])
    dv_te, de_te = compute_degree_vectors(H_sp_te)
    H_coo_te = H_sp_te.tocoo()
    idx_te = torch.LongTensor(np.vstack((H_coo_te.row, H_coo_te.col))).to(device)
    val_te = torch.FloatTensor(H_coo_te.data).to(device)
    H_test = torch.sparse_coo_tensor(idx_te, val_te, size=H_coo_te.shape).coalesce()

    test_obj = {
        "x": torch.FloatTensor(X_test).to(device),
        "y": torch.LongTensor(y_test).to(device),
        "H": H_test,
        "dv_inv": torch.FloatTensor(dv_te).to(device),
        "de_inv": torch.FloatTensor(de_te).to(device),
    }

    f1_before = evaluate_test_f1(model, test_obj)
    acc_before = evaluate_test_acc(model, test_obj)
    print(f"[Before Unlearning] Test F1={f1_before:.4f}, Acc={acc_before:.4f}")

    data_obj = {"x": fts, "y": lbls, "H": H, "dv_inv": dv, "de_inv": de, "train_mask": train_mask}

    # ---- GIF unlearning ----
    unlearning_time, _ = approx_gif(
        model,
        data_obj,
        (deleted_idx, hyperedges, args.neighbor_k),   # keep the same passing style as your bank row script
        iteration=args.gif_iters,
        damp=args.gif_damp,
        scale=args.gif_scale,
    )
    print(f"[Unlearn] Took {unlearning_time:.4f}s")

    f1_after = evaluate_test_f1(model, test_obj)
    acc_after = evaluate_test_acc(model, test_obj)
    print(f"[After Unlearning]  Test F1={f1_after:.4f}, Acc={acc_after:.4f}")

    # ---- deleted set eval (HGIF on deleted nodes) ----
    y_np = np.asarray(y)
    X_del = X[deleted_idx]
    y_del = y_np[deleted_idx]
    df_del = df.iloc[deleted_idx].reset_index(drop=True)

    hyperedges_del = generate_hyperedge_dict_bank(
        df_del, cat_cols, cont_cols,
        max_nodes_per_hyperedge=min(args.max_nodes_per_hyperedge_train, 50),
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
    test_obj_del = {
        "x": fts_del,
        "y": lbls_del,
        "H": H_del,
        "dv_inv": torch.FloatTensor(dv_np_del).to(device),
        "de_inv": torch.FloatTensor(de_np_del).to(device),
    }

    num_hyperedges = len(hyperedges_del)
    has_nonzero_features = (fts_del.abs().sum().item() > 0)
    print(f"Deleted-set hyperedge count: {num_hyperedges}, deleted-set node features are not zeroed out? {has_nonzero_features}")

    model.eval()
    with torch.no_grad():
        f1_del = evaluate_test_f1(model, test_obj_del)
        acc_del = evaluate_test_acc(model, test_obj_del)
    print(f"[Deleted-node Evaluation] HGIF scores on the deleted nodes: F1={f1_del:.4f}, Acc={acc_del:.4f}")

    # ---- MIA on unlearned model (keep vs del) ----
    all_idx = np.arange(X.shape[0])
    keep_idx = np.setdiff1d(all_idx, deleted_idx)
    X_keep, y_keep = X[keep_idx], y_np[keep_idx]
    X_del2, y_del2 = X[deleted_idx], y_np[deleted_idx]

    df_keep = df.drop(index=deleted_idx).reset_index(drop=True)
    df_del2 = df.iloc[deleted_idx].reset_index(drop=True)

    he_keep = generate_hyperedge_dict_bank(
        df_keep, cat_cols, cont_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge_train,
        device=device
    )
    he_del2 = generate_hyperedge_dict_bank(
        df_del2, cat_cols, cont_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge_train,
        device=device
    )

    he_attack = {}
    idx_he = 0
    for nodes in he_keep.values():
        he_attack[idx_he] = nodes
        idx_he += 1
    offset = len(X_keep)
    for nodes in he_del2.values():
        he_attack[idx_he] = [n + offset for n in nodes]
        idx_he += 1

    X_attack = np.vstack([X_keep, X_del2])
    y_attack = np.hstack([y_keep, y_del2])
    member_mask = np.concatenate([
        np.ones(len(X_keep), dtype=bool),
        np.zeros(len(X_del2), dtype=bool),
    ])

    _, _, (auc_un, f1_un) = membership_inference(
        X_train=X_attack,
        y_train=y_attack,
        hyperedges=he_attack,
        target_model=model,
        args=args,
        device=device,
        member_mask=member_mask
    )
    print(f"[MIA-Unlearned Keep/Del] AUC={auc_un:.4f}, F1={f1_un:.4f}")

    return model

def parse_value_rule(rule: str):
    """
    Support the simplest rules:
    - "marital=married&housing=yes"
    - "job=blue-collar"
    Return: list[ (col, value) ]
    """
    rule = rule.strip()
    if not rule:
        return []
    parts = [p.strip() for p in rule.split("&") if p.strip()]
    out = []
    for p in parts:
        if "=" not in p:
            continue
        c, v = p.split("=", 1)
        out.append((c.strip(), v.strip()))
    return out

def main():
    parser = argparse.ArgumentParser(description="HGNN on Bank (Value/Row-Group Unlearning)")

    parser.add_argument("--data-csv", type=str, default=BANK_DATA,
                        help="Bank Marketing CSV path (; separated)")
    parser.add_argument("--split-ratio", type=float, default=0.2, help="Train/test split ratio")

    parser.add_argument("--cat-cols", nargs="+", type=str,
                        default=['job','marital','education','default','housing','loan','contact','month','poutcome'],
                        help="List of categorical feature column names")
    parser.add_argument("--cont-cols", nargs="+", type=str,
                        default=['age','balance','day','duration','campaign','pdays','previous'],
                        help="List of continuous feature column names")

    parser.add_argument("--max-nodes-per-hyperedge", type=int, default=50, help="Maximum hyperedge size for test/evaluation")
    parser.add_argument("--max-nodes-per-hyperedge-train", type=int, default=50, help="Maximum hyperedge size for training")

    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--milestones", nargs="+", type=int, default=[100, 150])
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=10)

    parser.add_argument("--gif-iters", type=int, default=20)
    parser.add_argument("--gif-damp", type=float, default=0.01)
    parser.add_argument("--gif-scale", type=float, default=1e7)
    parser.add_argument("--neighbor-k", type=int, default=13, help="Neighbor/shared-hyperedge threshold (following your HGNN ROW style)")

    # Full-model MIA (optional, off by default; enable when needed)
    parser.add_argument("--do-full-mia", action="store_true")
    parser.add_argument("--full-hidden", type=int, default=128)
    parser.add_argument("--full-dropout", type=float, default=0.1)
    parser.add_argument("--full-lr", type=float, default=0.01)
    parser.add_argument("--full-epochs", type=int, default=100)

    # Value unlearning rule: by default reproduce your HGCN bank value example
    parser.add_argument("--value-rule", type=str, default="marital=married&housing=yes",
                        help='Value-based deletion rule, e.g. "marital=married&housing=yes"')

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # 1) Read CSV & split (aligned with your bank value example: 80/20 stratify)
    df_full = pd.read_csv(args.data_csv, sep=";", skipinitialspace=True)
    df_tr, df_te = train_test_split(
        df_full,
        test_size=args.split_ratio,
        stratify=df_full["y"],
        random_state=42,
    )
    df_tr = df_tr.reset_index(drop=True)
    df_te = df_te.reset_index(drop=True)
    print(f"Train: {len(df_tr)}, Test: {len(df_te)}")

    # 2) Preprocessing (HGNN Bank preprocess)
    X_tr, y_tr, df_tr_proc, transformer = preprocess_node_features_bank(df_tr, is_test=False)
    # Note: use the original df_tr for value mask (consistent with HGCN bank value), not df_tr_proc
    # Construct deleted indices deleted_idx
    conds = parse_value_rule(args.value_rule)
    mask = np.ones(len(df_tr), dtype=bool)
    for c, v in conds:
        mask &= (df_tr[c].astype(str) == str(v))
    deleted_idx = np.where(mask)[0]
    pct = 100.0 * len(deleted_idx) / len(df_tr)
    print(f"Value-based deletion rule: {args.value_rule}")
    print(f"Delete by value: {len(deleted_idx)}/{len(df_tr)} nodes in total (≈{pct:.2f}%), first 5 indices: {deleted_idx[:5]}")

    # 3) Optional: Full model MIA (enable if needed for paper-style MIA reporting)
    if args.do_full_mia:
        hyperedges_full = generate_hyperedge_dict_bank(
            df_tr_proc, args.cat_cols, args.cont_cols,
            max_nodes_per_hyperedge=args.max_nodes_per_hyperedge_train,
            device=device
        )
        auc_full, f1_full = mia_with_hgnn(X_tr, y_tr, hyperedges_full, args, device)
        print(f"[MIA-Full] AUC={auc_full:.4f}, F1={f1_full:.4f}")

    # 4) Print training label distribution (style consistent with your script)
    train_counter = Counter(y_tr)
    print(f"Training set → 0: {train_counter[0]}, 1: {train_counter[1]}")

    # 5) Retrain baseline
    print("\n=== Retrain on pruned dataset (Value Unlearning) ===")
    _ = retrain_on_pruned(
        X=X_tr,
        y=y_tr,
        df=df_tr,               # pass the original df_tr to preserve original categorical values
        transformer=transformer,
        deleted_idx=deleted_idx,
        args=args,
        device=device,
    )

    # 6) Full train + HGIF
    print("\n=== Full training + HGIF Unlearning (Value Unlearning) ===")
    _ = full_train_and_unlearn(
        X=X_tr,
        y=y_tr,
        df=df_tr,               # likewise pass the original df_tr
        transformer=transformer,
        deleted_idx=deleted_idx,
        args=args,
        device=device,
    )

if __name__ == "__main__":
    main()