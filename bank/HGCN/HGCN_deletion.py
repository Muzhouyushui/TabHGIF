#!/usr/bin/env python
# coding: utf-8
"""
"""

import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data

from bank.HGNNP.HGNNP_utils import evaluate_test_acc, evaluate_test_f1
from bank.HGNNP.data_preprocessing_bank import (
    preprocess_node_features_bank,
    generate_hyperedge_dict_bank,
)
from bank.HGNNP.HGNNP import HGNNP_implicit, build_incidence_matrix, compute_degree_vectors
from bank.HGNNP.GIF_HGNNP_ROW import rebuild_structure_after_node_deletion, approx_gif, train_model
from bank.HGNNP.MIA_HGNNP import train_shadow_model, membership_inference
from bank.HGNNP.config import get_args  # Keep only this one to avoid repeated import conflicts


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


def _to_sparse_h(H_sp, device):
    """scipy sparse -> torch sparse coo"""
    Hc = H_sp.tocoo()
    idx = torch.LongTensor(np.vstack((Hc.row, Hc.col))).to(device)
    val = torch.FloatTensor(Hc.data).to(device)
    return torch.sparse_coo_tensor(idx, val, size=Hc.shape).coalesce()


def _build_eval_obj_bank(X_np, y_np, df_np_df, cat_cols, cont_cols, args, device):
    """
    """
    hypers = generate_hyperedge_dict_bank(
        df_np_df, cat_cols, cont_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        device=device
    )
    H_sp = build_incidence_matrix(hypers, X_np.shape[0])
    dv_np, de_np = compute_degree_vectors(H_sp)

    H = _to_sparse_h(H_sp, device)
    fts = torch.FloatTensor(X_np).to(device)
    lbls = torch.LongTensor(np.asarray(y_np)).to(device)
    dv = torch.FloatTensor(dv_np).to(device)
    de = torch.FloatTensor(de_np).to(device)

    eval_obj = {
        "x": fts, "y": lbls,
        "H": H, "dv_inv": dv, "de_inv": de
    }
    return eval_obj, hypers


def mia_with_hgnn(X_train, y_train, hyperedges, args, device):
    """
    """
    N = X_train.shape[0]
    H_sp = build_incidence_matrix(hyperedges, N)
    dv_inv, de_inv = compute_degree_vectors(H_sp)
    H_tensor = _to_sparse_h(H_sp, device)

    data_full = Data(
        x=torch.from_numpy(X_train).float().to(device),
        y=torch.from_numpy(y_train).long().to(device),
        H=H_tensor,
        dv_inv=torch.from_numpy(dv_inv).float().to(device),
        de_inv=torch.from_numpy(de_inv).float().to(device),
        train_mask=torch.ones(N, dtype=torch.bool, device=device),
        train_indices=list(range(N)),
        test_indices=[]
    )

    full_model = HGNNP_implicit(
        in_ch=X_train.shape[1],
        n_class=int(y_train.max()) + 1,
        n_hid=args.hidden_dim,
        dropout=args.dropout
    ).to(device)

    full_model = train_shadow_model(
        full_model,
        data_full,
        lr=args.lr,
        epochs=args.epochs
    )

    _, (_, _), (auc_t, f1_t) = membership_inference(
        X_train=X_train, y_train=y_train,
        hyperedges=hyperedges,
        target_model=full_model,
        args=args, device=device
    )
    return auc_t, f1_t


def retrain_on_pruned(X, y, df, transformer, deleted_idx, args, device):
    """
    """
    X = np.asarray(X)
    y = np.asarray(y)

    num_train = X.shape[0]
    keep_mask = np.ones(num_train, dtype=bool)
    keep_mask[deleted_idx] = False

    X_pr = X[keep_mask]
    y_pr = y[keep_mask]
    df_pr = df.loc[keep_mask].reset_index(drop=True)
    print(f"[Retrain] Number of training nodes after pruning: {X_pr.shape[0]} / original {num_train}")

    cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
    cont_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

    # Timing (total)
    t_start = time.time()

    # ===== 1) Rebuild training hypergraph =====
    t_hg = time.time()
    hypers_pr = generate_hyperedge_dict_bank(
        df_pr, cat_cols, cont_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        device=device
    )
    t_hg = time.time() - t_hg
    print(f"[Retrain] Hypergraph rebuild time: {t_hg:.2f}s, number of hyperedges: {len(hypers_pr)}")

    H_sp = build_incidence_matrix(hypers_pr, X_pr.shape[0])
    dv_np, de_np = compute_degree_vectors(H_sp)
    H_pr = _to_sparse_h(H_sp, device)
    dv = torch.FloatTensor(dv_np).to(device)
    de = torch.FloatTensor(de_np).to(device)
    fts = torch.FloatTensor(X_pr).to(device)
    lbls = torch.LongTensor(y_pr).to(device)

    # ===== 2) Retraining =====
    model = HGNNP_implicit(
        in_ch=fts.shape[1],
        n_class=int(np.unique(y).size),
        n_hid=args.hidden_dim,
        dropout=args.dropout
    ).to(device)

    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sch = optim.lr_scheduler.MultiStepLR(opt, milestones=args.milestones, gamma=args.gamma)
    crit = FocalLoss(gamma=2.0, reduction='mean')

    t_train = time.time()
    model = train_model(
        model, crit, opt, sch,
        fts, lbls, H_pr, dv, de,
        num_epochs=args.epochs,
        print_freq=(args.log_every or 10)
    )
    t_train = time.time() - t_train
    t_total = time.time() - t_start
    print(f"[Retrain] Completed, training time {t_train:.2f}s, total time {t_total:.2f}s")

    # ===== 3) Test set evaluation =====
    test_source = args.test_csv or args.train_csv
    X_te, y_te, df_te, _ = preprocess_node_features_bank(test_source, is_test=True, transformer=transformer)
    test_obj, hypers_te = _build_eval_obj_bank(X_te, y_te, df_te, cat_cols, cont_cols, args, device)

    f1_test = evaluate_test_f1(model, test_obj)
    acc_test = evaluate_test_acc(model, test_obj)
    print(f"[Retrain] Test-set F1: {f1_test:.4f}, Acc: {acc_test:.4f}")

    # ===== 4) ForgetMetric on deleted set (orig-feat) =====
    # Note: original deleted-node features are used here, not zeroed-out features
    X_del = X[deleted_idx]
    y_del = y[deleted_idx]
    df_del = df.loc[deleted_idx].reset_index(drop=True)

    del_obj, hypers_del = _build_eval_obj_bank(X_del, y_del, df_del, cat_cols, cont_cols, args, device)

    has_nonzero_features = (del_obj["x"].abs().sum().item() > 0)
    print(f"Number of hyperedges in deleted set: {len(hypers_del)}, deleted-set node features not zeroed out? {has_nonzero_features}")

    with torch.no_grad():
        f1_forget = evaluate_test_f1(model, del_obj)
        acc_forget = evaluate_test_acc(model, del_obj)

    print(f"[ForgetMetric|orig-feat|Retrain] deleted-set F1={f1_forget:.4f}, Acc={acc_forget:.4f}")

    # ===== 5) MIA: Keep vs Del =====
    all_idx = np.arange(num_train)
    keep_idx = np.setdiff1d(all_idx, deleted_idx)

    X_keep, y_keep = X[keep_idx], y[keep_idx]
    # Reuse the rebuilt hypers_pr here as the keep hyperedges
    hypers_keep = hypers_pr

    # del hyperedges need to be concatenated after offsetting
    he_attack = {}
    ptr = 0
    for nodes in hypers_keep.values():
        he_attack[ptr] = nodes
        ptr += 1
    offset = len(X_keep)
    for nodes in hypers_del.values():
        he_attack[ptr] = [n + offset for n in nodes]
        ptr += 1

    X_attack = np.vstack([X_keep, X_del])
    y_attack = np.hstack([y_keep, y_del])
    member_mask = np.concatenate([
        np.ones(len(X_keep), dtype=bool),   # keep = member
        np.zeros(len(X_del), dtype=bool)    # deleted = non-member
    ])

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

    return {
        "model": model,
        "time_total": t_total,
        "test_f1": f1_test,
        "test_acc": acc_test,
        "forget_f1": f1_forget,
        "forget_acc": acc_forget,
        "mia_auc": auc_rt,
        "mia_f1": f1_rt,
    }


def full_train_and_unlearn(X, y, df, transformer, deleted_idx, args, device):
    """
    Full training + GIF unlearning + test set evaluation + ForgetMetric on deleted set (orig-feat) + MIA(Keep/Del)

    Return:
      {
        model, unlearn_time,
        before_test_f1, before_test_acc,
        after_test_f1, after_test_acc,
        forget_f1, forget_acc,
        mia_auc, mia_f1
      }
    """
    X = np.asarray(X)
    y = np.asarray(y)

    cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
    cont_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

    # ===== 1) Full hypergraph and training =====
    hypers = generate_hyperedge_dict_bank(
        df, cat_cols, cont_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        device=device
    )

    H_sp = build_incidence_matrix(hypers, X.shape[0])
    dv_np, de_np = compute_degree_vectors(H_sp)
    H = _to_sparse_h(H_sp, device)
    dv = torch.FloatTensor(dv_np).to(device)
    de = torch.FloatTensor(de_np).to(device)
    fts = torch.FloatTensor(X).to(device)
    lbls = torch.LongTensor(y).to(device)

    model = HGNNP_implicit(
        in_ch=fts.shape[1],
        n_class=int(np.unique(y).size),
        n_hid=args.hidden_dim,
        dropout=args.dropout
    ).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sch = optim.lr_scheduler.MultiStepLR(opt, milestones=args.milestones, gamma=args.gamma)
    crit = FocalLoss(gamma=2.0, reduction='mean')

    model = train_model(
        model, crit, opt, sch, fts, lbls, H, dv, de,
        num_epochs=args.epochs, print_freq=(args.log_every or 10)
    )

    # ===== 2) Test set evaluation (before) =====
    test_source = args.test_csv or args.train_csv
    X_te, y_te, df_te, _ = preprocess_node_features_bank(test_source, is_test=True, transformer=transformer)

    # Only used to display class distribution (can be removed)
    cnt = Counter(y_te)
    if len(cnt) > 0:
        keys_sorted = sorted(cnt.keys())
        msg = "，".join([f"{k}: {cnt[k]}" for k in keys_sorted])
        print(f"Test-set label distribution → {msg}")

    test_obj, _ = _build_eval_obj_bank(X_te, y_te, df_te, cat_cols, cont_cols, args, device)

    f1_before = evaluate_test_f1(model, test_obj)
    acc_before = evaluate_test_acc(model, test_obj)
    print(f"Before Unlearning — F1: {f1_before:.4f}, Acc: {acc_before:.4f}")

    # ===== 3) Attach GIF deletion logic =====
    def reason_once(data):
        return model(data["x"], data["H"], data["dv_inv"], data["de_inv"])

    def reason_once_unlearn(data):
        """
        This is only used for the internal logic of GIF updates:
        - zero out deleted-node features
        - rebuild the post-deletion structure
        Note: this is not the evaluation protocol for ForgetMetric
        """
        x0 = data["x"].clone()
        x0[deleted_idx] = 0.0
        Hn, dvn, den, _ = rebuild_structure_after_node_deletion(hypers, deleted_idx, X.shape[0], device)
        return model(x0, Hn, dvn, den)

    model.reason_once = reason_once
    model.reason_once_unlearn = reason_once_unlearn

    # ===== 4) GIF Unlearning =====
    data_obj = {
        "x": fts, "y": lbls, "H": H, "dv_inv": dv, "de_inv": de,
        "train_mask": torch.ones(X.shape[0], dtype=torch.bool, device=device)
    }

    # Compatible with different parameter naming
    gif_iters = getattr(args, "gif_iters", getattr(args, "if_iters", 10))
    gif_damp = getattr(args, "gif_damp", getattr(args, "if_damp", 0.0))
    gif_scale = getattr(args, "gif_scale", getattr(args, "if_scale", 25.0))
    neighbor_k = getattr(args, "neighbor_k", 12)

    unlearn_info = (deleted_idx, hypers, neighbor_k)
    t_un, _ = approx_gif(
        model,
        data_obj,
        unlearn_info,
        iteration=gif_iters,
        damp=gif_damp,
        scale=gif_scale
    )
    print(f"After Unlearning — GIF time: {t_un:.2f}s")

    # ===== 5) Test set evaluation (after) =====
    f1_after = evaluate_test_f1(model, test_obj)
    acc_after = evaluate_test_acc(model, test_obj)
    print(f"After Unlearning — F1: {f1_after:.4f}, Acc: {acc_after:.4f}")

    # ===== 6) ForgetMetric on deleted set (orig-feat) =====
    X_del = X[deleted_idx]   # original deleted-node features (not zeroed out)
    y_del = y[deleted_idx]
    df_del = df.loc[deleted_idx].reset_index(drop=True)

    del_obj, hypers_del = _build_eval_obj_bank(X_del, y_del, df_del, cat_cols, cont_cols, args, device)

    has_nonzero_features = (del_obj["x"].abs().sum().item() > 0)
    print(f"Number of hyperedges in deleted set: {len(hypers_del)}, deleted-set node features not zeroed out? {has_nonzero_features}")

    with torch.no_grad():
        f1_forget = evaluate_test_f1(model, del_obj)
        acc_forget = evaluate_test_acc(model, del_obj)

    print(f"[ForgetMetric|orig-feat|HGIF] deleted-set F1={f1_forget:.4f}, Acc={acc_forget:.4f}")

    # ===== 7) MIA: Keep vs Del (on the unlearned model) =====
    all_idx = np.arange(X.shape[0])
    keep_idx = np.setdiff1d(all_idx, deleted_idx)

    X_keep, y_keep = X[keep_idx], y[keep_idx]
    X_del_attack, y_del_attack = X[deleted_idx], y[deleted_idx]

    df_keep = df.loc[keep_idx].reset_index(drop=True)
    df_del_attack = df.loc[deleted_idx].reset_index(drop=True)

    hypers_keep = generate_hyperedge_dict_bank(
        df_keep, cat_cols, cont_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        device=device
    )
    hypers_del_attack = generate_hyperedge_dict_bank(
        df_del_attack, cat_cols, cont_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        device=device
    )

    he_attack = {}
    idx_he = 0
    for nodes in hypers_keep.values():
        he_attack[idx_he] = nodes
        idx_he += 1
    offset = len(X_keep)
    for nodes in hypers_del_attack.values():
        he_attack[idx_he] = [n + offset for n in nodes]
        idx_he += 1

    X_attack = np.vstack([X_keep, X_del_attack])
    y_attack = np.hstack([y_keep, y_del_attack])
    member_mask = np.concatenate([
        np.ones(len(X_keep), dtype=bool),   # keep = member
        np.zeros(len(X_del_attack), dtype=bool)  # deleted = non-member
    ])

    _, _, (auc_unlearned, f1_unlearned) = membership_inference(
        X_train=X_attack,
        y_train=y_attack,
        hyperedges=he_attack,
        target_model=model,
        args=args,
        device=device,
        member_mask=member_mask
    )
    print(f"[MIA-Unlearned Keep/Del] AUC={auc_unlearned:.4f}, F1={f1_unlearned:.4f}")

    return {
        "model": model,
        "unlearn_time": t_un,
        "before_test_f1": f1_before,
        "before_test_acc": acc_before,
        "after_test_f1": f1_after,
        "after_test_acc": acc_after,
        "forget_f1": f1_forget,
        "forget_acc": acc_forget,
        "mia_auc": auc_unlearned,
        "mia_f1": f1_unlearned,
    }


def main():
    args = get_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(args)

    # 1) Data loading & preprocessing (training set)
    X_tr, y_tr, df_tr, trans = preprocess_node_features_bank(
        args.train_csv, is_test=False
    )
    X_tr = np.asarray(X_tr)
    y_tr = np.asarray(y_tr)
    print(f"Training set: {X_tr.shape[0]} nodes, dim={X_tr.shape[1]}")

    # 2) Full MIA (optional reference)
    cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
    cont_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    hypers_full = generate_hyperedge_dict_bank(
        df_tr, cat_cols, cont_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        device=device
    )
    auc_full, f1_full = mia_with_hgnn(X_tr, y_tr, hypers_full, args, device)
    print(f"[MIA-Full] AUC={auc_full:.4f}, F1={f1_full:.4f}")

    # 3) Randomly sample deleted_idx
    n = X_tr.shape[0]
    k = int(args.remove_ratio * n)
    deleted_idx = np.random.choice(n, k, replace=False)
    print(f"Delete {k} nodes (~{args.remove_ratio * 100:.0f}%): {deleted_idx[:5]}...")

    # 4) Retrain
    retrain_res = retrain_on_pruned(X_tr, y_tr, df_tr, trans, deleted_idx, args, device)

    # 5) HGIF Unlearning
    unlearn_res = full_train_and_unlearn(X_tr, y_tr, df_tr, trans, deleted_idx, args, device)

    # 6) Summary printout (you can directly copy this later for reviewer response)
    print("\n================ Summary (single run) ================")
    print("[Retrain]")
    print(f"  time_total      = {retrain_res['time_total']:.4f}s")
    print(f"  test_f1 / acc   = {retrain_res['test_f1']:.4f} / {retrain_res['test_acc']:.4f}")
    print(f"  forget_f1 / acc = {retrain_res['forget_f1']:.4f} / {retrain_res['forget_acc']:.4f}  (orig-feat)")
    print(f"  mia_auc / f1    = {retrain_res['mia_auc']:.4f} / {retrain_res['mia_f1']:.4f}")

    print("[HGIF]")
    print(f"  unlearn_time    = {unlearn_res['unlearn_time']:.4f}s")
    print(f"  before test     = {unlearn_res['before_test_f1']:.4f} / {unlearn_res['before_test_acc']:.4f}")
    print(f"  after  test     = {unlearn_res['after_test_f1']:.4f} / {unlearn_res['after_test_acc']:.4f}")
    print(f"  forget_f1 / acc = {unlearn_res['forget_f1']:.4f} / {unlearn_res['forget_acc']:.4f}  (orig-feat)")
    print(f"  mia_auc / f1    = {unlearn_res['mia_auc']:.4f} / {unlearn_res['mia_f1']:.4f}")

    # Optional: concise comparison (convenient for copying into tables)
    print("\n[Compare]")
    print(f"  Test Acc  (Retrain vs HGIF-after): {retrain_res['test_acc']:.4f} vs {unlearn_res['after_test_acc']:.4f}")
    print(f"  Forget Acc(orig-feat):             {retrain_res['forget_acc']:.4f} vs {unlearn_res['forget_acc']:.4f}")
    print(f"  MIA AUC  (Keep/Del):               {retrain_res['mia_auc']:.4f} vs {unlearn_res['mia_auc']:.4f}")


if __name__ == "__main__":
    main()