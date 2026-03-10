#!/usr/bin/env python
# coding: utf-8
"""
train_unlearning_hgnn_refactored.py

Refactored version for Bank Marketing 数据集:
1) retrain_on_pruned: 删除指定节点后重训练并评估
2) full_train_and_unlearn: 全量训练 + GIF unlearning 并评估
使用同一组 deleted_idx。
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
from config import get_args
from bank.HGNNP.config import get_args

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


def mia_with_hgnn(X_train, y_train, hyperedges, args, device):
    N = X_train.shape[0]
    H_sp = build_incidence_matrix(hyperedges, N)
    dv_inv, de_inv = compute_degree_vectors(H_sp)
    Hc = H_sp.tocoo()
    idx = torch.LongTensor(np.vstack((Hc.row, Hc.col))).to(device)
    val = torch.FloatTensor(Hc.data).to(device)
    H_tensor = torch.sparse_coo_tensor(idx, val, Hc.shape).to(device)

    data_full = Data(
        x=torch.from_numpy(X_train).float().to(device),
        y=torch.from_numpy(y_train).long().to(device),
        H=H_tensor,
        dv_inv=torch.from_numpy(dv_inv).to(device),
        de_inv=torch.from_numpy(de_inv).to(device),
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
    y = np.asarray(y)
    num_train = X.shape[0]
    keep_mask = np.ones(num_train, dtype=bool)
    keep_mask[deleted_idx] = False

    X_pr = X[keep_mask]
    y_pr = y[keep_mask]
    df_pr = df.loc[keep_mask].reset_index(drop=True)
    print(f"[重训练] 剪裁后训练节点数: {X_pr.shape[0]} / 原始 {num_train}")

    t0 = time.time()
    cat_cols = ['job','marital','education','default','housing','loan','contact','month','poutcome']
    cont_cols = ['age','balance','day','duration','campaign','pdays','previous']
    hypers_pr = generate_hyperedge_dict_bank(
        df_pr, cat_cols, cont_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        device=device
    )
    print(f"[重训练] 超图重建时间: {time.time()-t0:.2f}s, 超边数: {len(hypers_pr)}")

    H_sp = build_incidence_matrix(hypers_pr, X_pr.shape[0])
    dv_np, de_np = compute_degree_vectors(H_sp)
    Hc = H_sp.tocoo()
    idx = torch.LongTensor(np.vstack((Hc.row, Hc.col))).to(device)
    val = torch.FloatTensor(Hc.data).to(device)
    H_pr = torch.sparse_coo_tensor(idx, val, size=Hc.shape).coalesce()
    dv = torch.FloatTensor(dv_np).to(device)
    de = torch.FloatTensor(de_np).to(device)
    fts = torch.FloatTensor(X_pr).to(device)
    lbls = torch.LongTensor(y_pr).to(device)

    model = HGNNP_implicit(
        in_ch=fts.shape[1],
        n_class=int(np.unique(y).size),
        n_hid=args.hidden_dim,
        dropout=args.dropout
    ).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sch = optim.lr_scheduler.MultiStepLR(opt, milestones=args.milestones, gamma=args.gamma)
    crit = FocalLoss(gamma=2.0, reduction='mean')

    model = train_model(model, crit, opt, sch,
                        fts, lbls, H_pr, dv, de,
                        num_epochs=args.epochs, print_freq=args.log_every or 10)
    print(f"[重训练] 完成, 耗时 {time.time()-t0:.2f}s")

    # 测试集评估
    test_source = args.test_csv or args.train_csv
    X_te, y_te, df_te, _ = preprocess_node_features_bank(test_source, is_test=True, transformer=transformer)
    hypers_te = generate_hyperedge_dict_bank(
        df_te, cat_cols, cont_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        device=device
    )
    H_sp_te = build_incidence_matrix(hypers_te, X_te.shape[0])
    dv_te, de_te = compute_degree_vectors(H_sp_te)
    Hc_te = H_sp_te.tocoo()
    idx_te = torch.LongTensor(np.vstack((Hc_te.row, Hc_te.col))).to(device)
    val_te = torch.FloatTensor(Hc_te.data).to(device)
    H_te = torch.sparse_coo_tensor(idx_te, val_te, size=Hc_te.shape).coalesce()
    fts_te = torch.FloatTensor(X_te).to(device)
    lbls_te = torch.LongTensor(y_te).to(device)
    test_obj = {
        "x": fts_te, "y": lbls_te,
        "H": H_te, "dv_inv": torch.FloatTensor(dv_te).to(device),
        "de_inv": torch.FloatTensor(de_te).to(device)
    }

    f1 = evaluate_test_f1(model, test_obj)
    acc = evaluate_test_acc(model, test_obj)
    print(f"[重训练] 测试集 F1: {f1:.4f}, Acc: {acc:.4f}")

    return model


def full_train_and_unlearn(X, y, df, transformer, deleted_idx, args, device):
    y = np.asarray(y)
    cat_cols = ['job','marital','education','default','housing','loan','contact','month','poutcome']
    cont_cols = ['age','balance','day','duration','campaign','pdays','previous']
    hypers = generate_hyperedge_dict_bank(
        df, cat_cols, cont_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        device=device
    )

    H_sp = build_incidence_matrix(hypers, X.shape[0])
    dv_np, de_np = compute_degree_vectors(H_sp)
    Hc = H_sp.tocoo()
    idx = torch.LongTensor(np.vstack((Hc.row, Hc.col))).to(device)
    val = torch.FloatTensor(Hc.data).to(device)
    H = torch.sparse_coo_tensor(idx, val, size=Hc.shape).coalesce()
    dv = torch.FloatTensor(dv_np).to(device)
    de = torch.FloatTensor(de_np).to(device)
    fts = torch.FloatTensor(X).to(device)
    lbls = torch.LongTensor(y).to(device)

    model = HGNNP_implicit(
        in_ch=fts.shape[1], n_class=int(np.unique(y).size),
        n_hid=args.hidden_dim, dropout=args.dropout
    ).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sch = optim.lr_scheduler.MultiStepLR(opt, milestones=args.milestones, gamma=args.gamma)
    crit = FocalLoss(gamma=2.0, reduction='mean')

    model = train_model(model, crit, opt, sch, fts, lbls, H, dv, de,
                        num_epochs=args.epochs, print_freq=args.log_every or 10)

    test_source = args.test_csv or args.train_csv
    X_te, y_te, df_te, _ = preprocess_node_features_bank(test_source, is_test=True, transformer=transformer)
    hypers_te = generate_hyperedge_dict_bank(
        df_te, cat_cols, cont_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        device=device
    )
    H_sp_te = build_incidence_matrix(hypers_te, X_te.shape[0])
    dv_te, de_te = compute_degree_vectors(H_sp_te)
    Hc_te = H_sp_te.tocoo()
    idx_te = torch.LongTensor(np.vstack((Hc_te.row, Hc_te.col))).to(device)
    val_te = torch.FloatTensor(Hc_te.data).to(device)
    H_te = torch.sparse_coo_tensor(idx_te, val_te, size=Hc_te.shape).coalesce()
    fts_te = torch.FloatTensor(X_te).to(device)
    lbls_te = torch.LongTensor(y_te).to(device)
    test_obj = {
        "x": fts_te, "y": lbls_te,
        "H": H_te, "dv_inv": torch.FloatTensor(dv_te).to(device),
        "de_inv": torch.FloatTensor(de_te).to(device)
    }

    f1b = evaluate_test_f1(model, test_obj)
    acb = evaluate_test_acc(model, test_obj)
    print(f"Before Unlearning — F1: {f1b:.4f}, Acc: {acb:.4f}")

    # 挂载删除逻辑
    def reason_once(data): return model(data["x"], data["H"], data["dv_inv"], data["de_inv"])
    def reason_once_unlearn(data):
        x0 = data["x"].clone(); x0[deleted_idx] = 0.0
        Hn, dvn, den, _ = rebuild_structure_after_node_deletion(hypers, deleted_idx, X.shape[0], device)
        return model(x0, Hn, dvn, den)
    model.reason_once = reason_once
    model.reason_once_unlearn = reason_once_unlearn

    # GIF Unlearning
    unlearn_info = (deleted_idx, hypers, args.neighbor_k)
    t_un, _ = approx_gif(
        model,
        {"x":fts, "y":lbls, "H":H, "dv_inv":dv, "de_inv":de, "train_mask":torch.ones(X.shape[0],dtype=torch.bool,device=device)},
        unlearn_info,
        iteration=args.gif_iters,
        damp=args.gif_damp,
        scale=args.gif_scale
    )
    print(f"After Unlearning — GIF 耗时: {t_un:.2f}s")

    f1a = evaluate_test_f1(model, test_obj)
    aca = evaluate_test_acc(model, test_obj)
    print(f"After Unlearning — F1: {f1a:.4f}, Acc: {aca:.4f}")

    # —————— 新增：Unlearned 后的 MIA 测试 ——————
    # 构造 keep / del 集合
    all_idx = np.arange(X.shape[0])
    keep_idx = np.setdiff1d(all_idx, deleted_idx)
    X_keep, y_keep = X[keep_idx], y[keep_idx]
    X_del,  y_del  = X[deleted_idx], y[deleted_idx]

    # 重建 keep / del 超边
    df_keep = df.loc[keep_idx].reset_index(drop=True)
    df_del   = df.loc[deleted_idx].reset_index(drop=True)
    hypers_keep = generate_hyperedge_dict_bank(
        df_keep, cat_cols, cont_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        device=device
    )
    hypers_del = generate_hyperedge_dict_bank(
        df_del, cat_cols, cont_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        device=device
    )

    # 合并到一个攻击超边字典 (del 部分索引 +offset)
    he_attack = {}
    idx_he = 0
    for nodes in hypers_keep.values():
        he_attack[idx_he] = nodes
        idx_he += 1
    offset = len(X_keep)
    for nodes in hypers_del.values():
        he_attack[idx_he] = [n + offset for n in nodes]
        idx_he += 1

    # 构造攻击集特征和标签，以及 member_mask
    X_attack = np.vstack([X_keep, X_del])
    y_attack = np.hstack([y_keep, y_del])
    member_mask = np.concatenate([
        np.ones(len(X_keep), dtype=bool),  # keep 为正例
        np.zeros(len(X_del),  dtype=bool)  # del 为负例
    ])

    # 调用 membership_inference 评估
    _, _, (auc_unlearned, f1_unlearned) = membership_inference(
        X_train      = X_attack,
        y_train      = y_attack,
        hyperedges   = he_attack,
        target_model = model,
        args         = args,
        device       = device,
        member_mask  = member_mask
    )
    print(f"[MIA-Unlearned]  AUC={auc_unlearned:.4f}, F1={f1_unlearned:.4f}")
    # ——————————————————————————————





    return model


def main():
    args=get_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(args)

    # 1) 数据载入 & 预处理
    X_tr, y_tr, df_tr, trans = preprocess_node_features_bank(
        args.train_csv, is_test=False
    )
    print(f"训练集: {X_tr.shape[0]} nodes, dim={X_tr.shape[1]}")

    # 2) MIA 全量评估
    cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
    cont_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    hypers_full = generate_hyperedge_dict_bank(
        df_tr, cat_cols, cont_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        device=device
    )
    auc_m, f1_m = mia_with_hgnn(X_tr, np.asarray(y_tr), hypers_full, args, device)
    print(f"[MIA-Full] AUC={auc_m:.4f}, F1={f1_m:.4f}")

    # 3) 随机采样 deleted_idx
    n = X_tr.shape[0]
    k = int(args.remove_ratio * n)
    deleted_idx = np.random.choice(n, k, replace=False)
    print(f"删除 {k} nodes (~{args.remove_ratio * 100:.0f}%): {deleted_idx[:5]}...")

    # 4) retrain_on_pruned
    retrain_on_pruned(X_tr, y_tr, df_tr, trans, deleted_idx, args, device)

    # 5) full_train_and_unlearn
    full_train_and_unlearn(X_tr, y_tr, df_tr, trans, deleted_idx, args, device)


if __name__ == "__main__":
    main()