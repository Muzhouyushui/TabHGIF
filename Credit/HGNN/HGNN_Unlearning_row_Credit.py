#!/usr/bin/env python
# coding: utf-8
"""
train_unlearning_hgnn_refactored.py

Refactored version: separates retraining on a pruned dataset and GIF-based unlearning
into two clear functions, using the same deleted node indices for both parts.
"""
import pandas as pd

from collections import Counter
import argparse                      # MOD: 增加 argparse 支持命令行参数

from sklearn.model_selection import train_test_split

from torch_geometric.data import Data

import torch.nn as nn
import torch.optim as optim
import time
import torch.nn.functional as F

from Credit.HGNN.HGNN_utils import evaluate_test_acc, evaluate_test_f1
from Credit.HGNN.data_preprocessing_Credit import (
    preprocess_node_features,
    generate_hyperedge_dict,
)
from Credit.HGNN.HGNN import HGNN_implicit, build_incidence_matrix, compute_degree_vectors
from Credit.HGNN.GIF_HGNN_ROW_Credit import (
    rebuild_structure_after_node_deletion,
    approx_gif,
    train_model,
)

import numpy as np
import torch
from MIA_HGNN        import train_shadow_model, membership_inference
from paths import CREDIT_DATA

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
def mia_with_hgnn(X_train, y_train, hyperedges, args, device):
    """
    1) 构造 Full model 用的数据结构 Data_full
    2) 训练 Full model
    3) 调用 membership_inference(...)，传入已经训练好的 full_model
    4) 返回在 Full model 上的 (AUC, F1)
    """
    N = X_train.shape[0]

    # 1) 构造超图及 Degree inv
    H_sp        = build_incidence_matrix(hyperedges, N)
    dv_inv, de_inv = compute_degree_vectors(H_sp)
    Hc = H_sp.tocoo()
    idx = torch.LongTensor([Hc.row, Hc.col])
    val = torch.FloatTensor(Hc.data)
    H_tensor = torch.sparse_coo_tensor(idx, val, Hc.shape).to(device)

    # 2) 准备 Data_full，全部节点都作为训练
    full_train_mask = torch.ones(N, dtype=torch.bool, device=device)
    data_full = Data(
        x = torch.from_numpy(X_train).float().to(device),
        y = torch.from_numpy(y_train).long().to(device),
        H         = H_tensor,
        dv_inv    = torch.from_numpy(dv_inv).to(device),
        de_inv    = torch.from_numpy(de_inv).to(device),
        train_mask    = full_train_mask,
        train_indices = list(range(N)),
        test_indices  = []               # 对 Full model 训练不需要 test_indices
    )

    # 3) 初始化并训练 Full model
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

    # 4) 用 membership_inference 在 Full model 上算 AUC/F1
    #    返回值格式： (atk_model, (auc_shadow,f1_shadow), (auc_target,f1_target))
    _, (_, _), (auc_target, f1_target) = membership_inference(
        X_train = X_train,
        y_train = y_train,
        hyperedges   = hyperedges,
        target_model = full_model,
        args         = args,
        device       = device
    )
    return auc_target, f1_target

    # # ——— 列名定义 ———
    # col_names = [
    #     "age", "job", "marital", "education", "default",
    #     "balance", "housing", "loan", "contact", "day",
    #     "month", "duration", "campaign", "pdays",
    #     "previous", "poutcome", "y"
    # ]

def retrain_on_pruned(df_full, transformer, deleted_idx, args,device):
    """
    Prunes training data by removing nodes at deleted_idx,
    retrains a HGNN model from scratch, and evaluates on the test set.
    所有超参都已写死，不再依赖 args。
    """
    # —— 0) 拆分 train / test（硬编码 30% 测试集） —— #
    df_tr, df_te = train_test_split(
        df_full, test_size=0.2, stratify=df_full["y"], random_state=42
    )
    df_tr = df_tr.reset_index(drop=True)
    df_te = df_te.reset_index(drop=True)

    # —— 1) 预处理训练集 —— #
    X_tr, y_tr, df_tr_proc, transformer = preprocess_node_features(df_tr, transformer)
    num_train = X_tr.shape[0]

    # —— 2) 剪裁训练集 —— #
    keep_mask = np.ones(num_train, dtype=bool)
    keep_mask[deleted_idx] = False

    X_pruned = X_tr[keep_mask]
    y_pruned = y_tr[keep_mask]
    df_pruned = df_tr_proc.loc[keep_mask].reset_index(drop=True)
    print(f"[重训练] 剪裁后训练节点数: {X_pruned.shape[0]} / 原始 {num_train}")

    # —— 3) 构造超图（Credit Approval 固定列） —— #
    cont_cols = ["A2","A3","A8","A11","A14","A15"]
    cat_cols  = [f"A{i}" for i in range(1,16) if f"A{i}" not in cont_cols]
    t0 = time.time()
    hyper_pruned = generate_hyperedge_dict(
        df_pruned,
        feature_cols=cat_cols+cont_cols,
        max_nodes_per_hyperedge=50,
        device=device
    )
    t_hg = time.time() - t0
    print(f"[重训练] 超图重建时间: {t_hg:.2f}s, 超边总数: {len(hyper_pruned)}")

    # —— 4) 转为 HGNN 输入 —— #
    H_sp = build_incidence_matrix(hyper_pruned, X_pruned.shape[0])
    dv_np, de_np = compute_degree_vectors(H_sp)
    H_coo = H_sp.tocoo()
    idx = torch.LongTensor(np.vstack((H_coo.row, H_coo.col))).to(device)
    val = torch.FloatTensor(H_coo.data).to(device)
    H_pruned = torch.sparse_coo_tensor(idx, val, H_coo.shape).coalesce()
    dv = torch.FloatTensor(dv_np).to(device)
    de = torch.FloatTensor(de_np).to(device)
    fts = torch.FloatTensor(X_pruned).to(device)
    lbls = torch.LongTensor(y_pruned).to(device)

    # —— 5) 初始化 & 训练 HGNN —— #
    t_start = time.time()
    t_init_start = time.time()
    model = HGNN_implicit(
        in_ch=fts.shape[1],
        n_class=int(lbls.max().item()) + 1,
        n_hid=128,       # 隐藏维度
        dropout=0.1      # Dropout 比例
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[100, 150],
        gamma=0.1
    )
    criterion = FocalLoss(gamma=2.0, reduction='mean')
    t_init_end = time.time()

    t_train_start = time.time()
    model = train_model(
        model, criterion, optimizer, scheduler,
        fts, lbls, H_pruned, dv, de,
        num_epochs=200, print_freq=10
    )
    t_train_end = time.time()
    t_total = time.time() - t_start

    print(
        f"[重训练] 初始化: {t_init_end-t_init_start:.2f}s | "
        f"训练: {t_train_end-t_train_start:.2f}s | 总计: {t_total:.2f}s"
    )

    # —— 6) 测试集评估 —— #
    X_te, y_te, df_te_proc, _ = preprocess_node_features(df_te, transformer)
    t1 = time.time()
    hyper_te = generate_hyperedge_dict(
        df_te_proc,
        feature_cols=cat_cols+cont_cols,
        max_nodes_per_hyperedge=50,
        device=device
    )
    t_hg_te = time.time() - t1
    print(f"[重训练] 测试集超图重建时间: {t_hg_te:.2f}s, 超边总数: {len(hyper_te)}")
    H_sp_te = build_incidence_matrix(hyper_te, X_te.shape[0])
    dv_te, de_te = compute_degree_vectors(H_sp_te)
    H_coo_te = H_sp_te.tocoo()
    idx_te = torch.LongTensor(np.vstack((H_coo_te.row, H_coo_te.col))).to(device)
    val_te = torch.FloatTensor(H_coo_te.data).to(device)
    H_test = torch.sparse_coo_tensor(idx_te, val_te, H_coo_te.shape).coalesce()
    dv_test = torch.FloatTensor(dv_te).to(device)
    de_test = torch.FloatTensor(de_te).to(device)
    fts_test = torch.FloatTensor(X_te).to(device)
    lbls_test = torch.LongTensor(y_te).to(device)

    test_obj = {"x": fts_test, "y": lbls_test, "H": H_test, "dv_inv": dv_test, "de_inv": de_test}
    f1_test = evaluate_test_f1(model, test_obj)
    acc_test = evaluate_test_acc(model, test_obj)
    print(f"[重训练] 测试集 F1: {f1_test:.4f}, Acc: {acc_test:.4f}")

    # —— 7) 删除节点评估 —— #
    X_del = X_tr[deleted_idx]
    y_del = y_tr[deleted_idx]
    df_del = df_tr_proc.iloc[deleted_idx].reset_index(drop=True)

    hyper_del = generate_hyperedge_dict(
        df_del,
        feature_cols=cat_cols+cont_cols,
        max_nodes_per_hyperedge=50,
        device=device
    )
    H_sp_del = build_incidence_matrix(hyper_del, len(deleted_idx))
    dv_np_del, de_np_del = compute_degree_vectors(H_sp_del)
    H_coo_del = H_sp_del.tocoo()
    idx_del = torch.LongTensor(np.vstack((H_coo_del.row, H_coo_del.col))).to(device)
    val_del = torch.FloatTensor(H_coo_del.data).to(device)
    H_del = torch.sparse_coo_tensor(idx_del, val_del, H_coo_del.shape).coalesce()
    fts_del = torch.FloatTensor(X_del).to(device)
    lbls_del = torch.LongTensor(y_del).to(device)
    dv_del = torch.FloatTensor(dv_np_del).to(device)
    de_del = torch.FloatTensor(de_np_del).to(device)
    test_obj_del = {"x": fts_del, "y": lbls_del, "H": H_del, "dv_inv": dv_del, "de_inv": de_del}

    num_hyp_del = len(hyper_del)
    has_feats = (fts_del.abs().sum().item() > 0)
    print(f"删除集超边数 {num_hyp_del}, 删除集节点特征没被置零？? {has_feats}")

    f1_del = evaluate_test_f1(model, test_obj_del)
    acc_del = evaluate_test_acc(model, test_obj_del)
    print(f"[删除节点评估] 重训练模型 在被删除节点上的分数: F1={f1_del:.4f}, Acc={acc_del:.4f}")

    # —— 8) MIA on retrained model —— #
    keep_idx = np.where(keep_mask)[0]
    all_he = {}
    cnt = 0
    for he in hyper_pruned.values():
        all_he[cnt] = he; cnt += 1
    offset = len(keep_idx)
    for he in hyper_del.values():
        all_he[cnt] = [n + offset for n in he]; cnt += 1

    X_attack = np.vstack([X_pruned, X_del])
    y_attack = np.hstack([y_pruned, y_del])
    member_mask = np.concatenate([np.ones(len(keep_idx),bool), np.zeros(len(deleted_idx),bool)])

    _, _, (auc_rt, f1_rt) = membership_inference(
        X_train=X_attack, y_train=y_attack,
        hyperedges=all_he, target_model=model,args=args,
        device=device, member_mask=member_mask
    )
    print(f"[MIA-Retrain Keep/Del] AUC={auc_rt:.4f}, F1={f1_rt:.4f}")

    return model
# def full_train_and_unlearn(X, y, df, transformer, deleted_idx, args, device):
# def full_train_and_unlearn(X_tr, y_tr, df_tr, df_te, transformer, deleted_idx, args, device):
#     # 定义连续和类别特征列
#     cont_cols = args.cont_cols
#     cat_cols  = args.cat_cols
#
#     # —— 1) 全量训练 —— #
#     # 构造超图
#     hyper_full = generate_hyperedge_dict(
#         df_tr,
#         feature_cols=cat_cols + cont_cols,
#         max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
#         device=device
#     )
#     H_sp = build_incidence_matrix(hyper_full, X_tr.shape[0])
#     dv_np, de_np = compute_degree_vectors(H_sp)
#     H_coo = H_sp.tocoo()
#     idx = torch.LongTensor(np.vstack((H_coo.row, H_coo.col))).to(device)
#     val = torch.FloatTensor(H_coo.data).to(device)
#     H = torch.sparse_coo_tensor(idx, val, H_coo.shape).coalesce()
#
#     # 准备张量
#     fts   = torch.FloatTensor(X_tr).to(device)
#     lbls  = torch.LongTensor(y_tr).to(device)
#     dv    = torch.FloatTensor(dv_np).to(device)
#     de    = torch.FloatTensor(de_np).to(device)
#
#     # 初始化并训练模型
#     model = HGNN_implicit(
#         in_ch=fts.shape[1],
#         n_class=len(np.unique(y_tr)),
#         n_hid=args.hidden_dim,
#         dropout=args.dropout
#     ).to(device)
#     optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
#     scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
#     criterion = FocalLoss(gamma=2.0, reduction='mean')
#
#     model = train_model(
#         model, criterion, optimizer, scheduler,
#         fts, lbls, H, dv, de,
#         num_epochs=args.epochs, print_freq=10
#     )
#
#     # —— 2) 测试集评估（使用 df_te） —— #
#     X_te, y_te, df_te_proc, _ = preprocess_node_features(df_te,  transformer=transformer)
#     hyper_te = generate_hyperedge_dict(
#         df_te_proc,
#         feature_cols=cat_cols + cont_cols,
#         max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
#         device=device
#     )
#     H_sp_te = build_incidence_matrix(hyper_te, X_te.shape[0])
#     dv_te, de_te = compute_degree_vectors(H_sp_te)
#     H_coo_te = H_sp_te.tocoo()
#     idx_te = torch.LongTensor(np.vstack((H_coo_te.row, H_coo_te.col))).to(device)
#     val_te = torch.FloatTensor(H_coo_te.data).to(device)
#     H_test = torch.sparse_coo_tensor(idx_te, val_te, H_coo_te.shape).coalesce()
#
#     fts_test  = torch.FloatTensor(X_te).to(device)
#     lbls_test = torch.LongTensor(y_te).to(device)
#     dv_test   = torch.FloatTensor(dv_te).to(device)
#     de_test   = torch.FloatTensor(de_te).to(device)
#     test_obj  = {
#         "x": fts_test,
#         "y": lbls_test,
#         "H": H_test,
#         "dv_inv": dv_test,
#         "de_inv": de_test
#     }
#
#     f1_before = evaluate_test_f1(model, test_obj)
#     acc_before = evaluate_test_acc(model, test_obj)
#     print(f"Before Unlearning — F1={f1_before:.4f}, Acc={acc_before:.4f}")
#
#     # —— 3) GIF-based Unlearning —— #
#     data_obj = {
#         "x": fts,
#         "y": lbls,
#         "H": H,
#         "dv_inv": dv,
#         "de_inv": de,
#         "train_mask": torch.ones_like(lbls, dtype=torch.bool)
#     }
#     unlearning_time, _ = approx_gif(
#         model,
#         data_obj,
#         (deleted_idx, hyper_full, 13),
#         iteration=args.gif_iters,
#         damp=args.gif_damp,
#         scale=args.gif_scale
#     )
#     print(f"[Unlearn] Time: {unlearning_time:.4f}s")
#
#     f1_after = evaluate_test_f1(model, test_obj)
#     acc_after = evaluate_test_acc(model, test_obj)
#     print(f"After Unlearning — F1={f1_after:.4f}, Acc={acc_after:.4f}")
#
#     return model
def full_train_and_unlearn(X_tr, y_tr, df_tr, df_te, transformer, deleted_idx, args, device):
    # 定义连续和类别特征列
    cont_cols = args.cont_cols
    cat_cols  = args.cat_cols

    # —— 1) 全量训练 —— #
    hyper_full = generate_hyperedge_dict(
        df_tr,
        feature_cols=cat_cols + cont_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        device=device
    )
    H_sp = build_incidence_matrix(hyper_full, X_tr.shape[0])
    dv_np, de_np = compute_degree_vectors(H_sp)
    H_coo = H_sp.tocoo()
    idx = torch.LongTensor(np.vstack((H_coo.row, H_coo.col))).to(device)
    val = torch.FloatTensor(H_coo.data).to(device)
    H = torch.sparse_coo_tensor(idx, val, H_coo.shape).coalesce()

    fts  = torch.FloatTensor(X_tr).to(device)
    lbls = torch.LongTensor(y_tr).to(device)
    dv   = torch.FloatTensor(dv_np).to(device)
    de   = torch.FloatTensor(de_np).to(device)

    model = HGNN_implicit(
        in_ch=fts.shape[1],
        n_class=len(np.unique(y_tr)),
        n_hid=args.hidden_dim,
        dropout=args.dropout
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    criterion = FocalLoss(gamma=2.0, reduction='mean')

    model = train_model(
        model, criterion, optimizer, scheduler,
        fts, lbls, H, dv, de,
        num_epochs=args.epochs, print_freq=10
    )

    # —— 2) 测试集评估 —— #
    X_te, y_te, df_te_proc, _ = preprocess_node_features(df_te,  transformer=transformer)
    hyper_te = generate_hyperedge_dict(
        df_te_proc,
        feature_cols=cat_cols + cont_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        device=device
    )
    H_sp_te = build_incidence_matrix(hyper_te, X_te.shape[0])
    dv_te, de_te = compute_degree_vectors(H_sp_te)
    H_coo_te = H_sp_te.tocoo()
    idx_te = torch.LongTensor(np.vstack((H_coo_te.row, H_coo_te.col))).to(device)
    val_te = torch.FloatTensor(H_coo_te.data).to(device)
    H_test = torch.sparse_coo_tensor(idx_te, val_te, H_coo_te.shape).coalesce()

    fts_test  = torch.FloatTensor(X_te).to(device)
    lbls_test = torch.LongTensor(y_te).to(device)
    dv_test   = torch.FloatTensor(dv_te).to(device)
    de_test   = torch.FloatTensor(de_te).to(device)
    test_obj  = {"x": fts_test, "y": lbls_test, "H": H_test, "dv_inv": dv_test, "de_inv": de_test}

    f1_before = evaluate_test_f1(model, test_obj)
    acc_before = evaluate_test_acc(model, test_obj)
    print(f"Before Unlearning — F1={f1_before:.4f}, Acc={acc_before:.4f}")

    # —— 3) GIF-based Unlearning —— #
    data_obj = {
        "x": fts,
        "y": lbls,
        "H": H,
        "dv_inv": dv,
        "de_inv": de,
        "train_mask": torch.ones_like(lbls, dtype=torch.bool)
    }
    unlearning_time, _ = approx_gif(
        model,
        data_obj,
        (deleted_idx, hyper_full, 13),
        iteration=args.gif_iters,
        damp=args.gif_damp,
        scale=args.gif_scale
    )
    print(f"[Unlearn] Time: {unlearning_time:.4f}s")

    f1_after = evaluate_test_f1(model, test_obj)
    acc_after = evaluate_test_acc(model, test_obj)
    print(f"After Unlearning — F1={f1_after:.4f}, Acc={acc_after:.4f}")

    # —— 4) MIA on Unlearned Model —— #
    # 构造攻击用的超边字典（keep + del，del 索引需偏移）
    N_keep = X_tr.shape[0] - len(deleted_idx)
    df_keep = df_tr.drop(index=deleted_idx).reset_index(drop=True)
    df_del  = df_tr.iloc[deleted_idx].reset_index(drop=True)

    he_keep = generate_hyperedge_dict(
        df_keep, feature_cols=cat_cols + cont_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge, device=device
    )
    he_del = generate_hyperedge_dict(
        df_del, feature_cols=cat_cols + cont_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge, device=device
    )
    he_attack = {}
    idx_he = 0
    for nodes in he_keep.values():
        he_attack[idx_he] = nodes
        idx_he += 1
    for nodes in he_del.values():
        he_attack[idx_he] = [n + N_keep for n in nodes]
        idx_he += 1

    X_attack = np.vstack([
        X_tr[~np.isin(np.arange(len(X_tr)), deleted_idx)],
        X_tr[deleted_idx]
    ])
    y_attack = np.hstack([
        y_tr[~np.isin(np.arange(len(y_tr)), deleted_idx)],
        y_tr[deleted_idx]
    ])
    member_mask = np.concatenate([
        np.ones(N_keep, dtype=bool),
        np.zeros(len(deleted_idx), dtype=bool)
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
    print(f"[MIA-Unlearned] AUC={auc_un:.4f}, F1={f1_un:.4f}")

    return model
def main():
    parser = argparse.ArgumentParser(description="HGNN on Credit Approval 数据集")
    # parser.add_argument("--data-csv", type=str,
    #                     help="Credit Approval data CSV file path")
    parser.add_argument("--data-csv", type=str,
                        default=CREDIT_DATA,
                        help="Credit Approval data CSV file path")
    parser.add_argument("--cat-cols", nargs="+", type=str,
                        default=['A1','A4','A5','A6','A7','A9','A10','A12','A13'],
                        help="List of categorical feature column names")
    parser.add_argument("--cont-cols", nargs="+", type=str,
                        default=['A2','A3','A8','A11','A14','A15'],
                        help="List of continuous feature column names")
    parser.add_argument("--split-ratio", type=float, default=0.2,
                        help="测试集比例")
    parser.add_argument("--max-nodes-per-hyperedge", type=int, default=50,
                        help="超边最大节点数")
    parser.add_argument("--hidden-dim", type=int, default=128, help="HGNN 隐藏维度")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout 比例")
    parser.add_argument("--lr", type=float, default=0.01, help="学习率")
    parser.add_argument("--epochs", type=int, default=200, help="训练轮数")
    parser.add_argument("--milestones", nargs='+', type=int, default=[100,150], help="学习率里程碑")
    parser.add_argument("--gamma", type=float, default=0.1, help="学习率衰减系数")
    parser.add_argument("--gif-iters", type=int, default=20, help="GIF 迭代次数")
    parser.add_argument("--gif-damp", type=float, default=0.01, help="GIF 阻尼")
    parser.add_argument("--gif-scale", type=float, default=1e7, help="GIF scale")
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {device}")

    # 读取并标记标签
    df_full = pd.read_csv(
        args.data_csv,
        header=None, na_values='?', skipinitialspace=True
    )
    df_full.columns = [
        "A1","A2","A3","A4","A5","A6","A7","A8","A9",
        "A10","A11","A12","A13","A14","A15","class"
    ]
    df_full["y"] = df_full["class"].map({"+": 1, "-": 0})

    # 划分训练/测试
    df_tr, df_te = train_test_split(
        df_full,
        test_size=args.split_ratio,
        stratify=df_full["y"],
        random_state=21
    )
    df_tr.reset_index(drop=True, inplace=True)
    df_te.reset_index(drop=True, inplace=True)
    print(f"训练集: {len(df_tr)} 样本, 测试集: {len(df_te)} 样本")

    # 训练集预处理
    X_tr, y_tr, df_tr_proc, transformer = preprocess_node_features(df_tr)

    # 随机选 20% 节点作为删除集
    num_to_delete = int(0.2 * X_tr.shape[0])
    deleted_idx = np.random.choice(X_tr.shape[0], size=num_to_delete, replace=False)

    # 1) 重训练评估
    retrained_model = retrain_on_pruned(
        df_full=df_full,
        transformer=transformer,
        deleted_idx=deleted_idx,
        args=args,
        device=device
    )

    # 2) 全量训练 + GIF Unlearning
    print("\n=== Full training + GIF Unlearning ===")
    _ = full_train_and_unlearn(
        X_tr        = X_tr,
        y_tr        = y_tr,
        df_tr       = df_tr,
        df_te       = df_te,
        transformer = transformer,
        deleted_idx = deleted_idx,
        args        = args,
        device      = device
    )
if __name__ == "__main__":
    # main()
    for run in range(1,5):
        print("=== Run",run,"===")
        main()