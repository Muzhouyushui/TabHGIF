#!/usr/bin/env python
# coding: utf-8
"""
HGNN_Unlearning_row_Credit_Value.py

HGNN + Credit Approval 数据集：Value Unlearning（按列取值删除一组样本）

对齐的参考逻辑：
- Value 删除索引：mask -> np.where(mask)[0]
- 两段式流程：retrain on pruned + full train + GIF unlearning
- MIA：keep/del 拼接（del 节点 index 需要 offset）+ member_mask
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

from MIA_HGNN import train_shadow_model, membership_inference
from torch_geometric.data import Data
from paths import CREDIT_DATA

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
    1) 构造 Full model 的 Data_full
    2) 训练 full_model
    3) 调用 membership_inference(...) 返回 target AUC/F1
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
        test_indices=[]
    )

    full_model = HGNN_implicit(
        in_ch=X_train.shape[1],
        n_class=int(y_train.max()) + 1,
        n_hid=args.full_hidden,
        dropout=args.full_dropout
    ).to(device)

    full_model = train_shadow_model(
        full_model, data_full,
        lr=args.full_lr,
        epochs=args.full_epochs
    )

    _, (_, _), (auc_target, f1_target) = membership_inference(
        X_train=X_train,
        y_train=y_train,
        hyperedges=hyperedges,
        target_model=full_model,
        args=args,
        device=device
    )
    return auc_target, f1_target

def retrain_on_pruned(X_tr, y_tr, df_tr_proc, df_tr_raw, transformer, deleted_idx, args, device, df_te_raw):
    """
    删除 deleted_idx 后，从头重训一个 HGNN，并：
      - 测试集评估
      - 删除集评估
      - keep/del MIA（对齐你 HGNN 的拼接逻辑）
    """
    num_train = X_tr.shape[0]
    keep_mask = np.ones(num_train, dtype=bool)
    keep_mask[deleted_idx] = False

    X_pruned = X_tr[keep_mask]
    y_pruned = y_tr[keep_mask]
    df_pruned = df_tr_proc.loc[keep_mask].reset_index(drop=True)

    print(f"[重训练] 剪裁后训练节点数: {X_pruned.shape[0]} / 原始 {num_train}")

    cont_cols = args.cont_cols
    cat_cols = args.cat_cols

    t_start = time.time()

    # 1) 重建超图
    t0 = time.time()
    hyper_pruned = generate_hyperedge_dict(
        df_pruned,
        feature_cols=cat_cols + cont_cols,
        max_nodes_per_hyperedge=getattr(args, "max_nodes_per_hyperedge_train", args.max_nodes_per_hyperedge),
        device=device
    )
    t_hg = time.time() - t0
    print(f"[重训练] 超图重建时间: {t_hg:.2f}s, 超边总数: {len(hyper_pruned)}")

    # 2) HGNN 输入
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

    # 3) 训练
    t_init_start = time.time()
    model = HGNN_implicit(
        in_ch=fts.shape[1],
        n_class=int(lbls.max().item()) + 1,
        n_hid=args.hidden_dim,
        dropout=args.dropout
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    criterion = FocalLoss(gamma=2.0, reduction="mean")
    t_init_end = time.time()

    t_train_start = time.time()
    model = train_model(
        model, criterion, optimizer, scheduler,
        fts, lbls, H_pruned, dv, de,
        num_epochs=args.epochs, print_freq=args.log_every
    )
    t_train_end = time.time()

    t_total = time.time() - t_start
    print(
        f"[重训练] 初始化: {t_init_end - t_init_start:.2f}s | "
        f"训练: {t_train_end - t_train_start:.2f}s | 总计: {t_total:.2f}s"
    )

    # 4) 测试集评估
    X_te, y_te, df_te_proc, _ = preprocess_node_features(df_te_raw, transformer)
    t1 = time.time()
    hyper_te = generate_hyperedge_dict(
        df_te_proc,
        feature_cols=cat_cols + cont_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        device=device
    )
    print(f"[重训练] 测试集超图重建时间: {time.time() - t1:.2f}s, 超边总数: {len(hyper_te)}")

    H_sp_te = build_incidence_matrix(hyper_te, X_te.shape[0])
    dv_te, de_te = compute_degree_vectors(H_sp_te)

    H_coo_te = H_sp_te.tocoo()
    idx_te = torch.LongTensor(np.vstack((H_coo_te.row, H_coo_te.col))).to(device)
    val_te = torch.FloatTensor(H_coo_te.data).to(device)
    H_test = torch.sparse_coo_tensor(idx_te, val_te, H_coo_te.shape).coalesce()

    test_obj = {
        "x": torch.FloatTensor(X_te).to(device),
        "y": torch.LongTensor(y_te).to(device),
        "H": H_test,
        "dv_inv": torch.FloatTensor(dv_te).to(device),
        "de_inv": torch.FloatTensor(de_te).to(device),
    }
    f1_test = evaluate_test_f1(model, test_obj)
    acc_test = evaluate_test_acc(model, test_obj)
    print(f"[重训练] 测试集 F1: {f1_test:.4f}, Acc: {acc_test:.4f}")

    # 5) 删除节点评估（重训练模型在 deleted nodes 上的表现）
    X_del = X_tr[deleted_idx]
    y_del = y_tr[deleted_idx]
    df_del = df_tr_proc.iloc[deleted_idx].reset_index(drop=True)

    hyper_del = generate_hyperedge_dict(
        df_del,
        feature_cols=cat_cols + cont_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        device=device
    )
    H_sp_del = build_incidence_matrix(hyper_del, len(deleted_idx))
    dv_np_del, de_np_del = compute_degree_vectors(H_sp_del)

    H_coo_del = H_sp_del.tocoo()
    idx_del = torch.LongTensor(np.vstack((H_coo_del.row, H_coo_del.col))).to(device)
    val_del = torch.FloatTensor(H_coo_del.data).to(device)
    H_del = torch.sparse_coo_tensor(idx_del, val_del, H_coo_del.shape).coalesce()

    test_obj_del = {
        "x": torch.FloatTensor(X_del).to(device),
        "y": torch.LongTensor(y_del).to(device),
        "H": H_del,
        "dv_inv": torch.FloatTensor(dv_np_del).to(device),
        "de_inv": torch.FloatTensor(de_np_del).to(device),
    }

    has_nonzero_features = (test_obj_del["x"].abs().sum().item() > 0)
    print(f"删除集超边数 {len(hyper_del)}, 删除集节点特征没被置零？？ {has_nonzero_features}")

    f1_del = evaluate_test_f1(model, test_obj_del)
    acc_del = evaluate_test_acc(model, test_obj_del)
    print(f"[删除节点评估] 重训练模型 在被删除节点上的分数: F1={f1_del:.4f}, Acc={acc_del:.4f}")

    # 6) MIA on retrained model (keep vs del)
    all_idx = np.arange(num_train)
    keep_idx = np.setdiff1d(all_idx, deleted_idx)
    X_keep, y_keep = X_tr[keep_idx], y_tr[keep_idx]

    he_keep = hyper_pruned  # dict
    he_del = hyper_del      # dict（已在上面建好）

    # 合并超边（del 需要 offset）：完全对齐你 HGNN 的套路
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
        np.zeros(len(X_del), dtype=bool),
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

    return model

def full_train_and_unlearn(X_tr, y_tr, df_tr_proc, df_te_raw, transformer, deleted_idx, args, device):
    """
    全量训练 + GIF unlearning + (测试集/删除集)评估 + keep/del MIA
    """
    cont_cols = args.cont_cols
    cat_cols = args.cat_cols

    # 1) 构造全量超图
    hyper_full = generate_hyperedge_dict(
        df_tr_proc,
        feature_cols=cat_cols + cont_cols,
        max_nodes_per_hyperedge=getattr(args, "max_nodes_per_hyperedge_train", args.max_nodes_per_hyperedge),
        device=device
    )

    H_sp = build_incidence_matrix(hyper_full, X_tr.shape[0])
    dv_np, de_np = compute_degree_vectors(H_sp)

    H_coo = H_sp.tocoo()
    idx = torch.LongTensor(np.vstack((H_coo.row, H_coo.col))).to(device)
    val = torch.FloatTensor(H_coo.data).to(device)
    H = torch.sparse_coo_tensor(idx, val, H_coo.shape).coalesce()

    fts = torch.FloatTensor(X_tr).to(device)
    lbls = torch.LongTensor(y_tr).to(device)
    dv = torch.FloatTensor(dv_np).to(device)
    de = torch.FloatTensor(de_np).to(device)

    train_mask = torch.ones(X_tr.shape[0], dtype=torch.bool, device=device)

    # 2) 训练 full model
    model = HGNN_implicit(
        in_ch=fts.shape[1],
        n_class=len(np.unique(y_tr)),
        n_hid=args.hidden_dim,
        dropout=args.dropout
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    criterion = FocalLoss(gamma=2.0, reduction="mean")

    # 给 GIF 挂 reason_once / reason_once_unlearn（和你 bank/HGNN 风格一致）
    def reason_once(data):
        return model(data["x"], data["H"], data["dv_inv"], data["de_inv"])

    def reason_once_unlearn(data):
        # 删除节点：特征置零 + 重建结构（与你 HGNN 的 GIF 实现接口保持一致）
        x_mod = data["x"].clone()
        x_mod[deleted_idx] = 0.0
        H_new, dv_new, de_new, _ = rebuild_structure_after_node_deletion(
            hyper_full, deleted_idx, X_tr.shape[0], device
        )
        return model(x_mod, H_new, dv_new, de_new)

    model.reason_once = reason_once
    model.reason_once_unlearn = reason_once_unlearn

    model = train_model(
        model, criterion, optimizer, scheduler,
        fts, lbls, H, dv, de,
        num_epochs=args.epochs, print_freq=args.log_every
    )

    # 3) 测试集评估（Before）
    X_te, y_te, df_te_proc, _ = preprocess_node_features(df_te_raw, transformer)
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

    test_obj = {
        "x": torch.FloatTensor(X_te).to(device),
        "y": torch.LongTensor(y_te).to(device),
        "H": H_test,
        "dv_inv": torch.FloatTensor(dv_te).to(device),
        "de_inv": torch.FloatTensor(de_te).to(device),
    }

    f1_before = evaluate_test_f1(model, test_obj)
    acc_before = evaluate_test_acc(model, test_obj)
    print(f"Before Unlearning — F1={f1_before:.4f}, Acc={acc_before:.4f}")

    # 4) GIF unlearning
    data_obj = {
        "x": fts,
        "y": lbls,
        "H": H,
        "dv_inv": dv,
        "de_inv": de,
        "train_mask": train_mask
    }

    unlearning_time, _ = approx_gif(
        model,
        data_obj,
        (deleted_idx, hyper_full, getattr(args, "neighbor_k", 13)),
        iteration=args.gif_iters,
        damp=args.gif_damp,
        scale=args.gif_scale
    )
    print(f"[Unlearn] Time: {unlearning_time:.4f}s")

    # 5) 测试集评估（After）
    f1_after = evaluate_test_f1(model, test_obj)
    acc_after = evaluate_test_acc(model, test_obj)
    print(f"After Unlearning — F1={f1_after:.4f}, Acc={acc_after:.4f}")

    # 6) 删除集评估（After）
    X_del = X_tr[deleted_idx]
    y_del = y_tr[deleted_idx]
    df_del = df_tr_proc.iloc[deleted_idx].reset_index(drop=True)

    hyper_del = generate_hyperedge_dict(
        df_del,
        feature_cols=cat_cols + cont_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        device=device
    )
    H_sp_del = build_incidence_matrix(hyper_del, len(deleted_idx))
    dv_np_del, de_np_del = compute_degree_vectors(H_sp_del)

    H_coo_del = H_sp_del.tocoo()
    idx_del = torch.LongTensor(np.vstack((H_coo_del.row, H_coo_del.col))).to(device)
    val_del = torch.FloatTensor(H_coo_del.data).to(device)
    H_del = torch.sparse_coo_tensor(idx_del, val_del, H_coo_del.shape).coalesce()

    del_obj = {
        "x": torch.FloatTensor(X_del).to(device),
        "y": torch.LongTensor(y_del).to(device),
        "H": H_del,
        "dv_inv": torch.FloatTensor(dv_np_del).to(device),
        "de_inv": torch.FloatTensor(de_np_del).to(device),
    }
    f1_del = evaluate_test_f1(model, del_obj)
    acc_del = evaluate_test_acc(model, del_obj)
    print(f"[删除节点评估] Unlearned 模型 在被删除节点上的分数: F1={f1_del:.4f}, Acc={acc_del:.4f}")

    # 7) MIA on Unlearned Model（keep vs del，与你 Credit row 的写法一致）
    N_keep = X_tr.shape[0] - len(deleted_idx)
    df_keep = df_tr_proc.drop(index=deleted_idx).reset_index(drop=True)
    df_del2 = df_tr_proc.iloc[deleted_idx].reset_index(drop=True)

    he_keep = generate_hyperedge_dict(
        df_keep,
        feature_cols=cat_cols + cont_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        device=device
    )
    he_del2 = generate_hyperedge_dict(
        df_del2,
        feature_cols=cat_cols + cont_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        device=device
    )

    he_attack = {}
    idx_he = 0
    for nodes in he_keep.values():
        he_attack[idx_he] = nodes
        idx_he += 1
    for nodes in he_del2.values():
        he_attack[idx_he] = [n + N_keep for n in nodes]
        idx_he += 1

    keep_mask = ~np.isin(np.arange(len(X_tr)), deleted_idx)
    X_attack = np.vstack([X_tr[keep_mask], X_tr[deleted_idx]])
    y_attack = np.hstack([y_tr[keep_mask], y_tr[deleted_idx]])
    member_mask = np.concatenate([
        np.ones(N_keep, dtype=bool),
        np.zeros(len(deleted_idx), dtype=bool)
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
    print(f"[MIA-Unlearned] AUC={auc_un:.4f}, F1={f1_un:.4f}")

    return model

def main():
    parser = argparse.ArgumentParser(description="HGNN on Credit Approval 数据集（Value Unlearning）")

    parser.add_argument("--data-csv", type=str,
                        default=CREDIT_DATA,
                        help="Credit Approval data file path (crx.data)")

    parser.add_argument("--split-ratio", type=float, default=0.2, help="测试集比例")
    parser.add_argument("--max-nodes-per-hyperedge", type=int, default=50, help="超边最大节点数")
    parser.add_argument("--max-nodes-per-hyperedge-train", type=int, default=50, help="训练超边最大节点数")

    parser.add_argument("--cat-cols", nargs="+", type=str,
                        default=['A1','A4','A5','A6','A7','A9','A10','A12','A13'])
    parser.add_argument("--cont-cols", nargs="+", type=str,
                        default=['A2','A3','A8','A11','A14','A15'])

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
    parser.add_argument("--neighbor-k", type=int, default=13, help="共享超边阈值 K")

    # Value unlearning 配置：默认给一个和你 HGCN value 示例相同风格的参数（A1=b）
    parser.add_argument("--unl-col", type=str, default="A1", help="用于 value 删除的列名，例如 A1")
    parser.add_argument("--unl-val", type=str, default="b",  help="用于 value 删除的具体取值，例如 b")

    # 可选：Full-model MIA（如果你需要）
    parser.add_argument("--do-full-mia", action="store_true")
    parser.add_argument("--full-hidden", type=int, default=128)
    parser.add_argument("--full-dropout", type=float, default=0.1)
    parser.add_argument("--full-lr", type=float, default=0.01)
    parser.add_argument("--full-epochs", type=int, default=100)

    # 多次运行
    parser.add_argument("--runs", type=int, default=4)

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # 1) 读取并标记标签（沿用你 Credit row 脚本的数据读取方式）
    df_full = pd.read_csv(args.data_csv, header=None, na_values="?", skipinitialspace=True)
    df_full.columns = [
        "A1","A2","A3","A4","A5","A6","A7","A8","A9",
        "A10","A11","A12","A13","A14","A15","class"
    ]
    df_full["y"] = df_full["class"].map({"+": 1, "-": 0})

    for run in range(1, args.runs + 1):
        print(f"\n==================== Run {run}/{args.runs} ====================")

        # 2) 划分 train/test
        df_tr_raw, df_te_raw = train_test_split(
            df_full,
            test_size=args.split_ratio,
            stratify=df_full["y"],
            random_state=21 + run
        )
        df_tr_raw = df_tr_raw.reset_index(drop=True)
        df_te_raw = df_te_raw.reset_index(drop=True)
        print(f"训练集: {len(df_tr_raw)} 样本, 测试集: {len(df_te_raw)} 样本")

        # 3) 训练集预处理
        X_tr, y_tr, df_tr_proc, transformer = preprocess_node_features(df_tr_raw)

        # 4) Value 删除索引：df_tr_raw 上按值筛选（mask -> deleted_idx）
        df_tr_raw = df_tr_raw.reset_index(drop=True)
        mask = (df_tr_raw[args.unl_col].astype(str) == str(args.unl_val))
        deleted_idx = np.where(mask)[0]
        if len(deleted_idx) == 0:
            raise ValueError(f"[Error] 训练集中找不到 {args.unl_col}={args.unl_val} 的样本，无法执行 Value Unlearning")

        pct = 100.0 * len(deleted_idx) / len(df_tr_raw)
        print(f"[Value Unlearning] 列={args.unl_col}, 值={args.unl_val} -> 删除 {len(deleted_idx)}/{len(df_tr_raw)} (≈{pct:.2f}%)")
        print(f"[Value Unlearning] deleted_idx[:5] = {deleted_idx[:5]}")

        # 标签分布（保持你脚本的输出风格）
        train_counter = Counter(y_tr)
        print(f"训练集标签分布 → 0: {train_counter[0]}，1: {train_counter[1]}")

        # 5) 可选：Full model MIA（使用全量超图）
        if args.do_full_mia:
            hyper_full_for_mia = generate_hyperedge_dict(
                df_tr_proc,
                feature_cols=args.cat_cols + args.cont_cols,
                max_nodes_per_hyperedge=args.max_nodes_per_hyperedge_train,
                device=device
            )
            auc_full, f1_full = mia_with_hgnn(X_tr, y_tr, hyper_full_for_mia, args, device)
            print(f"[MIA-Full] AUC={auc_full:.4f}, F1={f1_full:.4f}")

        # 6) Retrain baseline
        print("\n=== Retrain on pruned dataset (Value Unlearning) ===")
        _ = retrain_on_pruned(
            X_tr=X_tr,
            y_tr=y_tr,
            df_tr_proc=df_tr_proc,
            df_tr_raw=df_tr_raw,
            transformer=transformer,
            deleted_idx=deleted_idx,
            args=args,
            device=device,
            df_te_raw=df_te_raw
        )

        # 7) Full train + GIF unlearning
        print("\n=== Full training + GIF Unlearning (Value Unlearning) ===")
        _ = full_train_and_unlearn(
            X_tr=X_tr,
            y_tr=y_tr,
            df_tr_proc=df_tr_proc,
            df_te_raw=df_te_raw,
            transformer=transformer,
            deleted_idx=deleted_idx,
            args=args,
            device=device
        )

if __name__ == "__main__":
    main()