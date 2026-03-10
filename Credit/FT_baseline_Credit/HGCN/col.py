#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_ft_hgcn_col_credit.py

Credit 数据集（单文件 crx.data）上的 HGCN 列级删除（Column Unlearning）FT 基线：
- Full@EditedHG
- FT-K@EditedHG
- FT-head@EditedHG

特点：
1) 仿照 Credit 的 HGCN 列级流程（单文件读入 -> train/test split -> preprocess -> hypergraph）
2) 在 Edited Hypergraph 上做 practical fine-tuning baseline
3) 默认数据路径：Credit/credit_data/crx.data (通过 paths.py 配置)
"""

import os
import time
import copy
import csv
import random
import argparse
import warnings
from collections import defaultdict, Counter
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# ===== Credit 工程依赖（按你给的脚本接口）=====
from Credit.HGCN.config import get_args as get_base_args
from Credit.HGCN.data_preprocessing_Credit_col import (
    preprocess_node_features,
    generate_hyperedge_dict,
    delete_feature_columns_hgcn,
)
from Credit.HGCN.HGCN_utils import evaluate_hgcn_f1, evaluate_hgcn_acc
from Credit.HGCN.GIF_HGCN_COL_Credit import train_model
from Credit.HGCN.HGCN import HyperGCN, laplacian

# 可选 MIA（如果你本地有这个模块且接口兼容）
try:
    from Credit.HGCN.MIA_HGCN import membership_inference_hgcn
from paths import CREDIT_DATA
    _HAS_MIA = True
except Exception:
    membership_inference_hgcn = None
    _HAS_MIA = False

# =========================================================
# 参数补齐（不改原 config）
# =========================================================
def _ensure_attr(args, name, value):
    if not hasattr(args, name):
        setattr(args, name, value)

def _normalize_ft_steps(v):
    if isinstance(v, (list, tuple)):
        return [int(x) for x in v]
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return [50, 100, 200]
        if "," in s:
            return [int(x.strip()) for x in s.split(",") if x.strip()]
        return [int(s)]
    return [int(v)]

def get_args():
    """
    复用 Credit.HGCN.config.get_args()，并补充 FT 实验所需参数。
    同时把 Credit 数据路径默认改成你指定的 crx.data。
    """
    args = get_base_args()

    # ===== 数据路径（单文件）=====
    # 你的 Credit col 脚本里很多地方可能还是 train_csv 命名，这里保留兼容
    _ensure_attr(args, "train_csv", CREDIT_DATA)
    # 统一一个别名
    _ensure_attr(args, "data_csv", args.train_csv)
    # split 参数（单文件切分）
    _ensure_attr(args, "split_ratio", 0.2)
    _ensure_attr(args, "split_seed", 21)

    # ===== FT baseline 参数 =====
    _ensure_attr(args, "runs", 1)
    _ensure_attr(args, "ft_steps", [50, 100, 200])
    _ensure_attr(args, "ft_lr", 1e-3)
    _ensure_attr(args, "ft_wd", 0.0)
    _ensure_attr(args, "ft_milestones", [])
    _ensure_attr(args, "ft_gamma", 0.1)

    # ===== 列删除设置 =====
    # 兼容 string / list；默认你可改成想删的列名（如 "A1" 不一定可用，先用 age 风格仅占位）
    _ensure_attr(args, "columns_to_unlearn", "A1")

    # ===== 训练日志 =====
    _ensure_attr(args, "log_every", 10)
    _ensure_attr(args, "epochs", 100)

    # ===== 输出 =====
    _ensure_attr(args, "out_csv", "ft_hgcn_col_credit_results.csv")
    _ensure_attr(args, "run_mia", False)

    # ===== 常见 HGCN 参数兜底 =====
    _ensure_attr(args, "depth", 3)
    _ensure_attr(args, "hidden_dim", 128)
    _ensure_attr(args, "dropout", 0.1)
    _ensure_attr(args, "fast", False)
    _ensure_attr(args, "mediators", False)
    _ensure_attr(args, "cuda", torch.cuda.is_available())

    _ensure_attr(args, "lr", 0.01)
    _ensure_attr(args, "weight_decay", 1e-3)
    _ensure_attr(args, "milestones", [80, 120])
    _ensure_attr(args, "gamma", 0.1)

    _ensure_attr(args, "max_nodes_per_hyperedge", 50)

    # Credit 列级脚本常用列配置（如果 config 里没有）
    _ensure_attr(args, "cat_cols", [])
    _ensure_attr(args, "cont_cols", [])
    # 有些旧脚本用 categate_cols 命名
    _ensure_attr(args, "categate_cols", getattr(args, "cat_cols", []))

    # 规范化
    args.ft_steps = _normalize_ft_steps(args.ft_steps)

    # 外部强制覆盖：把 train_csv / data_csv 统一到你的路径（防止 config 里写死）
    args.train_csv = CREDIT_DATA
    args.data_csv = args.train_csv

    return args

# =========================================================
# 工具函数
# =========================================================
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _device(args):
    if getattr(args, "cuda", False) and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")

def _set_model_structure(model: nn.Module, A: torch.Tensor):
    model.structure = A
    if hasattr(model, "layers"):
        for layer in model.layers:
            if hasattr(layer, "reapproximate"):
                layer.reapproximate = False

def _forward_hgcn(model, x):
    out = model(x)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out

def _clone_model(model: nn.Module) -> nn.Module:
    return copy.deepcopy(model)

def _get_last_trainable_module(model: nn.Module):
    """
    尝试找到最后分类头用于 FT-head
    """
    for name in ["classifier", "final", "out_layer", "fc", "linear"]:
        if hasattr(model, name):
            mod = getattr(model, name)
            if isinstance(mod, nn.Module):
                return mod

    last_linear = None
    for _, m in model.named_modules():
        if isinstance(m, nn.Linear):
            last_linear = m
    if last_linear is not None:
        return last_linear

    children = [m for m in model.children()]
    if len(children) > 0:
        return children[-1]
    return None

def _freeze_all_then_unfreeze(model: nn.Module, target_module: Optional[nn.Module]):
    for p in model.parameters():
        p.requires_grad = False
    if target_module is not None:
        for p in target_module.parameters():
            p.requires_grad = True
    else:
        # 兜底：至少放开最后一个参数张量
        params = list(model.parameters())
        if len(params) > 0:
            params[-1].requires_grad = True
            print("[WARN] FT-head target module not found, fallback to last parameter tensor.")

def _build_hgcn_model(X_tr_np, y_tr_np, hyperedges_tr_list, args, device):
    """
    仿照你 Credit col 脚本的 HyperGCN 构造方式。:contentReference[oaicite:2]{index=2}
    """
    cfg = lambda: None
    cfg.d = X_tr_np.shape[1]
    cfg.c = int(np.max(y_tr_np)) + 1
    cfg.depth = args.depth
    cfg.hidden = args.hidden_dim
    cfg.dropout = args.dropout
    cfg.fast = args.fast
    cfg.mediators = args.mediators
    cfg.cuda = bool(getattr(args, "cuda", False) and torch.cuda.is_available())
    cfg.dataset = getattr(args, "dataset", "credit")

    model = HyperGCN(
        X_tr_np.shape[0],
        hyperedges_tr_list,
        X_tr_np,
        cfg
    ).to(device)
    return model

def _train_full_model_hgcn(model, fts_tr, lbls_tr, args):
    """
    保持与你 Credit GIF_HGCN_COL_Credit 的 train_model 一致的训练口径
    """
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.milestones, gamma=args.gamma
    )
    criterion = nn.NLLLoss()

    model = train_model(
        model,
        criterion,
        optimizer,
        scheduler,
        fts_tr,
        lbls_tr,
        num_epochs=args.epochs,
        print_freq=args.log_every,
    )
    return model

def _finetune_steps_hgcn(
    model,
    x_train,
    y_train,
    K: int,
    lr: float,
    weight_decay: float,
    milestones=None,
    gamma=0.1,
):
    """
    在 Edited Hypergraph 上进行 K 步微调（全参数或 head-only 都复用）
    """
    model.train()
    criterion = nn.NLLLoss()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError("No trainable parameters for finetuning.")

    optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = None
    if milestones is not None and len(milestones) > 0:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    t0 = time.time()
    for _ in range(1, int(K) + 1):
        optimizer.zero_grad()
        out = _forward_hgcn(model, x_train)
        loss = criterion(out, y_train)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

    update_time = time.time() - t0
    return model, update_time

def _eval_acc(model, x, y):
    """
    兼容 HGCN_utils 中的输入键名差异：有的用 {"lap":...}，有的用 {"x":...}
    """
    try:
        return float(evaluate_hgcn_acc(model, {"lap": x, "y": y}))
    except Exception:
        return float(evaluate_hgcn_acc(model, {"x": x, "y": y}))

def _eval_f1(model, x, y):
    try:
        return float(evaluate_hgcn_f1(model, {"lap": x, "y": y}))
    except Exception:
        return float(evaluate_hgcn_f1(model, {"x": x, "y": y}))

def _maybe_mia_hgcn(model, args, hyperedges_for_train, x_train_np, y_train_np, retain_member_mask=None):
    """
    可选 MIA（若接口不匹配则返回 NA）
    """
    if not getattr(args, "run_mia", False):
        return None, None

    if not _HAS_MIA:
        print("[MIA] Credit.HGCN.MIA_HGCN not found. Returning NA.")
        return None, None

    try:
        # 这里按你之前 row 脚本常见接口尝试调用；如果你本地签名不同，会被 except 接住。
        out = membership_inference_hgcn(
            X_train=x_train_np,
            y_train=y_train_np,
            hyperedges=hyperedges_for_train,
            target_model=model,
            args=args,
            device=_device(args),
            member_mask=retain_member_mask
        )
        # 兼容 (..., ..., (auc,f1)) 或 (..., (auc,f1))
        if isinstance(out, tuple) and len(out) > 0:
            last = out[-1]
            if isinstance(last, tuple) and len(last) >= 2:
                return float(last[0]), float(last[1])
        return None, None
    except Exception as e:
        print(f"[MIA] failed: {e}")
        return None, None

def _fmt_num(x):
    if x is None:
        return "NA"
    return f"{x:.4f}"

def _save_rows_csv(rows, out_csv):
    if len(rows) == 0:
        return
    keys = list(rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

def _print_summary(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["method"], int(r["K"]))].append(r)

    print("\n== Summary (mean±std) ==")
    for (method, K) in sorted(groups.keys(), key=lambda x: (x[0], x[1])):
        vals = groups[(method, K)]

        def mstd(field):
            arr = np.array([float(v[field]) for v in vals if v[field] is not None], dtype=float)
            if len(arr) == 0:
                return None, None
            return arr.mean(), arr.std(ddof=0)

        e_m, e_s = mstd("edit")
        u_m, u_s = mstd("update")
        t_m, t_s = mstd("total")
        a_m, a_s = mstd("test_acc")
        mia_m, mia_s = mstd("mia_overall")

        def fmt_ms(m, s):
            if m is None:
                return "NA"
            if s is None:
                return f"{m:.4f}"
            return f"{m:.4f}±{s:.4f}"

        print(
            f"{method:16s} K={K:4d} | "
            f"edit={fmt_ms(e_m, e_s)} | "
            f"update={fmt_ms(u_m, u_s)} | "
            f"total={fmt_ms(t_m, t_s)} | "
            f"test_acc={fmt_ms(a_m, a_s)} | "
            f"mia_overall={fmt_ms(mia_m, mia_s)}"
        )

# =========================================================
# 单次运行（一个 seed / run）
# =========================================================
def run_one(args, run_id: int):
    seed = int(getattr(args, "seed", 1)) + run_id
    seed_everything(seed)
    device = _device(args)

    print(f"\n================= RUN {run_id + 1}/{args.runs} =================")
    print(f"[Device] {device} | seed={seed}")

    # -----------------------------------------------------
    # 1) 读 Credit 单文件 + split（仿照你 Credit col 主脚本）
    # -----------------------------------------------------
    # 你的 Credit col 脚本用 pd.read_csv(..., header=None, na_values='?') 再 split。:contentReference[oaicite:3]{index=3}
    df = pd.read_csv(args.data_csv, header=None, na_values='?')
    df = df.reset_index(drop=True)

    # 最后一列为标签，按你原脚本习惯 stratify=df.iloc[:, -1]
    df_tr, df_te = train_test_split(
        df,
        test_size=args.split_ratio,
        random_state=args.split_seed,
        stratify=df.iloc[:, -1]
    )
    df_tr = df_tr.reset_index(drop=True)
    df_te = df_te.reset_index(drop=True)

    print(f"TRAIN samples: {len(df_tr)}, TEST samples: {len(df_te)}")
    print("– TRAIN label dist:", Counter(df_tr.iloc[:, -1]))
    print("– TEST  label dist:", Counter(df_te.iloc[:, -1]))

    # -----------------------------------------------------
    # 2) 预处理（train fit transformer, test transform）
    # -----------------------------------------------------
    X_tr, y_tr, df_tr_proc, transformer = preprocess_node_features(
        data=df_tr,
        transformer=None
    )
    X_te, y_te, df_te_proc, _ = preprocess_node_features(
        data=df_te,
        transformer=transformer
    )

    print(f"➤ TRAIN: X_tr={X_tr.shape}, y_tr={np.shape(y_tr)}")
    print(f"➤ TEST:  X_te={X_te.shape}, y_te={np.shape(y_te)}")

    # -----------------------------------------------------
    # 3) 构建超边（仿照 Credit col 版本）
    # -----------------------------------------------------
    # 你的脚本用 feature_cols = args.cat_cols + args.cont_cols。:contentReference[oaicite:4]{index=4}
    if len(getattr(args, "cat_cols", [])) + len(getattr(args, "cont_cols", [])) > 0:
        feature_cols = list(getattr(args, "cat_cols", [])) + list(getattr(args, "cont_cols", []))
    elif len(getattr(args, "categate_cols", [])) > 0:
        # 老命名兼容（不一定适用于 Credit col，但先兜底）
        feature_cols = list(getattr(args, "categate_cols", []))
    else:
        # 如果 config 里没给列名列表，就退化为“除最后标签列外全部特征列”
        feature_cols = [c for c in df_tr_proc.columns.tolist() if c != df_tr_proc.columns[-1]]
        print(f"[WARN] feature_cols not found in args; fallback to all processed feature cols ({len(feature_cols)} cols).")

    hyper_tr_dict = generate_hyperedge_dict(
        df=df_tr_proc,
        feature_cols=feature_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        device=device
    )
    hyper_te_dict = generate_hyperedge_dict(
        df=df_te_proc,
        feature_cols=feature_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        device=device
    )

    hyper_tr_list = list(hyper_tr_dict.values())
    hyper_te_list = list(hyper_te_dict.values())

    # -----------------------------------------------------
    # 4) Tensor 化 + 构建模型 + 原图 Laplacian
    # -----------------------------------------------------
    fts_tr = torch.from_numpy(X_tr).float().to(device)
    lbls_tr = torch.tensor(y_tr, dtype=torch.long, device=device)

    fts_te = torch.from_numpy(X_te).float().to(device)
    lbls_te = torch.tensor(y_te, dtype=torch.long, device=device)

    model = _build_hgcn_model(X_tr, y_tr, hyper_tr_list, args, device)

    A_tr_before = laplacian(hyper_tr_list, X_tr, args.mediators).to(device)
    A_te_before = laplacian(hyper_te_list, X_te, args.mediators).to(device)

    _set_model_structure(model, A_tr_before)

    # -----------------------------------------------------
    # 5) 训练 Full 模型（原始列）
    # -----------------------------------------------------
    print("== Train Full Model ==")
    t_train0 = time.time()
    model = _train_full_model_hgcn(model, fts_tr, lbls_tr, args)
    train_time = time.time() - t_train0
    print(f"[Full] train_time={train_time:.4f}s")

    _set_model_structure(model, A_te_before)
    test_acc_before = _eval_acc(model, fts_te, lbls_te)
    test_f1_before = _eval_f1(model, fts_te, lbls_te)
    print(f"[Before Col-Unlearning] Test ACC={test_acc_before:.4f} | Test F1={test_f1_before:.4f}")

    # -----------------------------------------------------
    # 6) 列删除（训练集）-> EditedHG
    # -----------------------------------------------------
    cols = args.columns_to_unlearn
    if isinstance(cols, str):
        cols = [c.strip() for c in cols.split(",") if c.strip()]
    elif isinstance(cols, (tuple, set)):
        cols = list(cols)
    elif not isinstance(cols, list):
        cols = [cols]
    print(f"[Column Unlearning] columns_to_unlearn={cols}")

    t_edit0 = time.time()
    X_tr_zero, hyper_tr_zero_dict, A_tr_zero = delete_feature_columns_hgcn(
        X_tensor=fts_tr,
        transformer=transformer,
        column_names=cols,
        hyperedges=hyper_tr_dict,
        df_proc=df_tr_proc,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        mediators=args.mediators,
        use_cuda=(device.type == "cuda")
    )
    edit_time = time.time() - t_edit0

    x_tr_after = X_tr_zero.float().to(device) if isinstance(X_tr_zero, torch.Tensor) else torch.from_numpy(X_tr_zero).float().to(device)

    he_tr_before = len(hyper_tr_dict)
    he_tr_after = len(hyper_tr_zero_dict)
    print(f"[Train Hyperedges] before: {he_tr_before}, after: {he_tr_after}, removed: {he_tr_before - he_tr_after}")

    # -----------------------------------------------------
    # 7) 列删除（测试集）-> 用于评估
    # -----------------------------------------------------
    X_te_zero, hyper_te_zero_dict, A_te_zero = delete_feature_columns_hgcn(
        X_tensor=fts_te,
        transformer=transformer,
        column_names=cols,
        hyperedges=hyper_te_dict,
        df_proc=df_te_proc,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        mediators=args.mediators,
        use_cuda=(device.type == "cuda")
    )
    x_te_after = X_te_zero.float().to(device) if isinstance(X_te_zero, torch.Tensor) else torch.from_numpy(X_te_zero).float().to(device)

    he_te_before = len(hyper_te_dict)
    he_te_after = len(hyper_te_zero_dict)
    print(f"[Test  Hyperedges] before: {he_te_before}, after: {he_te_after}, removed: {he_te_before - he_te_after}")

    # -----------------------------------------------------
    # 8) Full@EditedHG（不更新参数）
    # -----------------------------------------------------
    rows = []

    full_edit_model = _clone_model(model)
    _set_model_structure(full_edit_model, A_te_zero.to(device))
    full_test_acc = _eval_acc(full_edit_model, x_te_after, lbls_te)
    full_test_f1 = _eval_f1(full_edit_model, x_te_after, lbls_te)

    mia_auc, mia_aux = _maybe_mia_hgcn(
        full_edit_model, args,
        hyperedges_for_train=list(hyper_tr_zero_dict.values()),
        x_train_np=x_tr_after.detach().cpu().numpy(),
        y_train_np=lbls_tr.detach().cpu().numpy(),
        retain_member_mask=None
    )

    print(
        f"Full@EditedHG    K={0:4d} | edit={edit_time:.4f} | update={0.0:.4f} | total={edit_time:.4f} | "
        f"test_acc={full_test_acc:.4f} | mia_overall={_fmt_num(mia_auc)}"
    )

    rows.append({
        "run_id": run_id + 1,
        "method": "Full@EditedHG",
        "K": 0,
        "edit": float(edit_time),
        "update": 0.0,
        "total": float(edit_time),
        "test_acc": float(full_test_acc),
        "test_f1": float(full_test_f1),
        "mia_overall": None if mia_auc is None else float(mia_auc),
        "mia_aux": None if mia_aux is None else float(mia_aux),
    })

    # -----------------------------------------------------
    # 9) FT-K（全参数微调）
    # -----------------------------------------------------
    print("\n== FT-K (warm-start on EditedHG) ==")
    for K in args.ft_steps:
        m_ft = _clone_model(model)

        # 训练阶段结构切到编辑后的训练图
        _set_model_structure(m_ft, A_tr_zero.to(device))
        for p in m_ft.parameters():
            p.requires_grad = True

        m_ft, update_time = _finetune_steps_hgcn(
            m_ft,
            x_train=x_tr_after,
            y_train=lbls_tr,
            K=int(K),
            lr=float(args.ft_lr),
            weight_decay=float(args.ft_wd),
            milestones=list(getattr(args, "ft_milestones", [])),
            gamma=float(getattr(args, "ft_gamma", 0.1)),
        )

        # 测试阶段结构切到编辑后的测试图
        _set_model_structure(m_ft, A_te_zero.to(device))
        test_acc = _eval_acc(m_ft, x_te_after, lbls_te)
        test_f1 = _eval_f1(m_ft, x_te_after, lbls_te)
        mia_auc, mia_aux = _maybe_mia_hgcn(
            m_ft, args,
            hyperedges_for_train=list(hyper_tr_zero_dict.values()),
            x_train_np=x_tr_after.detach().cpu().numpy(),
            y_train_np=lbls_tr.detach().cpu().numpy(),
            retain_member_mask=None
        )

        total_time = edit_time + update_time
        print(
            f"FT-K@EditedHG    K={int(K):4d} | edit={edit_time:.4f} | update={update_time:.4f} | total={total_time:.4f} | "
            f"test_acc={test_acc:.4f} | mia_overall={_fmt_num(mia_auc)}"
        )

        rows.append({
            "run_id": run_id + 1,
            "method": "FT-K@EditedHG",
            "K": int(K),
            "edit": float(edit_time),
            "update": float(update_time),
            "total": float(total_time),
            "test_acc": float(test_acc),
            "test_f1": float(test_f1),
            "mia_overall": None if mia_auc is None else float(mia_auc),
            "mia_aux": None if mia_aux is None else float(mia_aux),
        })

    # -----------------------------------------------------
    # 10) FT-head（只训练最后层）
    # -----------------------------------------------------
    print("\n== FT-head (only train last layer on EditedHG) ==")
    for K in args.ft_steps:
        m_head = _clone_model(model)

        head_module = _get_last_trainable_module(m_head)
        _freeze_all_then_unfreeze(m_head, head_module)

        _set_model_structure(m_head, A_tr_zero.to(device))

        m_head, update_time = _finetune_steps_hgcn(
            m_head,
            x_train=x_tr_after,
            y_train=lbls_tr,
            K=int(K),
            lr=float(args.ft_lr),
            weight_decay=float(args.ft_wd),
            milestones=list(getattr(args, "ft_milestones", [])),
            gamma=float(getattr(args, "ft_gamma", 0.1)),
        )

        _set_model_structure(m_head, A_te_zero.to(device))
        test_acc = _eval_acc(m_head, x_te_after, lbls_te)
        test_f1 = _eval_f1(m_head, x_te_after, lbls_te)
        mia_auc, mia_aux = _maybe_mia_hgcn(
            m_head, args,
            hyperedges_for_train=list(hyper_tr_zero_dict.values()),
            x_train_np=x_tr_after.detach().cpu().numpy(),
            y_train_np=lbls_tr.detach().cpu().numpy(),
            retain_member_mask=None
        )

        total_time = edit_time + update_time
        print(
            f"FT-head@EditedHG K={int(K):4d} | edit={edit_time:.4f} | update={update_time:.4f} | total={total_time:.4f} | "
            f"test_acc={test_acc:.4f} | mia_overall={_fmt_num(mia_auc)}"
        )

        rows.append({
            "run_id": run_id + 1,
            "method": "FT-head@EditedHG",
            "K": int(K),
            "edit": float(edit_time),
            "update": float(update_time),
            "total": float(total_time),
            "test_acc": float(test_acc),
            "test_f1": float(test_f1),
            "mia_overall": None if mia_auc is None else float(mia_auc),
            "mia_aux": None if mia_aux is None else float(mia_aux),
        })

    return rows

# =========================================================
# 主函数
# =========================================================
def main():
    warnings.filterwarnings("ignore")
    args = get_args()

    # 打印关键配置（方便排错）
    print("==== FT HGCN Column Baseline (Credit) Config ====")
    for k in [
        "data_csv", "train_csv", "split_ratio", "split_seed",
        "columns_to_unlearn", "cat_cols", "cont_cols",
        "max_nodes_per_hyperedge", "epochs", "lr", "weight_decay",
        "ft_steps", "ft_lr", "ft_wd", "runs", "run_mia", "out_csv"
    ]:
        if hasattr(args, k):
            print(f"{k}: {getattr(args, k)}")
    print("=================================================")

    if not os.path.exists(args.data_csv):
        print(f"[WARN] data file not found: {args.data_csv}")

    all_rows = []
    for run_id in range(int(args.runs)):
        rows = run_one(args, run_id)
        all_rows.extend(rows)

    _save_rows_csv(all_rows, args.out_csv)
    print(f"\n[Saved] {args.out_csv}")

    _print_summary(all_rows)

if __name__ == "__main__":
    main()