#!/usr/bin/env python3
# coding: utf-8

import os
import time
import copy
import random
import argparse
import warnings
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# ===== Credit project modules =====
from Credit.HGCN.data_preprocessing_credit import (
    preprocess_node_features,
    generate_hyperedge_dict,
)
from Credit.HGCN.HGCN import HyperGCN, laplacian
from Credit.HGCN.GIF_HGCN_ROW_Credit import train_model
from Credit.HGCN.HGCN_utils import evaluate_hgcn_acc, evaluate_hgcn_f1
from Credit.HGCN.MIA_HGCN import membership_inference_hgcn
from paths import CREDIT_DATA

# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 为了稳定复现（会略慢）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _device(args):
    if args.cuda and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")

def _clone_model(model):
    return copy.deepcopy(model)

def freeze_all_but_head(model: nn.Module):
    """
    FT-head：只训练最后分类层/头部参数
    这里做法尽量稳妥：先全冻结，再按名字启用包含 classifier/fc/out/head 的参数。
    如果你的 HyperGCN 头层命名不同，可在下面关键词里补充。
    """
    for p in model.parameters():
        p.requires_grad = False

    head_keywords = ["classifier", "fc", "out", "head", "linear"]
    enabled = 0
    for name, p in model.named_parameters():
        lname = name.lower()
        if any(k in lname for k in head_keywords):
            p.requires_grad = True
            enabled += p.numel()

    # 兜底：如果没匹配到命名，就启用最后一个参数张量，避免“没有可训练参数”的报错
    if enabled == 0:
        params = list(model.parameters())
        if len(params) > 0:
            params[-1].requires_grad = True
            enabled = params[-1].numel()
        print("[WARN] FT-head did not match head parameter names; fallback to last parameter tensor.")

    print(f"[FT-head] trainable params = {enabled}")

def build_masks(N: int, deleted: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    del_mask = torch.zeros(N, dtype=torch.bool, device=device)
    del_mask[deleted] = True
    retain_mask = ~del_mask
    return retain_mask, del_mask

def member_mask_from_retain(retain_mask: torch.Tensor) -> np.ndarray:
    # retained=1, deleted=0
    return retain_mask.detach().cpu().numpy().astype(np.int64)

def member_mask_from_deleted(deleted: torch.Tensor, N: int) -> np.ndarray:
    # deleted=1, others=0
    mm = np.zeros(N, dtype=np.int64)
    mm[deleted.detach().cpu().numpy()] = 1
    return mm

def _remove_nodes_from_hyperedges(hyperedges: List[List[int]], deleted_nodes: List[int]) -> List[List[int]]:
    """
    Edited hypergraph（行删除后）：
    - 保持节点编号不变（仍是 0..N-1）
    - 仅在结构上把 deleted 节点从每条超边中移除
    - 超边长度 < 2 的丢弃
    """
    del_set = set(int(x) for x in deleted_nodes)
    new_edges = []
    for e in hyperedges:
        e2 = [int(v) for v in e if int(v) not in del_set]
        if len(e2) >= 2:
            new_edges.append(e2)
    return new_edges

# ============================================================
# Model / Train / Eval helpers
# ============================================================

def build_hgcn_model(X_np: np.ndarray, hyperedges: List[List[int]], args, device: torch.device):
    cfg = lambda: None
    cfg.d = X_np.shape[1]
    # Credit 二分类（通常 0/1）
    cfg.c = 2
    cfg.depth = args.depth
    cfg.hidden = args.hidden_dim
    cfg.dropout = args.dropout
    cfg.fast = args.fast
    cfg.mediators = args.mediators
    cfg.cuda = bool(args.cuda and torch.cuda.is_available())
    cfg.dataset = "credit"

    model = HyperGCN(
        X_np.shape[0],
        hyperedges,
        X_np,
        cfg
    ).to(device)

    A = laplacian(hyperedges, X_np, args.mediators).to(device)
    model.structure = A
    for l in model.layers:
        l.reapproximate = False
    return model, A

def train_full_model(model, fts_tr, lbls_tr, args):
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    criterion = nn.NLLLoss()
    model = train_model(
        model, criterion, optimizer, scheduler,
        fts_tr, lbls_tr,
        num_epochs=args.epochs,
        print_freq=args.log_every
    )
    return model

def finetune_steps(model, x_after, y, retain_mask, steps: int, lr: float, weight_decay: float):
    """
    在 edited hypergraph 上对 retain 子集做小步 finetune
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError("No trainable parameters found for finetune.")

    optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
    criterion = nn.NLLLoss()

    model.train()
    for _ in range(int(steps)):
        optimizer.zero_grad()
        out = model(x_after)
        loss = criterion(out[retain_mask], y[retain_mask])
        loss.backward()
        optimizer.step()

    return model

@torch.no_grad()
def eval_split(model, x, y, A, mask: Optional[torch.Tensor] = None):
    """
    返回 (f1, acc)，与现有日志风格兼容
    """
    model.eval()
    model.structure = A
    for l in model.layers:
        l.reapproximate = False

    data = {"lap": x, "y": y}
    if mask is not None:
        data["train_mask"] = mask

    f1 = evaluate_hgcn_f1(model, data)
    acc = evaluate_hgcn_acc(model, data)
    return f1, acc

def _safe_mia(args, tag, model, A_used, hedges_used, X_tr, y_tr_np, device, retain_mask, deleted, N):
    """
    尽量兼容你现有 MIA 实现；失败时返回 None，不阻断 FT 实验。
    """
    mia_overall_auc = None
    mia_deleted_auc = None

    if not args.run_mia:
        return mia_overall_auc, mia_deleted_auc

    model.structure = A_used
    for l in model.layers:
        l.reapproximate = False

    # 1) overall / retain-membership
    try:
        mm_retain = member_mask_from_retain(retain_mask)
        print(f"— MIA on {tag} (overall/retain-membership) —")
        out = membership_inference_hgcn(
            X_train=X_tr,
            y_train=y_tr_np,
            hyperedges=hedges_used,
            target_model=model,
            args=args,
            device=device,
            member_mask=mm_retain
        )
        # 兼容不同返回格式：有的返回 (..., ..., (auc,f1))，有的返回 (..., (auc,f1))
        if isinstance(out, tuple) and len(out) >= 1:
            last = out[-1]
            if isinstance(last, tuple) and len(last) >= 1:
                mia_overall_auc = float(last[0])
    except Exception as e:
        print(f"[WARN] MIA overall failed on {tag}: {e}")

    # 2) deleted-only
    try:
        mm_deleted = member_mask_from_deleted(deleted, N)
        print(f"— MIA on {tag} (deleted-only membership) —")
        out2 = membership_inference_hgcn(
            X_train=X_tr,
            y_train=y_tr_np,
            hyperedges=hedges_used,
            target_model=model,
            args=args,
            device=device,
            member_mask=mm_deleted
        )
        if isinstance(out2, tuple) and len(out2) >= 1:
            last2 = out2[-1]
            if isinstance(last2, tuple) and len(last2) >= 1:
                mia_deleted_auc = float(last2[0])
    except Exception as e:
        print(f"[WARN] MIA deleted-only failed on {tag}: {e}")

    return mia_overall_auc, mia_deleted_auc

# ============================================================
# One run
# ============================================================

def run_one(args, run_id: int):
    device = _device(args)
    seed = args.seed + run_id
    set_seed(seed)

    print(f"[Device] {device} | seed={seed}")

    # ===== Load full Credit dataset (single file) =====
    X_full, y_full, df_full, transformer = preprocess_node_features(
        data=args.data_csv,
        transformer=None
    )

    # ===== Split train/test =====
    df_tr, df_te, y_tr_np, y_te_np = train_test_split(
        df_full, y_full,
        test_size=args.split_ratio,
        stratify=y_full,
        random_state=args.split_seed
    )
    df_tr = df_tr.reset_index(drop=True)
    df_te = df_te.reset_index(drop=True)
    print(f"Train: {len(df_tr)}, Test: {len(df_te)}")

    # ===== Re-transform with same transformer =====
    X_tr, y_tr_np, df_tr_proc, _ = preprocess_node_features(data=df_tr, transformer=transformer)
    X_te, y_te_np, df_te_proc, _ = preprocess_node_features(data=df_te, transformer=transformer)

    # ===== Hyperedges =====
    hedges_tr = list(generate_hyperedge_dict(
        df_tr_proc,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        device=device
    ).values())

    hedges_te = list(generate_hyperedge_dict(
        df_te_proc,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        device=device
    ).values())

    # ===== Torch tensors =====
    fts_tr = torch.from_numpy(X_tr).float().to(device)
    lbls_tr = torch.tensor(y_tr_np, dtype=torch.long, device=device)

    fts_te = torch.from_numpy(X_te).float().to(device)
    lbls_te = torch.tensor(y_te_np, dtype=torch.long, device=device)

    # ===== Build model + structures =====
    model_full, A_tr = build_hgcn_model(X_tr, hedges_tr, args, device)
    A_te = laplacian(hedges_te, X_te, args.mediators).to(device)

    # ===== Train full =====
    print("== Train Full Model ==")
    t0 = time.time()
    model_full = train_full_model(model_full, fts_tr, lbls_tr, args)
    full_train_time = time.time() - t0

    f1_test, acc_test = eval_split(model_full, fts_te, lbls_te, A_te, mask=None)
    print(f"[Full] Test ACC={acc_test:.4f} F1={f1_test:.4f} | train_time={full_train_time:.4f}s")

    # ===== Sample deleted rows =====
    N = X_tr.shape[0]
    n_del = max(1, int(N * args.remove_ratio))
    rng = np.random.default_rng(seed)
    deleted_idx = rng.choice(np.arange(N), size=n_del, replace=False)
    deleted = torch.tensor(deleted_idx, dtype=torch.long, device=device)
    print(f"[Delete] ratio={args.remove_ratio:.4f}, deleted={len(deleted)}/{N}")

    retain_mask, del_mask = build_masks(N, deleted, device)

    # ===== Feature-zero edited training features =====
    fts_tr_after = fts_tr.clone()
    fts_tr_after[deleted] = 0.0

    # ===== Edited HG + edited Laplacian time =====
    t_edit0 = time.time()
    deleted_cpu = deleted.detach().cpu().tolist()
    hedges_tr_edit = _remove_nodes_from_hyperedges(hedges_tr, deleted_cpu)
    A_tr_edit = laplacian(hedges_tr_edit, X_tr, args.mediators).to(device)
    edit_time_sec = time.time() - t_edit0

    print(
        f"[Edited HG] #edges(original)={len(hedges_tr)} -> #edges(edited)={len(hedges_tr_edit)} "
        f"| edit_time={edit_time_sec:.4f}s"
    )

    # ===== Evaluate retain/forget before FT (on edited HG) =====
    f1_ret, acc_ret = eval_split(model_full, fts_tr, lbls_tr, A_tr_edit, mask=retain_mask)
    f1_for, acc_for = eval_split(model_full, fts_tr, lbls_tr, A_tr_edit, mask=del_mask)
    print(
        f"[Full@EditedHG] retain_acc={acc_ret:.4f} retain_f1={f1_ret:.4f} | "
        f"forget_acc={acc_for:.4f} forget_f1={f1_for:.4f}"
    )

    results = []

    # ===== Record Full@EditedHG =====
    mia_o, mia_d = _safe_mia(
        args=args, tag="Full@EditedHG", model=model_full, A_used=A_tr_edit, hedges_used=hedges_tr_edit,
        X_tr=X_tr, y_tr_np=y_tr_np, device=device, retain_mask=retain_mask, deleted=deleted, N=N
    )
    results.append({
        "run": run_id,
        "method": "Full@EditedHG",
        "K": 0,
        "edit_time_sec": edit_time_sec,
        "time_sec": full_train_time,
        "total_time_sec": edit_time_sec + full_train_time,
        "test_acc": float(acc_test),
        "retain_acc": float(acc_ret),
        "forget_acc": float(acc_for),
        "mia_overall_auc": mia_o,
        "mia_deleted_auc": mia_d,
        "seed": seed
    })

    # ============================================================
    # FT-K
    # ============================================================
    print("\n== FT-K (warm-start) on Edited Hypergraph ==")
    for K in args.ft_steps:
        m = _clone_model(model_full)

        m.structure = A_tr_edit
        for l in m.layers:
            l.reapproximate = False

        t1 = time.time()
        m = finetune_steps(
            m,
            x_after=fts_tr_after,
            y=lbls_tr,
            retain_mask=retain_mask,
            steps=K,
            lr=args.ft_lr,
            weight_decay=args.ft_wd
        )
        ft_time = time.time() - t1

        f1_test_k, acc_test_k = eval_split(m, fts_te, lbls_te, A_te, mask=None)
        f1_ret_k, acc_ret_k = eval_split(m, fts_tr, lbls_tr, A_tr_edit, mask=retain_mask)
        f1_for_k, acc_for_k = eval_split(m, fts_tr, lbls_tr, A_tr_edit, mask=del_mask)

        mia_o, mia_d = _safe_mia(
            args=args, tag=f"FT-K@EditedHG(K={K})", model=m, A_used=A_tr_edit, hedges_used=hedges_tr_edit,
            X_tr=X_tr, y_tr_np=y_tr_np, device=device, retain_mask=retain_mask, deleted=deleted, N=N
        )

        print(
            f"FT-K@EditedHG  K={K:4d} | edit={edit_time_sec:.4f} | update={ft_time:.4f} | total={ft_time + edit_time_sec:.4f} "
            f"| test_acc={acc_test_k:.4f} | retain_acc={acc_ret_k:.4f} | forget_acc={acc_for_k:.4f} "
            f"| mia_overall={'NA' if mia_o is None else f'{mia_o:.4f}'} | mia_deleted={'NA' if mia_d is None else f'{mia_d:.4f}'}"
        )

        results.append({
            "run": run_id,
            "method": "FT-K@EditedHG",
            "K": int(K),
            "edit_time_sec": float(edit_time_sec),
            "time_sec": float(ft_time),
            "total_time_sec": float(ft_time + edit_time_sec),
            "test_acc": float(acc_test_k),
            "retain_acc": float(acc_ret_k),
            "forget_acc": float(acc_for_k),
            "mia_overall_auc": mia_o,
            "mia_deleted_auc": mia_d,
            "seed": seed
        })

    # ============================================================
    # FT-head
    # ============================================================
    print("\n== FT-head (only train last layer) on Edited Hypergraph ==")
    for K in args.ft_steps:
        m = _clone_model(model_full)
        freeze_all_but_head(m)

        m.structure = A_tr_edit
        for l in m.layers:
            l.reapproximate = False

        t1 = time.time()
        m = finetune_steps(
            m,
            x_after=fts_tr_after,
            y=lbls_tr,
            retain_mask=retain_mask,
            steps=K,
            lr=args.ft_lr,
            weight_decay=args.ft_wd
        )
        ft_time = time.time() - t1

        f1_test_k, acc_test_k = eval_split(m, fts_te, lbls_te, A_te, mask=None)
        f1_ret_k, acc_ret_k = eval_split(m, fts_tr, lbls_tr, A_tr_edit, mask=retain_mask)
        f1_for_k, acc_for_k = eval_split(m, fts_tr, lbls_tr, A_tr_edit, mask=del_mask)

        mia_o, mia_d = _safe_mia(
            args=args, tag=f"FT-head@EditedHG(K={K})", model=m, A_used=A_tr_edit, hedges_used=hedges_tr_edit,
            X_tr=X_tr, y_tr_np=y_tr_np, device=device, retain_mask=retain_mask, deleted=deleted, N=N
        )

        print(
            f"FT-head@EditedHG K={K:4d} | edit={edit_time_sec:.4f} | update={ft_time:.4f} | total={ft_time + edit_time_sec:.4f} "
            f"| test_acc={acc_test_k:.4f} | retain_acc={acc_ret_k:.4f} | forget_acc={acc_for_k:.4f} "
            f"| mia_overall={'NA' if mia_o is None else f'{mia_o:.4f}'} | mia_deleted={'NA' if mia_d is None else f'{mia_d:.4f}'}"
        )

        results.append({
            "run": run_id,
            "method": "FT-head@EditedHG",
            "K": int(K),
            "edit_time_sec": float(edit_time_sec),
            "time_sec": float(ft_time),
            "total_time_sec": float(ft_time + edit_time_sec),
            "test_acc": float(acc_test_k),
            "retain_acc": float(acc_ret_k),
            "forget_acc": float(acc_for_k),
            "mia_overall_auc": mia_o,
            "mia_deleted_auc": mia_d,
            "seed": seed
        })

    return results

# ============================================================
# Summary
# ============================================================

def summarize_and_save(all_rows, out_csv):
    df = pd.DataFrame(all_rows)
    df.to_csv(out_csv, index=False)

    agg_cols = [
        "edit_time_sec", "time_sec", "total_time_sec",
        "test_acc", "retain_acc", "forget_acc",
        "mia_overall_auc", "mia_deleted_auc"
    ]
    g = df.groupby(["method", "K"], dropna=False)[agg_cols].agg(["mean", "std"]).reset_index()

    print("\n== Summary (mean±std) ==")
    for _, r in g.iterrows():
        method = r[("method", "")]
        K = r[("K", "")]

        def fmt(col):
            m = r[(col, "mean")]
            s = r[(col, "std")]
            if pd.isna(m):
                return "NA"
            if pd.isna(s):
                return f"{m:.4f}"
            return f"{m:.4f}±{s:.4f}"

        print(
            f"{method:16s} K={int(K):4d} | "
            f"edit={fmt('edit_time_sec')} | "
            f"update={fmt('time_sec')} | "
            f"total={fmt('total_time_sec')} | "
            f"test_acc={fmt('test_acc')} | "
            f"retain_acc={fmt('retain_acc')} | "
            f"forget_acc={fmt('forget_acc')} | "
            f"mia_overall={fmt('mia_overall_auc')} | "
            f"mia_deleted={fmt('mia_deleted_auc')}"
        )

    print(f"\n[Saved] {out_csv}")

# ============================================================
# Args / Main
# ============================================================

def get_args():
    parser = argparse.ArgumentParser(description="HGCN FT row baselines on Credit (Edited Hypergraph)")

    # ===== Data =====
    parser.add_argument(
        "--data-csv", type=str,
        default=CREDIT_DATA,
        help="Credit Approval data path"
    )
    parser.add_argument("--split-ratio", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--split-seed", type=int, default=42, help="Train/test split random seed")

    # ===== Hypergraph / model =====
    parser.add_argument("--max-nodes-per-hyperedge", type=int, default=50)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--mediators", action="store_true")

    # ===== Full training =====
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--milestones", nargs="+", type=int, default=[100, 150])
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=10)

    # ===== FT baselines =====
    parser.add_argument("--ft-steps", nargs="+", type=int, default=[50, 100, 200],
                        help="FT steps list for FT-K / FT-head")
    parser.add_argument("--ft-lr", type=float, default=1e-3)
    parser.add_argument("--ft-wd", type=float, default=1e-4)

    # ===== Row deletion =====
    parser.add_argument("--remove-ratio", type=float, default=0.1)

    # ===== MIA =====
    parser.add_argument("--run-mia", action="store_true", help="Whether to run MIA (can be slow)",default=True)
    # 兼容你 Credit MIA 代码可能依赖的参数
    parser.add_argument("--neighbor-k", type=int, default=12)   # 即使本脚本不用，也保留兼容
    parser.add_argument("--gif-iters", type=int, default=80)    # 兼容 MIA/其它共用 parser 的情况
    parser.add_argument("--gif-damp", type=float, default=0.01)
    parser.add_argument("--gif-scale", type=float, default=1e7)

    # ===== Run / device =====
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda", action="store_true", default=True)
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument("--out-csv", type=str, default="ft_credit_hgcn_row_results.csv")

    args = parser.parse_args()

    if args.cpu:
        args.cuda = False

    return args

def main():
    warnings.filterwarnings("ignore")
    args = get_args()

    if not os.path.exists(args.data_csv):
        print(f"[WARN] data file not found: {args.data_csv}")

    all_rows = []
    for run_id in range(args.runs):
        print(f"\n================= FT-Credit RUN {run_id + 1}/{args.runs} =================")
        rows = run_one(args, run_id)
        all_rows.extend(rows)

    summarize_and_save(all_rows, args.out_csv)

if __name__ == "__main__":
    main()