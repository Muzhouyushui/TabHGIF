#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Credit + HGAT + Column Unlearning FT baselines (Edited Hypergraph)

Methods:
  - Full@EditedHG
  - FT-K@EditedHG
  - FT-head@EditedHG

Dataset:
  - Credit Approval (crx.data, single file)
  - internal train/test split

Column-unlearning behavior:
  1) zero-out encoded dimensions corresponding to deleted raw columns
  2) try to remove hyperedges associated with deleted columns (if key names carry column provenance)
  3) rebuild incidence matrix H for edited train/test hypergraphs
"""

import os
import time
import copy
import random
import argparse
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# ===== Credit HGAT modules (adapt to your project) =====
from Credit.HGAT.data_preprocessing_credit import (
    preprocess_node_features,
    generate_hyperedge_dict,
)
from Credit.HGAT.HGAT_new import HGAT_JK

# Optional MIA
try:
    from Credit.HGAT.MIA_HGAT import membership_inference_hgat
    _HAS_MIA = True
except Exception:
    membership_inference_hgat = None
    _HAS_MIA = False


# =========================================================
# Utilities
# =========================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_str: str) -> torch.device:
    if torch.cuda.is_available() and ("cuda" in device_str):
        return torch.device(device_str)
    return torch.device("cpu")


def build_incidence_matrix(hyperedges: Dict, num_nodes: int, device=None) -> torch.Tensor:
    """
    Build sparse incidence matrix H with shape [E, N]
    hyperedges: dict(edge_id -> list[node_id])
    """
    n_edges = len(hyperedges)
    H = torch.zeros((n_edges, num_nodes), dtype=torch.float32, device=device)
    for i, nodes in enumerate(hyperedges.values()):
        if len(nodes) == 0:
            continue
        idx = torch.as_tensor(nodes, dtype=torch.long, device=device)
        H[i, idx] = 1.0
    return H.to_sparse()


def get_feature_names_safe(transformer) -> List[str]:
    try:
        return list(transformer.get_feature_names_out())
    except Exception:
        return list(transformer.get_feature_names())


def find_encoded_dims_for_raw_cols(transformer, del_cols: List[str]) -> Tuple[List[int], List[str]]:
    """
    Map raw column names -> encoded feature dimensions (supports one-hot names).
    """
    feat_names = get_feature_names_safe(transformer)
    deleted_idxs = []

    for col in del_cols:
        matches = []
        for i, fn in enumerate(feat_names):
            # robust token matching across styles like:
            # cat__A4_u, A4_u, num__A2, A2
            normalized = fn.replace("=", "_").replace("-", "_")
            tokens = [seg for seg in normalized.split("_") if seg]

            if (col in tokens) or (f"__{col}_" in fn) or fn.startswith(col + "_") or (f"__{col}" in fn) or (fn == col):
                matches.append(i)

        if len(matches) == 0:
            # fallback substring match
            matches = [i for i, fn in enumerate(feat_names) if col in fn]

        if len(matches) == 0:
            raise ValueError(f"在编码后特征中未找到原始列 '{col}'，请检查列名是否正确。")

        deleted_idxs.extend(matches)

    deleted_idxs = sorted(set(deleted_idxs))
    return deleted_idxs, feat_names


def remove_hyperedges_by_deleted_columns(hyperedges: Dict, deleted_names: List[str]) -> Dict:
    """
    Try to remove hyperedges generated from deleted columns.
    Works best if hyperedge keys contain column names (string keys).
    If keys are purely integers and no column provenance exists, keep all edges (safe fallback).
    """
    if len(hyperedges) == 0:
        return {}

    first_key = next(iter(hyperedges.keys()))

    # If keys contain semantic names, filter them
    if isinstance(first_key, str):
        new_edges = {}
        eid = 0
        for k, nodes in hyperedges.items():
            hit = False
            k_str = str(k)
            for col in deleted_names:
                if col in k_str:
                    hit = True
                    break
            if (not hit) and len(nodes) > 0:
                new_edges[eid] = list(nodes)
                eid += 1
        return new_edges

    # Fallback: cannot infer edge-column mapping
    return {i: list(v) for i, v in enumerate(hyperedges.values())}


def delete_feature_columns_hgat_ft(
    X_t: torch.Tensor,
    transformer,
    deleted_names: List[str],
    hyperedges: Dict,
    device: torch.device
):
    """
    Column deletion for FT baseline:
      1) zero corresponding encoded dimensions
      2) remove hyperedges associated with deleted columns (if inferable)
      3) rebuild H
    """
    X_u = X_t.clone()

    deleted_idxs, feat_names = find_encoded_dims_for_raw_cols(transformer, deleted_names)
    if len(deleted_idxs) > 0:
        idx_t = torch.as_tensor(deleted_idxs, dtype=torch.long, device=device)
        X_u[:, idx_t] = 0.0

    edges_u = remove_hyperedges_by_deleted_columns(hyperedges, deleted_names)
    H_u = build_incidence_matrix(edges_u, X_u.size(0), device=device)

    return X_u, edges_u, H_u, deleted_idxs, feat_names


@torch.no_grad()
def eval_acc(model, X_t, y_t, H_t) -> float:
    model.eval()
    logits = model(X_t, H_t)
    preds = logits.argmax(dim=1)
    return float((preds == y_t).float().mean().item())


@torch.no_grad()
def eval_f1_acc(model, X_t, y_t, H_t) -> Tuple[float, float]:
    model.eval()
    logits = model(X_t, H_t)
    preds = logits.argmax(dim=1)
    acc = accuracy_score(y_t.detach().cpu().numpy(), preds.detach().cpu().numpy())
    f1 = f1_score(y_t.detach().cpu().numpy(), preds.detach().cpu().numpy(), average="micro")
    return float(acc), float(f1)


def train_model_hgat(
    model,
    criterion,
    optimizer,
    scheduler,
    X_train_t,
    y_train_t,
    H_train,
    num_epochs=200,
    print_freq=50
):
    best_state = copy.deepcopy(model.state_dict())
    best_acc = -1.0
    t0 = time.time()

    for ep in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        logits = model(X_train_t, H_train)
        loss = criterion(logits, y_train_t)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            tr_acc = (preds == y_train_t).float().mean().item()

        if tr_acc > best_acc:
            best_acc = tr_acc
            best_state = copy.deepcopy(model.state_dict())

        if ep == 1 or ep % print_freq == 0 or ep == num_epochs:
            print(f"[Train] ep {ep:4d}/{num_epochs} | loss={loss.item():.4f} | train_acc={tr_acc:.4f}")

    train_time = time.time() - t0
    model.load_state_dict(best_state)
    print(f"Training complete in {train_time:.2f}s")
    print(f"Best Train Acc: {best_acc:.4f}")
    return model, train_time


def freeze_all_but_head_hgat(model: nn.Module):
    """
    Best-effort head-only finetuning.
    """
    for p in model.parameters():
        p.requires_grad = False

    # Try common names first
    cand_names = ["classifier", "fc", "out", "lin", "proj", "head", "mlp", "hgat2", "att2"]
    for name in cand_names:
        if hasattr(model, name):
            m = getattr(model, name)
            if isinstance(m, nn.Module):
                found = False
                for p in m.parameters():
                    p.requires_grad = True
                    found = True
                if found:
                    print(f"[FT-head] Unfreezing module: {name}")
                    return

    # fallback: last child with params
    for child in reversed(list(model.children())):
        ps = list(child.parameters())
        if len(ps) > 0:
            for p in ps:
                p.requires_grad = True
            print(f"[FT-head] Fallback unfreezing last child: {child.__class__.__name__}")
            return

    # final fallback
    for p in model.parameters():
        p.requires_grad = True
    print("[FT-head] Warning: fallback to full-parameter finetune")


def finetune_steps(
    model,
    X_t,
    y_t,
    H_t,
    steps: int,
    lr: float,
    wd: float
):
    if steps <= 0:
        return model, 0.0

    params = [p for p in model.parameters() if p.requires_grad]
    if len(params) == 0:
        raise RuntimeError("No trainable parameters in finetune_steps.")

    optimizer = optim.Adam(params, lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()

    t0 = time.time()
    for _ in range(steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(X_t, H_t)
        loss = criterion(logits, y_t)
        loss.backward()
        optimizer.step()
    dt = time.time() - t0
    return model, dt


def maybe_mia_hgat(model, X_np, y_np, edges, args, device):
    if (not args.run_mia) or (not _HAS_MIA):
        return None
    try:
        # keep compatible with common return format:
        # _, (auc_s, f1_s), (auc_t, f1_t)
        out = membership_inference_hgat(
            X_np, y_np, edges, target_model=model, args=args, device=device
        )
        if isinstance(out, tuple) and len(out) >= 3:
            target_part = out[-1]
            if isinstance(target_part, tuple) and len(target_part) >= 1:
                return float(target_part[0])
        return None
    except Exception as e:
        print(f"[WARN] MIA failed: {e}")
        return None


# =========================================================
# Credit single-file loading
# =========================================================
def load_credit_train_test(args):
    """
    Credit Approval dataset (crx.data)
    16 attributes + 1 label -> A1 ... A16, where A16 is label (common convention)
    """
    col_names = [f"A{i}" for i in range(1, 17)]  # A1..A16
    df_full = pd.read_csv(
        args.data_csv,
        header=None,
        names=col_names,
        na_values="?",
        skipinitialspace=True
    )

    if args.label_col not in df_full.columns:
        raise ValueError(f"Label column '{args.label_col}' not in columns: {list(df_full.columns)}")

    df_train, df_test = train_test_split(
        df_full,
        test_size=args.split_ratio,
        stratify=df_full[args.label_col],
        random_state=args.split_seed
    )
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    return df_train, df_test


# =========================================================
# Main one run
# =========================================================
def run_one(args, run_id: int):
    seed = args.seed + run_id
    set_seed(seed)
    device = get_device(args.device)

    print(f"[Device] {device} | seed={seed}")

    # 1) Load & split Credit single-file
    df_train_raw, df_test_raw = load_credit_train_test(args)
    print(f"Train: {len(df_train_raw)} rows, Test: {len(df_test_raw)} rows")
    print("Train label dist:", Counter(df_train_raw[args.label_col]))
    print("Test  label dist:", Counter(df_test_raw[args.label_col]))

    # 2) Preprocess (fit transformer on train, reuse on test)
    X_train, y_train, df_train_proc, transformer = preprocess_node_features(df_train_raw)
    X_test, y_test, df_test_proc, _ = preprocess_node_features(df_test_raw, transformer=transformer)

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train, dtype=np.int64)
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test, dtype=np.int64)

    print(f"Processed train: X={X_train.shape}, y={y_train.shape}")
    print(f"Processed test : X={X_test.shape}, y={y_test.shape}")

    # 3) Build original hypergraph
    t_hg0 = time.time()
    # Credit 版本通常 generate_hyperedge_dict(df_proc, max_nodes..., device=...)
    try:
        train_edges = generate_hyperedge_dict(
            df_train_proc,
            max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
            device=device
        )
        test_edges = generate_hyperedge_dict(
            df_test_proc,
            max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
            device=device
        )
    except TypeError:
        # 如果你的 Credit 实现要求显式 cat_cols
        train_edges = generate_hyperedge_dict(
            df_train_proc, args.cat_cols,
            max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
            device=device
        )
        test_edges = generate_hyperedge_dict(
            df_test_proc, args.cat_cols,
            max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
            device=device
        )

    H_train = build_incidence_matrix(train_edges, len(X_train), device=device)
    H_test = build_incidence_matrix(test_edges, len(X_test), device=device)
    if args.transpose_H:
        H_train = H_train.t().coalesce()
        H_test = H_test.t().coalesce()
    hg_build_time = time.time() - t_hg0

    # 4) Tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_test_t = torch.tensor(y_test, dtype=torch.long, device=device)

    # 5) Train original HGAT
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
        milestones=[args.epochs // 2, args.epochs // 4 * 3],
        gamma=0.1
    )

    print("== Train Full Model ==")
    model, full_train_time = train_model_hgat(
        model, criterion, optimizer, scheduler,
        X_train_t, y_train_t, H_train,
        num_epochs=args.epochs,
        print_freq=args.print_freq
    )

    full_test_acc_raw, full_test_f1_raw = eval_f1_acc(model, X_test_t, y_test_t, H_test)
    print(f"[Full] Raw Test ACC={full_test_acc_raw:.4f} | F1={full_test_f1_raw:.4f} | train_time={full_train_time + hg_build_time:.4f}s")

    # 6) Column deletion on train/test => EditedHG
    if not args.del_cols or len(args.del_cols) == 0:
        raise ValueError("请通过 --del_cols 指定要遗忘的原始列名，例如：--del_cols A4")

    print(f"[Column Unlearning] columns_to_unlearn={args.del_cols}")

    t_edit0 = time.time()
    X_train_edit_t, train_edges_edit, H_train_edit, deleted_idxs, feat_names = delete_feature_columns_hgat_ft(
        X_train_t, transformer, args.del_cols, train_edges, device
    )
    X_test_edit_t, test_edges_edit, H_test_edit, _, _ = delete_feature_columns_hgat_ft(
        X_test_t, transformer, args.del_cols, test_edges, device
    )

    if args.transpose_H:
        H_train_edit = H_train_edit.t().coalesce()
        H_test_edit = H_test_edit.t().coalesce()

    edit_time = time.time() - t_edit0

    print(f"Deleting columns {args.del_cols} -> zeroing encoded dims {deleted_idxs}")
    print(f"[EditedHG] train #hyperedges(orig)={len(train_edges)} -> #hyperedges(edit)={len(train_edges_edit)}")
    print(f"[EditedHG] test  #hyperedges(orig)={len(test_edges)} -> #hyperedges(edit)={len(test_edges_edit)}")

    # Column setting: use edited-train/test acc (train_acc / test_acc)
    full_train_edit_acc = eval_acc(model, X_train_edit_t, y_train_t, H_train_edit)
    full_test_edit_acc, full_test_edit_f1 = eval_f1_acc(model, X_test_edit_t, y_test_t, H_test_edit)
    print(f"[Full@EditedHG] train_acc={full_train_edit_acc:.4f} | test_acc={full_test_edit_acc:.4f} | test_f1={full_test_edit_f1:.4f}")

    rows = []

    # Full@EditedHG
    mia_full = maybe_mia_hgat(model, X_train, y_train, train_edges, args, device)
    print(
        f"Full@EditedHG    K={0:4d} | edit={edit_time:.4f} | update={0.0:.4f} | total={edit_time:.4f} | "
        f"test_acc={full_test_edit_acc:.4f} | train_acc={full_train_edit_acc:.4f} | "
        f"mia_overall={'NA' if mia_full is None else f'{mia_full:.4f}'}"
    )
    rows.append({
        "run": run_id,
        "seed": seed,
        "method": "Full@EditedHG",
        "K": 0,
        "edit_sec": float(edit_time),
        "update_sec": 0.0,
        "total_sec": float(edit_time),
        "test_acc": float(full_test_edit_acc),
        "train_acc": float(full_train_edit_acc),
        "mia_overall": None if mia_full is None else float(mia_full),
    })

    # 7) FT-K on EditedHG
    print("\n== FT-K (warm-start on EditedHG) ==")
    for K in args.ft_steps:
        m = copy.deepcopy(model)
        for p in m.parameters():
            p.requires_grad = True

        m, update_time = finetune_steps(
            m, X_train_edit_t, y_train_t, H_train_edit,
            steps=int(K), lr=args.ft_lr, wd=args.ft_wd
        )

        train_acc = eval_acc(m, X_train_edit_t, y_train_t, H_train_edit)
        test_acc = eval_acc(m, X_test_edit_t, y_test_t, H_test_edit)
        total = edit_time + update_time

        mia = maybe_mia_hgat(m, X_train, y_train, train_edges, args, device)

        print(
            f"FT-K@EditedHG    K={int(K):4d} | edit={edit_time:.4f} | update={update_time:.4f} | total={total:.4f} | "
            f"test_acc={test_acc:.4f} | train_acc={train_acc:.4f} | "
            f"mia_overall={'NA' if mia is None else f'{mia:.4f}'}"
        )

        rows.append({
            "run": run_id,
            "seed": seed,
            "method": "FT-K@EditedHG",
            "K": int(K),
            "edit_sec": float(edit_time),
            "update_sec": float(update_time),
            "total_sec": float(total),
            "test_acc": float(test_acc),
            "train_acc": float(train_acc),
            "mia_overall": None if mia is None else float(mia),
        })

    # 8) FT-head on EditedHG
    print("\n== FT-head (only train last layer) ==")
    for K in args.ft_steps:
        m = copy.deepcopy(model)
        freeze_all_but_head_hgat(m)

        m, update_time = finetune_steps(
            m, X_train_edit_t, y_train_t, H_train_edit,
            steps=int(K), lr=args.ft_lr, wd=args.ft_wd
        )

        train_acc = eval_acc(m, X_train_edit_t, y_train_t, H_train_edit)
        test_acc = eval_acc(m, X_test_edit_t, y_test_t, H_test_edit)
        total = edit_time + update_time

        mia = maybe_mia_hgat(m, X_train, y_train, train_edges, args, device)

        print(
            f"FT-head@EditedHG K={int(K):4d} | edit={edit_time:.4f} | update={update_time:.4f} | total={total:.4f} | "
            f"test_acc={test_acc:.4f} | train_acc={train_acc:.4f} | "
            f"mia_overall={'NA' if mia is None else f'{mia:.4f}'}"
        )

        rows.append({
            "run": run_id,
            "seed": seed,
            "method": "FT-head@EditedHG",
            "K": int(K),
            "edit_sec": float(edit_time),
            "update_sec": float(update_time),
            "total_sec": float(total),
            "test_acc": float(test_acc),
            "train_acc": float(train_acc),
            "mia_overall": None if mia is None else float(mia),
        })

    print("\nDone.")
    return rows


# =========================================================
# Summary
# =========================================================
def summarize_and_save(all_rows, out_csv):
    df = pd.DataFrame(all_rows)
    df.to_csv(out_csv, index=False)

    num_cols = ["edit_sec", "update_sec", "total_sec", "test_acc", "train_acc", "mia_overall"]
    grouped = df.groupby(["method", "K"], dropna=False)[num_cols].agg(["mean", "std"]).reset_index()

    print("\n== Summary (mean±std) ==")

    def _fmt(m, s):
        if pd.isna(m):
            return "NA"
        if pd.isna(s):
            return f"{m:.4f}"
        return f"{m:.4f}±{s:.4f}"

    for _, r in grouped.iterrows():
        method = r[("method", "")]
        K = int(r[("K", "")])

        edit_s = _fmt(r[("edit_sec", "mean")], r[("edit_sec", "std")])
        upd_s = _fmt(r[("update_sec", "mean")], r[("update_sec", "std")])
        tot_s = _fmt(r[("total_sec", "mean")], r[("total_sec", "std")])
        te_s = _fmt(r[("test_acc", "mean")], r[("test_acc", "std")])
        tr_s = _fmt(r[("train_acc", "mean")], r[("train_acc", "std")])
        mo_s = _fmt(r[("mia_overall", "mean")], r[("mia_overall", "std")])

        print(
            f"{method:14s} K={K:4d} | "
            f"edit={edit_s} | update={upd_s} | total={tot_s} | "
            f"test_acc={te_s} | train_acc={tr_s} | mia_overall={mo_s}"
        )

    print(f"\n[Saved] {out_csv}")


# =========================================================
# Args
# =========================================================
def get_args():
    p = argparse.ArgumentParser("Credit HGAT FT baseline (column deletion on edited hypergraph)")

    # Data (Credit single-file)
    p.add_argument("--data_csv", type=str, default="/root/autodl-tmp/TabHGIF/Credit/credit_data/crx.data")
    p.add_argument("--label_col", type=str, default="A16")
    p.add_argument("--split_ratio", type=float, default=0.2)
    p.add_argument("--split_seed", type=int, default=42)

    # Hypergraph
    p.add_argument("--max_nodes_per_hyperedge", type=int, default=50)

    # Column deletion
    p.add_argument("--del_cols", type=str, nargs="+", default=["A4"],
                   help="要遗忘的原始列名，例如: --del_cols A4 或 --del_cols A4 A9")

    # If your Credit generate_hyperedge_dict implementation needs cat_cols explicitly
    p.add_argument("--cat_cols", type=str, nargs="+", default=[])

    # Model / Train
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--print_freq", type=int, default=50)

    # FT
    p.add_argument("--ft_steps", type=int, nargs="+", default=[50, 100, 200])
    p.add_argument("--ft_lr", type=float, default=1e-3)
    p.add_argument("--ft_wd", type=float, default=0.0)

    # HGAT incidence direction
    p.add_argument("--transpose_H", action="store_true", default=False,
                   help="如果你的 HGAT 实现要求 H 形状为 [N,E]，打开此选项")

    # MIA (optional)
    p.add_argument("--run_mia", action="store_true", default=False)
    p.add_argument("--shadow_test_ratio", type=float, default=0.3)
    p.add_argument("--num_shadows", type=int, default=5)
    p.add_argument("--num_attack_samples", type=int, default=None)
    p.add_argument("--shadow_lr", type=float, default=5e-3)
    p.add_argument("--shadow_epochs", type=int, default=100)
    p.add_argument("--attack_test_split", type=float, default=0.3)
    p.add_argument("--attack_lr", type=float, default=1e-2)
    p.add_argument("--attack_epochs", type=int, default=50)

    # Misc
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--out_csv", type=str, default="ft_hgat_col_credit_results.csv")

    return p.parse_args()


def main():
    args = get_args()

    if not os.path.exists(args.data_csv):
        print(f"[WARN] data file not found: {args.data_csv}")

    all_rows = []
    for run_id in range(args.runs):
        print(f"\n================= RUN {run_id + 1}/{args.runs} =================")
        rows = run_one(args, run_id)
        all_rows.extend(rows)

    summarize_and_save(all_rows, args.out_csv)


if __name__ == "__main__":
    main()