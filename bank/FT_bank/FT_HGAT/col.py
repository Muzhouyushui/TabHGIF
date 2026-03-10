#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bank dataset - HGAT FT baselines on edited hypergraph (COLUMN unlearning), no GIF.

Methods:
  - Full@EditedHG    : evaluate original trained model on edited train/test hypergraph after column deletion
  - FT-K@EditedHG    : warm-start, finetune ALL params on edited train hypergraph for K steps
  - FT-head@EditedHG : warm-start, finetune HEAD only for K steps on edited train hypergraph for K steps

Column unlearning behavior:
  - zero-out encoded dimensions corresponding to deleted raw columns
  - remove hyperedges generated from deleted raw columns
  - rebuild H for train/test on edited hypergraphs

This version is adapted for BANK dataset (single CSV split internally).
"""

import os
import time
import copy
import random
import argparse
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# ====== Bank HGAT pipeline imports (adjust these paths to your project) ======
from bank.HGAT.data_preprocessing_bank_col import (
    preprocess_node_features_bank,
    generate_hyperedge_dict_bank,
)
from bank.HGAT.HGAT_new import HGAT_JK

# optional MIA (if you have a bank-HGAT MIA implementation)
try:
    from bank.HGAT.MIA_HGAT import membership_inference_hgat_bank
    from paths import BANK_DATA
    _HAS_MIA = True
except Exception:
    _HAS_MIA = False
# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_incidence_matrix(hyperedges: dict, num_nodes: int, device=None) -> torch.Tensor:
    n_edges = len(hyperedges)
    H = torch.zeros((n_edges, num_nodes), dtype=torch.float32, device=device)
    for i, nodes in enumerate(hyperedges.values()):
        if len(nodes) == 0:
            continue
        H[i, torch.as_tensor(nodes, dtype=torch.long, device=device)] = 1.0
    return H.to_sparse()

def get_feature_names_safe(transformer):
    try:
        return list(transformer.get_feature_names_out())
    except Exception:
        return list(transformer.get_feature_names())

def find_encoded_dims_for_raw_cols(transformer, del_cols):
    """
    Match encoded dimensions generated from raw columns.
    Compatible with names like:
      cat__job_admin.
      job_admin.
      num__age
    """
    feat_names = get_feature_names_safe(transformer)
    deleted_idxs = []

    for col in del_cols:
        matches = []
        for i, fn in enumerate(feat_names):
            fn_norm = fn.replace("=", "_").replace("-", "_").replace(".", "_")
            tokens = [seg for seg in fn_norm.split("_") if seg]
            if (col in tokens) or (f"__{col}_" in fn) or (fn.startswith(col + "_")) or (f"__{col}" in fn):
                matches.append(i)

        if len(matches) == 0:
            # final fallback: substring contains
            matches = [i for i, fn in enumerate(feat_names) if col in fn]

        if len(matches) == 0:
            raise ValueError(f"在编码后特征中未找到原始列 '{col}'")

        deleted_idxs.extend(matches)

    deleted_idxs = sorted(set(deleted_idxs))
    return deleted_idxs, feat_names

def remove_hyperedges_by_deleted_columns(hyperedges: dict, deleted_names):
    """
    If hyperedge dict keys contain column names (string keys), remove edges whose key matches deleted cols.
    If keys are integer-only, this function cannot infer provenance and will keep all edges.
    For bank pipeline, generate_hyperedge_dict_bank often can be adapted to preserve semantic keys.
    """
    if len(hyperedges) == 0:
        return {}

    first_key = next(iter(hyperedges.keys()))

    # Case 1: semantic string keys -> can safely remove by column name matching
    if isinstance(first_key, str):
        new_edges = {}
        eid = 0
        for k, nodes in hyperedges.items():
            hit = False
            for col in deleted_names:
                if col in k:
                    hit = True
                    break
            if (not hit) and (len(nodes) > 0):
                new_edges[eid] = list(nodes)
                eid += 1
        return new_edges

    # Case 2: integer keys only -> cannot infer edge-column mapping from key
    # keep all edges (feature-zero only)
    # NOTE: if your bank unlearning code has exact delete_feature_columns_hgat() logic, replace this.
    return {i: list(v) for i, v in enumerate(hyperedges.values())}

def delete_feature_columns_hgat_ft(X_t, transformer, deleted_names, hyperedges, device):
    """
    FT version of column deletion:
      1) zero corresponding encoded dimensions
      2) remove hyperedges associated with deleted raw columns when key names allow it
      3) rebuild incidence matrix
    """
    X_u = X_t.clone()
    deleted_idxs, _ = find_encoded_dims_for_raw_cols(transformer, deleted_names)
    if len(deleted_idxs) > 0:
        X_u[:, torch.as_tensor(deleted_idxs, dtype=torch.long, device=device)] = 0.0

    edges_u = remove_hyperedges_by_deleted_columns(hyperedges, deleted_names)
    H_u = build_incidence_matrix(edges_u, X_u.size(0), device=device)
    return X_u, edges_u, H_u, deleted_idxs

@torch.no_grad()
def eval_acc(model, X_t, y_t, H_t):
    model.eval()
    logits = model(X_t, H_t)
    preds = logits.argmax(dim=1)
    return float((preds == y_t).float().mean().item())

@torch.no_grad()
def eval_f1_acc(model, X_t, y_t, H_t):
    model.eval()
    logits = model(X_t, H_t)
    preds = logits.argmax(dim=1)
    acc = accuracy_score(y_t.detach().cpu().numpy(), preds.detach().cpu().numpy())
    f1 = f1_score(y_t.detach().cpu().numpy(), preds.detach().cpu().numpy(), average="micro")
    return float(acc), float(f1)

def train_model_hgat(model, criterion, optimizer, scheduler, X_train_t, y_train_t, H_train,
                     num_epochs=130, print_freq=20):
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
    for p in model.parameters():
        p.requires_grad = False

    # Try common head names first
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

    # Fallback: unfreeze last child module
    for child in reversed(list(model.children())):
        ps = list(child.parameters())
        if len(ps) > 0:
            for p in ps:
                p.requires_grad = True
            print(f"[FT-head] Fallback unfreezing last child: {child.__class__.__name__}")
            return

    # Final fallback
    for p in model.parameters():
        p.requires_grad = True
    print("[FT-head] Warning: fallback to full-parameter finetune")

def finetune_steps(model, X_t, y_t, H_t, steps, lr, wd):
    if steps <= 0:
        return model, 0.0

    params = [p for p in model.parameters() if p.requires_grad]
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

def maybe_mia_hgat_bank(model, X_np, y_np, edges, args, device):
    if (not args.run_mia) or (not _HAS_MIA):
        return None
    try:
        # Please adapt this call signature to your bank MIA implementation if needed
        _, (_auc_s, _f1_s), (auc_t, _f1_t) = membership_inference_hgat_bank(
            X_np, y_np, edges, target_model=model, args=args, device=device
        )
        return float(auc_t)
    except Exception as e:
        print(f"[WARN] MIA failed: {e}")
        return None

# -----------------------------
# Main run
# -----------------------------
def run_one(args, run_id):
    seed = args.seed + run_id
    set_seed(seed)

    device = torch.device(args.device if (torch.cuda.is_available() and "cuda" in args.device) else "cpu")
    print(f"[Device] {device}")

    # 1) Load BANK CSV and split
    df_full = pd.read_csv(args.data_csv, sep=';', skipinitialspace=True)
    df_train, df_test = train_test_split(
        df_full,
        test_size=args.split_ratio,
        stratify=df_full[args.label_col],
        random_state=args.split_seed
    )
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    # 2) Preprocess (bank-specific)
    X_train, y_train, df_train_proc, transformer = preprocess_node_features_bank(df_train, is_test=False)
    X_test, y_test, df_test_proc, _ = preprocess_node_features_bank(df_test, is_test=True, transformer=transformer)

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train, dtype=np.int64)
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test, dtype=np.int64)

    c_tr = Counter(y_train.tolist())
    c_te = Counter(y_test.tolist())
    print(f"标签分布(train) -> class0: {c_tr.get(0,0)}，class1: {c_tr.get(1,0)}")
    print(f"标签分布(test)  -> class0: {c_te.get(0,0)}，class1: {c_te.get(1,0)}")

    # 3) Build original hypergraph
    t_hg0 = time.time()
    train_edges = generate_hyperedge_dict_bank(
        df_train_proc,
        args.cat_cols,
        args.cont_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        device=device
    )
    test_edges = generate_hyperedge_dict_bank(
        df_test_proc,
        args.cat_cols,
        args.cont_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        device=device
    )

    H_train = build_incidence_matrix(train_edges, len(X_train), device=device)
    H_test = build_incidence_matrix(test_edges, len(X_test), device=device)

    # IMPORTANT: if your HGAT implementation expects H=[N,E], enable transpose_H
    if args.transpose_H:
        H_train = H_train.t().coalesce()
        H_test = H_test.t().coalesce()

    hg_build_time = time.time() - t_hg0

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_test_t = torch.tensor(y_test, dtype=torch.long, device=device)

    # 4) Train original HGAT
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
        gamma=args.gamma
    )

    print("== Train Full Model ==")
    model, full_train_time = train_model_hgat(
        model, criterion, optimizer, scheduler,
        X_train_t, y_train_t, H_train,
        num_epochs=args.epochs, print_freq=args.print_freq
    )

    full_test_acc, full_test_f1 = eval_f1_acc(model, X_test_t, y_test_t, H_test)
    print(f"[Full] Test ACC={full_test_acc:.4f} | F1={full_test_f1:.4f} | train_time={full_train_time + hg_build_time:.2f}s")

    # 5) Column deletion on train/test (edited hypergraph)
    if not args.del_cols:
        raise ValueError("请通过 --del_cols 指定要遗忘的列名列表，例如 --del_cols education")

    print(f"[Column Unlearning] columns_to_unlearn={args.del_cols}")

    t_edit0 = time.time()
    X_train_edit_t, train_edges_edit, H_train_edit, deleted_idxs = delete_feature_columns_hgat_ft(
        X_train_t, transformer, args.del_cols, train_edges, device
    )
    X_test_edit_t, test_edges_edit, H_test_edit, _ = delete_feature_columns_hgat_ft(
        X_test_t, transformer, args.del_cols, test_edges, device
    )

    if args.transpose_H:
        H_train_edit = H_train_edit.t().coalesce()
        H_test_edit = H_test_edit.t().coalesce()

    edit_time = time.time() - t_edit0

    print(f"Deleting columns {args.del_cols} -> zeroing dims {deleted_idxs}")
    print(f"[EditedHG] train #hyperedges(orig)={len(train_edges)} -> #hyperedges(edit)={len(train_edges_edit)}")
    print(f"[EditedHG] test  #hyperedges(orig)={len(test_edges)} -> #hyperedges(edit)={len(test_edges_edit)}")

    full_train_edit_acc = eval_acc(model, X_train_edit_t, y_train_t, H_train_edit)
    full_test_edit_acc = eval_acc(model, X_test_edit_t, y_test_t, H_test_edit)
    print(f"[Full@EditedHG] Train ACC={full_train_edit_acc:.4f} | Test ACC={full_test_edit_acc:.4f}")

    rows = []
    mia_full = maybe_mia_hgat_bank(model, X_train, y_train, train_edges, args, device)
    rows.append({
        "run": run_id,
        "seed": seed,
        "method": "Full@EditedHG",
        "K": 0,
        "edit_sec": edit_time,
        "update_sec": 0.0,
        "total_sec": edit_time,
        "test_acc": full_test_edit_acc,
        "train_acc": full_train_edit_acc,
        "mia_overall": mia_full,
    })

    # 6) FT-K on EditedHG
    print("\n== FT-K (warm-start on EditedHG) ==")
    for K in args.ft_steps:
        m = copy.deepcopy(model)
        for p in m.parameters():
            p.requires_grad = True

        m, update_time = finetune_steps(
            m, X_train_edit_t, y_train_t, H_train_edit,
            steps=K, lr=args.ft_lr, wd=args.ft_wd
        )

        train_acc = eval_acc(m, X_train_edit_t, y_train_t, H_train_edit)
        test_acc = eval_acc(m, X_test_edit_t, y_test_t, H_test_edit)
        total = edit_time + update_time
        print(f"[FT-K] K={K:4d} | Train ACC={train_acc:.4f} | Test ACC={test_acc:.4f} | "
              f"edit={edit_time:.4f}s | update={update_time:.4f}s | total={total:.4f}s")

        mia = maybe_mia_hgat_bank(m, X_train, y_train, train_edges, args, device)
        rows.append({
            "run": run_id,
            "seed": seed,
            "method": "FT-K@EditedHG",
            "K": K,
            "edit_sec": edit_time,
            "update_sec": update_time,
            "total_sec": total,
            "test_acc": test_acc,
            "train_acc": train_acc,
            "mia_overall": mia,
        })

    # 7) FT-head on EditedHG
    print("\n== FT-head (only train last layer) ==")
    for K in args.ft_steps:
        m = copy.deepcopy(model)
        freeze_all_but_head_hgat(m)

        m, update_time = finetune_steps(
            m, X_train_edit_t, y_train_t, H_train_edit,
            steps=K, lr=args.ft_lr, wd=args.ft_wd
        )

        train_acc = eval_acc(m, X_train_edit_t, y_train_t, H_train_edit)
        test_acc = eval_acc(m, X_test_edit_t, y_test_t, H_test_edit)
        total = edit_time + update_time
        print(f"[FT-head] K={K:4d} | Train ACC={train_acc:.4f} | Test ACC={test_acc:.4f} | "
              f"edit={edit_time:.4f}s | update={update_time:.4f}s | total={total:.4f}s")

        mia = maybe_mia_hgat_bank(m, X_train, y_train, train_edges, args, device)
        rows.append({
            "run": run_id,
            "seed": seed,
            "method": "FT-head@EditedHG",
            "K": K,
            "edit_sec": edit_time,
            "update_sec": update_time,
            "total_sec": total,
            "test_acc": test_acc,
            "train_acc": train_acc,
            "mia_overall": mia,
        })

    print("\nDone.")
    return rows

# -----------------------------
# Summary
# -----------------------------
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

# -----------------------------
# Args
# -----------------------------
def get_args():
    p = argparse.ArgumentParser("Bank HGAT FT baseline (column deletion on edited hypergraph)")

    p.add_argument("--data_csv", type=str,
                   default=BANK_DATA)
    p.add_argument("--split_ratio", type=float, default=0.3)
    p.add_argument("--split_seed", type=int, default=21)
    p.add_argument("--label_col", type=str, default="y")

    p.add_argument("--cat_cols", type=str, nargs="+", default=[
        "job", "marital", "education", "default", "housing", "loan",
        "contact", "month", "poutcome"
    ])
    p.add_argument("--cont_cols", type=str, nargs="+", default=[
        "age", "balance", "day", "duration", "campaign", "pdays", "previous"
    ])

    p.add_argument("--max_nodes_per_hyperedge", type=int, default=10000)

    p.add_argument("--del_cols", type=str, nargs="+", default=["education"],
                   help="Original column names to unlearn, e.g.: --del_cols education")

    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=130)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--gamma", type=float, default=0.1)
    p.add_argument("--print_freq", type=int, default=20)

    p.add_argument("--ft_steps", type=int, nargs="+", default=[50, 100, 200])
    p.add_argument("--ft_lr", type=float, default=1e-3)
    p.add_argument("--ft_wd", type=float, default=0.0)

    p.add_argument("--transpose_H", action="store_true", default=False,
                   help="Enable this if the HGAT implementation requires H to be [N, E]")

    p.add_argument("--run_mia", action="store_true", default=False)
    p.add_argument("--shadow_test_ratio", type=float, default=0.3)
    p.add_argument("--num_shadows", type=int, default=5)
    p.add_argument("--num_attack_samples", type=int, default=None)
    p.add_argument("--shadow_lr", type=float, default=5e-3)
    p.add_argument("--shadow_epochs", type=int, default=100)
    p.add_argument("--attack_test_split", type=float, default=0.3)
    p.add_argument("--attack_lr", type=float, default=1e-2)
    p.add_argument("--attack_epochs", type=int, default=50)

    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--out_csv", type=str, default="ft_hgat_bank_col_zero_results.csv")

def main():
    args = get_args()
    all_rows = []

    for run_id in range(args.runs):
        print(f"\n================= RUN {run_id+1}/{args.runs} =================")
        rows = run_one(args, run_id)
        all_rows.extend(rows)

    summarize_and_save(all_rows, args.out_csv)

if __name__ == "__main__":
    main()