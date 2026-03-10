#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HGAT FT baselines on edited hypergraph (ROW unlearning / node deletion) for Credit dataset.

Methods:
  - Full@EditedHG
  - FT-K@EditedHG
  - FT-head@EditedHG

Credit setting:
  - single file input: crx.data
  - internal train/test split
"""

import os
import time
import copy
import random
import argparse
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# ===== Credit HGAT modules (from your project) =====
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
    _HAS_MIA = False


# -----------------------------
# Basic utils
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_incidence_matrix(hyperedges: dict, num_nodes: int, device=None) -> torch.Tensor:
    """
    Build sparse incidence matrix H [E, N]
    hyperedges: {edge_id: [node_idx, ...]}
    """
    n_edges = len(hyperedges)
    H = torch.zeros((n_edges, num_nodes), dtype=torch.float32, device=device)
    for i, nodes in enumerate(hyperedges.values()):
        if len(nodes) == 0:
            continue
        H[i, torch.as_tensor(nodes, dtype=torch.long, device=device)] = 1.0
    return H.to_sparse()


@torch.no_grad()
def eval_acc_hgat(model, X_t, y_t, H_t) -> float:
    model.eval()
    logits = model(X_t, H_t)
    preds = logits.argmax(dim=1)
    return float((preds == y_t).float().mean().item())


def train_model_hgat(
    model,
    criterion,
    optimizer,
    scheduler,
    X_train_t,
    y_train_t,
    H_train,
    num_epochs=200,
    print_freq=50,
):
    """
    Train HGAT and keep best state by train acc.
    """
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
    Best-effort head-only finetune for HGAT_JK.
    """
    for p in model.parameters():
        p.requires_grad = False

    # Try common names first
    cand_names = ["classifier", "fc", "out", "lin", "proj", "head", "mlp", "hgat2", "att2"]
    for name in cand_names:
        if hasattr(model, name):
            m = getattr(model, name)
            if isinstance(m, nn.Module):
                has_param = False
                for p in m.parameters():
                    p.requires_grad = True
                    has_param = True
                if has_param:
                    print(f"[FT-head] Unfreezing module: {name}")
                    return

    # Fallback to last child with params
    for child in reversed(list(model.children())):
        ps = list(child.parameters())
        if len(ps) > 0:
            for p in ps:
                p.requires_grad = True
            print(f"[FT-head] Fallback unfreezing last child: {child.__class__.__name__}")
            return

    # Ultimate fallback
    for p in model.parameters():
        p.requires_grad = True
    print("[FT-head] Warning: fallback to full-parameter finetune")


def finetune_steps_hgat(
    model,
    X_train_t,
    y_train_t,
    H_train,
    steps: int,
    lr: float,
    wd: float,
):
    if steps <= 0:
        return model, 0.0

    params = [p for p in model.parameters() if p.requires_grad]
    if len(params) == 0:
        raise RuntimeError("No trainable params in finetune_steps_hgat.")

    optimizer = optim.Adam(params, lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()

    t0 = time.time()
    for _ in range(steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(X_train_t, H_train)
        loss = criterion(logits, y_train_t)
        loss.backward()
        optimizer.step()
    dt = time.time() - t0
    return model, dt


# -----------------------------
# Row deletion / edited HG rebuild
# -----------------------------
def rebuild_structure_after_node_deletion(
    X_train_np: np.ndarray,
    y_train_np: np.ndarray,
    train_edges: dict,
    deleted_idx_np: np.ndarray,
    device: torch.device,
):
    """
    Rebuild edited train hypergraph by removing deleted nodes and reindexing retained nodes.
    """
    X_train_np = np.asarray(X_train_np)
    y_train_np = np.asarray(y_train_np)
    deleted_idx_np = np.asarray(deleted_idx_np, dtype=np.int64)

    N = X_train_np.shape[0]
    deleted_set = set(int(i) for i in deleted_idx_np.tolist())
    retain_old_idx = np.array([i for i in range(N) if i not in deleted_set], dtype=np.int64)

    old2new = {int(old): int(new) for new, old in enumerate(retain_old_idx.tolist())}

    edited_edges_list = []
    for _, nodes in train_edges.items():
        new_nodes = [old2new[n] for n in nodes if int(n) in old2new]
        if len(new_nodes) >= 1:   # 与你的 FT 模板一致，保留单节点边
            edited_edges_list.append(new_nodes)

    edited_edges = {eid: nodes for eid, nodes in enumerate(edited_edges_list)}

    X_ret_np = X_train_np[retain_old_idx]
    y_ret_np = y_train_np[retain_old_idx]
    H_edit = build_incidence_matrix(edited_edges, len(retain_old_idx), device=device)

    return (
        X_ret_np, y_ret_np, H_edit, edited_edges,
        retain_old_idx, np.array(sorted(list(deleted_set)), dtype=np.int64), old2new
    )


# -----------------------------
# MIA helpers (optional)
# -----------------------------
def maybe_mia_hgat_overall(
    model,
    X_train_np,
    y_train_np,
    train_edges,
    args,
    device,
    member_mask=None,
):
    if (not args.run_mia) or (not _HAS_MIA):
        return None

    try:
        _, (_auc_s, _f1_s), (auc_t, _f1_t) = membership_inference_hgat(
            X_train_np,
            y_train_np,
            train_edges,
            target_model=model,
            args=args,
            device=device,
            member_mask=member_mask
        )
        return float(auc_t)
    except Exception as e:
        print(f"[WARN] MIA overall failed: {e}")
        return None


def maybe_mia_hgat_deleted(
    model,
    X_train_np,
    y_train_np,
    train_edges,
    deleted_idx_np,
    args,
    device,
):
    if (not args.run_mia) or (not _HAS_MIA):
        return None

    try:
        member_mask = np.zeros(X_train_np.shape[0], dtype=bool)
        member_mask[deleted_idx_np] = True
        _, (_auc_s, _f1_s), (auc_t, _f1_t) = membership_inference_hgat(
            X_train_np,
            y_train_np,
            train_edges,
            target_model=model,
            args=args,
            device=device,
            member_mask=member_mask
        )
        return float(auc_t)
    except Exception as e:
        print(f"[WARN] MIA deleted failed: {e}")
        return None


# -----------------------------
# Data loading (Credit single-file)
# -----------------------------
def load_credit_train_test(args):
    """
    Credit Approval dataset:
      - file has no header
      - 16 features + 1 label (A16)
    """
    col_names = [
        "A1",  "A2",  "A3",  "A4",  "A5",
        "A6",  "A7",  "A8",  "A9",  "A10",
        "A11", "A12", "A13", "A14", "A15",
        "A16",  # label
    ]

    df_full = pd.read_csv(
        args.data_csv,
        header=None,
        names=col_names,
        na_values="?",
        skipinitialspace=True
    )

    label_col = args.label_col
    if label_col not in df_full.columns:
        raise ValueError(f"Label column '{label_col}' not found. Available={list(df_full.columns)}")

    df_train, df_test = train_test_split(
        df_full,
        test_size=args.split_ratio,
        stratify=df_full[label_col],
        random_state=args.split_seed
    )
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    return df_train, df_test


# -----------------------------
# Main run
# -----------------------------
def run_one(args, run_id: int):
    seed = args.seed + run_id
    set_seed(seed)

    if torch.cuda.is_available() and ("cuda" in args.device):
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")
    print(f"[Device] {device} | seed={seed}")

    # ===== Load Credit single-file and split =====
    df_train, df_test = load_credit_train_test(args)
    print(f"Train: {len(df_train)} rows, Test: {len(df_test)} rows")
    print("Train label dist:", Counter(df_train[args.label_col]))
    print("Test  label dist:", Counter(df_test[args.label_col]))

    # ===== Preprocess (fit transformer on train, reuse on test) =====
    X_train, y_train, df_train_proc, transformer = preprocess_node_features(df_train)
    X_test, y_test, df_test_proc, _ = preprocess_node_features(df_test, transformer=transformer)

    # ===== Build hypergraph =====
    t_hg0 = time.time()
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
    H_train = build_incidence_matrix(train_edges, len(X_train), device=device)
    H_test = build_incidence_matrix(test_edges, len(X_test), device=device)
    hg_build_time = time.time() - t_hg0

    # ===== Tensors =====
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_test_t = torch.tensor(y_test, dtype=torch.long, device=device)

    # ===== Train full model =====
    num_classes = int(y_train_t.max().item() + 1)
    model_full = HGAT_JK(
        in_dim=X_train_t.size(1),
        hidden_dim=args.hidden_dim,
        out_dim=num_classes,
        dropout=args.dropout,
        alpha=0.5,
        num_layers=2,
        use_jk=False,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_full.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[args.epochs // 2, args.epochs // 4 * 3],
        gamma=0.1
    )

    print("== Train Full Model ==")
    model_full, full_train_time = train_model_hgat(
        model_full, criterion, optimizer, scheduler,
        X_train_t, y_train_t, H_train,
        num_epochs=args.epochs,
        print_freq=args.print_freq
    )

    full_test_acc = eval_acc_hgat(model_full, X_test_t, y_test_t, H_test)
    print(f"[Full] Test ACC={full_test_acc:.4f} | train_time={full_train_time + hg_build_time:.4f}s")

    # ===== Row deletion =====
    num_train = X_train_t.size(0)
    num_del = max(1, int(args.remove_ratio * num_train))
    perm = torch.randperm(num_train, device=device)
    deleted_idx = perm[:num_del]
    deleted_idx_np = deleted_idx.detach().cpu().numpy()
    deleted_idx_np.sort()

    print(f"[Delete] remove_ratio={args.remove_ratio}, deleted={num_del}/{num_train}")

    t_edit0 = time.time()
    (
        X_ret_np, y_ret_np, H_train_edit, train_edges_edit,
        retain_old_idx_np, deleted_old_idx_np, old2new
    ) = rebuild_structure_after_node_deletion(
        X_train, y_train, train_edges, deleted_idx_np, device
    )
    edit_time = time.time() - t_edit0

    X_ret_t = torch.tensor(X_ret_np, dtype=torch.float32, device=device)
    y_ret_t = torch.tensor(y_ret_np, dtype=torch.long, device=device)

    print(f"[EditedHG] #hyperedges(orig)={len(train_edges)} -> #hyperedges(edit)={len(train_edges_edit)}")

    rows = []

    # ===== Full@EditedHG =====
    retain_acc_full = eval_acc_hgat(model_full, X_ret_t, y_ret_t, H_train_edit)

    with torch.no_grad():
        logits_train_orig = model_full(X_train_t, H_train)
        preds_train_orig = logits_train_orig.argmax(dim=1)
        forget_acc_full = float((preds_train_orig[deleted_idx] == y_train_t[deleted_idx]).float().mean().item())

    mia_overall_full = maybe_mia_hgat_overall(
        model_full, np.asarray(X_train), np.asarray(y_train), train_edges, args, device, member_mask=None
    )
    mia_deleted_full = maybe_mia_hgat_deleted(
        model_full, np.asarray(X_train), np.asarray(y_train), train_edges, deleted_idx_np, args, device
    )

    print(
        f"Full@EditedHG    K={0:4d} | edit={edit_time:.4f} | update={0.0:.4f} | total={edit_time:.4f} | "
        f"test_acc={full_test_acc:.4f} | retain_acc={retain_acc_full:.4f} | forget_acc={forget_acc_full:.4f} | "
        f"mia_overall={'NA' if mia_overall_full is None else f'{mia_overall_full:.4f}'} | "
        f"mia_deleted={'NA' if mia_deleted_full is None else f'{mia_deleted_full:.4f}'}"
    )

    rows.append({
        "run": run_id,
        "seed": seed,
        "method": "Full@EditedHG",
        "K": 0,
        "edit_sec": float(edit_time),
        "update_sec": 0.0,
        "total_sec": float(edit_time),
        "test_acc": float(full_test_acc),
        "retain_acc": float(retain_acc_full),
        "forget_acc": float(forget_acc_full),
        "mia_overall": None if mia_overall_full is None else float(mia_overall_full),
        "mia_deleted": None if mia_deleted_full is None else float(mia_deleted_full),
    })

    # ===== FT-K =====
    print("\n== FT-K (warm-start) ==")
    for K in args.ft_steps:
        m = copy.deepcopy(model_full)
        for p in m.parameters():
            p.requires_grad = True

        m, update_time = finetune_steps_hgat(
            m, X_ret_t, y_ret_t, H_train_edit,
            steps=K, lr=args.ft_lr, wd=args.ft_wd
        )

        test_acc = eval_acc_hgat(m, X_test_t, y_test_t, H_test)
        retain_acc = eval_acc_hgat(m, X_ret_t, y_ret_t, H_train_edit)

        with torch.no_grad():
            logits_train_orig = m(X_train_t, H_train)
            preds_train_orig = logits_train_orig.argmax(dim=1)
            forget_acc = float((preds_train_orig[deleted_idx] == y_train_t[deleted_idx]).float().mean().item())

        mia_overall = maybe_mia_hgat_overall(
            m, np.asarray(X_train), np.asarray(y_train), train_edges, args, device, member_mask=None
        )
        mia_deleted = maybe_mia_hgat_deleted(
            m, np.asarray(X_train), np.asarray(y_train), train_edges, deleted_idx_np, args, device
        )

        total = edit_time + update_time
        print(
            f"FT-K@EditedHG    K={K:4d} | edit={edit_time:.4f} | update={update_time:.4f} | total={total:.4f} | "
            f"test_acc={test_acc:.4f} | retain_acc={retain_acc:.4f} | forget_acc={forget_acc:.4f} | "
            f"mia_overall={'NA' if mia_overall is None else f'{mia_overall:.4f}'} | "
            f"mia_deleted={'NA' if mia_deleted is None else f'{mia_deleted:.4f}'}"
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
            "retain_acc": float(retain_acc),
            "forget_acc": float(forget_acc),
            "mia_overall": None if mia_overall is None else float(mia_overall),
            "mia_deleted": None if mia_deleted is None else float(mia_deleted),
        })

    # ===== FT-head =====
    print("\n== FT-head (only train last layer) ==")
    for K in args.ft_steps:
        m = copy.deepcopy(model_full)
        freeze_all_but_head_hgat(m)

        m, update_time = finetune_steps_hgat(
            m, X_ret_t, y_ret_t, H_train_edit,
            steps=K, lr=args.ft_lr, wd=args.ft_wd
        )

        test_acc = eval_acc_hgat(m, X_test_t, y_test_t, H_test)
        retain_acc = eval_acc_hgat(m, X_ret_t, y_ret_t, H_train_edit)

        with torch.no_grad():
            logits_train_orig = m(X_train_t, H_train)
            preds_train_orig = logits_train_orig.argmax(dim=1)
            forget_acc = float((preds_train_orig[deleted_idx] == y_train_t[deleted_idx]).float().mean().item())

        mia_overall = maybe_mia_hgat_overall(
            m, np.asarray(X_train), np.asarray(y_train), train_edges, args, device, member_mask=None
        )
        mia_deleted = maybe_mia_hgat_deleted(
            m, np.asarray(X_train), np.asarray(y_train), train_edges, deleted_idx_np, args, device
        )

        total = edit_time + update_time
        print(
            f"FT-head@EditedHG K={K:4d} | edit={edit_time:.4f} | update={update_time:.4f} | total={total:.4f} | "
            f"test_acc={test_acc:.4f} | retain_acc={retain_acc:.4f} | forget_acc={forget_acc:.4f} | "
            f"mia_overall={'NA' if mia_overall is None else f'{mia_overall:.4f}'} | "
            f"mia_deleted={'NA' if mia_deleted is None else f'{mia_deleted:.4f}'}"
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
            "retain_acc": float(retain_acc),
            "forget_acc": float(forget_acc),
            "mia_overall": None if mia_overall is None else float(mia_overall),
            "mia_deleted": None if mia_deleted is None else float(mia_deleted),
        })

    print("\nDone.")
    return rows


# -----------------------------
# Summary
# -----------------------------
def summarize_and_save(all_rows, out_csv):
    df = pd.DataFrame(all_rows)
    df.to_csv(out_csv, index=False)

    num_cols = ["edit_sec", "update_sec", "total_sec", "test_acc", "retain_acc", "forget_acc", "mia_overall", "mia_deleted"]
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
        re_s = _fmt(r[("retain_acc", "mean")], r[("retain_acc", "std")])
        fo_s = _fmt(r[("forget_acc", "mean")], r[("forget_acc", "std")])
        mo_s = _fmt(r[("mia_overall", "mean")], r[("mia_overall", "std")])
        md_s = _fmt(r[("mia_deleted", "mean")], r[("mia_deleted", "std")])

        print(
            f"{method:14s} K={K:4d} | "
            f"edit={edit_s} | update={upd_s} | total={tot_s} | "
            f"test_acc={te_s} | retain_acc={re_s} | forget_acc={fo_s} | "
            f"mia_overall={mo_s} | mia_deleted={md_s}"
        )

    print(f"\n[Saved] {out_csv}")


# -----------------------------
# Args
# -----------------------------
def get_args():
    p = argparse.ArgumentParser("HGAT FT baseline (row deletion on edited hypergraph) - Credit")

    # Credit data (single file)
    p.add_argument("--data_csv", type=str, default="/root/autodl-tmp/TabHGIF/Credit/credit_data/crx.data")
    p.add_argument("--label_col", type=str, default="A16")
    p.add_argument("--split_ratio", type=float, default=0.2)
    p.add_argument("--split_seed", type=int, default=42)

    # hypergraph
    p.add_argument("--max_nodes_per_hyperedge", type=int, default=50)

    # model/train
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--print_freq", type=int, default=50)

    # deletion
    p.add_argument("--remove_ratio", type=float, default=0.10)

    # FT
    p.add_argument("--ft_steps", type=int, nargs="+", default=[50, 100, 200])
    p.add_argument("--ft_lr", type=float, default=1e-3)
    p.add_argument("--ft_wd", type=float, default=0.0)

    # MIA (optional)
    p.add_argument("--run_mia", action="store_true", default=True)
    # keep for compatibility with Credit.HGAT.MIA_HGAT
    p.add_argument("--shadow_test_ratio", type=float, default=0.3)
    p.add_argument("--num_shadows", type=int, default=5)
    p.add_argument("--num_attack_samples", type=int, default=None)
    p.add_argument("--shadow_lr", type=float, default=5e-3)
    p.add_argument("--shadow_epochs", type=int, default=100)
    p.add_argument("--attack_test_split", type=float, default=0.3)
    p.add_argument("--attack_lr", type=float, default=1e-2)
    p.add_argument("--attack_epochs", type=int, default=50)

    # misc
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--out_csv", type=str, default="ft_hgat_row_credit_results.csv")

    return p.parse_args()


def main():
    args = get_args()

    if not os.path.exists(args.data_csv):
        print(f"[WARN] data file not found: {args.data_csv}")

    all_rows = []
    for run_id in range(args.runs):
        print(f"\n================= RUN {run_id+1}/{args.runs} =================")
        rows = run_one(args, run_id)
        all_rows.extend(rows)

    summarize_and_save(all_rows, args.out_csv)


if __name__ == "__main__":
    main()