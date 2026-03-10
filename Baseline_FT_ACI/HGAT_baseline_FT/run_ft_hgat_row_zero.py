#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HGAT FT baselines on edited hypergraph (ROW unlearning / node deletion), no GIF.

Methods:
  - Full@EditedHG    : evaluate original trained model on edited train hypergraph (retain/forget)
  - FT-K@EditedHG    : warm-start, finetune ALL params on edited train hypergraph for K steps
  - FT-head@EditedHG : warm-start, finetune HEAD only for K steps on edited train hypergraph

Notes:
  - Test graph is kept unchanged (same as many row-unlearning evaluations in your pipeline).
  - Edited train hypergraph is rebuilt by removing deleted nodes and reindexing retained nodes.
"""

import os
import time
import copy
import random
import argparse
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

# ===== Data tools (same as your HGAT row script) =====
from data_preprocessing.data_preprocessing_K import (
    preprocess_node_features,
    generate_hyperedge_dict,
)

# ===== HGAT model =====
from HGAT.HGAT_new import HGAT_JK

# ===== MIA (optional, same family as your HGAT row script) =====
try:
    from MIA.MIA_HGAT import membership_inference_hgat
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
    Dense->sparse incidence matrix [E, N], same style as your HGAT script.
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
    num_epochs=130,
    print_freq=20,
):
    """
    Train HGAT and keep best state by train acc (matching your style).
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
    Try common classifier names first; fallback to last child module.
    """
    for p in model.parameters():
        p.requires_grad = False

    # common names in graph attention implementations
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

    # fallback: last child with parameters
    for child in reversed(list(model.children())):
        ps = list(child.parameters())
        if len(ps) > 0:
            for p in ps:
                p.requires_grad = True
            print(f"[FT-head] Fallback unfreezing last child: {child.__class__.__name__}")
            return

    # ultimate fallback: if no child modules found, unfreeze all (avoid dead training)
    for p in model.parameters():
        p.requires_grad = True
    print("[FT-head] Warning: fallback to full-parameter finetune (no identifiable head module)")


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
    # --- add these lines ---
    X_train_np = np.asarray(X_train_np)
    y_train_np = np.asarray(y_train_np)
    deleted_idx_np = np.asarray(deleted_idx_np, dtype=np.int64)
    # -----------------------

    N = X_train_np.shape[0]
    deleted_set = set(int(i) for i in deleted_idx_np.tolist())
    retain_old_idx = np.array([i for i in range(N) if i not in deleted_set], dtype=np.int64)

    old2new = {int(old): int(new) for new, old in enumerate(retain_old_idx.tolist())}

    edited_edges_list = []
    for _, nodes in train_edges.items():
        new_nodes = [old2new[n] for n in nodes if int(n) in old2new]
        if len(new_nodes) >= 1:
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
        return None, None

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
        # If member_mask is None -> overall MIA (depends on your impl)
        return float(auc_t), None
    except Exception as e:
        print(f"[WARN] MIA failed: {e}")
        return None, None


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
        member_mask[deleted_idx_np] = True  # deleted nodes as “member” positives (your prior usage)
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
        print(f"[WARN] Deleted-node MIA failed: {e}")
        return None


# -----------------------------
# Main run
# -----------------------------
def run_one(args, run_id: int):
    seed = args.seed + run_id
    set_seed(seed)
    device = torch.device(args.device if (torch.cuda.is_available() and "cuda" in args.device) else "cpu")
    print(f"[Device] {device}")

    # ===== Data preprocess =====
    X_train, y_train, df_train, transformer = preprocess_node_features(args.train_csv, is_test=False)
    X_test, y_test, df_test, _ = preprocess_node_features(args.test_csv, is_test=True, transformer=transformer)

    c_tr = Counter(y_train)
    c_te = Counter(y_test)
    print(f"标签分布 → <=50K: {c_tr.get(0,0)}，>50K: {c_tr.get(1,0)}")
    print(f"标签分布 → <=50K: {c_te.get(0,0)}，>50K: {c_te.get(1,0)}")

    # ===== Build hypergraph =====
    t_hg0 = time.time()
    train_edges = generate_hyperedge_dict(
        df_train, args.cat_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        device=device
    )
    test_edges = generate_hyperedge_dict(
        df_test, args.cat_cols,
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
    print(f"[Full] Test ACC={full_test_acc:.4f} | train_time={full_train_time + hg_build_time:.2f}s")

    # ===== Row deletion =====
    num_train = X_train_t.size(0)
    num_del = int(args.remove_ratio * num_train)
    perm = torch.randperm(num_train, device=device)
    deleted_idx = perm[:num_del]
    deleted_idx_np = deleted_idx.detach().cpu().numpy()
    deleted_idx_np.sort()

    print(f"[Delete] remove_ratio={args.remove_ratio}, deleted={num_del}")

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

    # ===== Full@EditedHG =====
    retain_acc_full = eval_acc_hgat(model_full, X_ret_t, y_ret_t, H_train_edit)

    # deleted accuracy on original graph (before any FT), for reference
    with torch.no_grad():
        logits_train_orig = model_full(X_train_t, H_train)
        preds_train_orig = logits_train_orig.argmax(dim=1)
        forget_acc_full = float(
            (preds_train_orig[deleted_idx] == y_train_t[deleted_idx]).float().mean().item()
        )

    print(f"[Full@EditedHG] Retain ACC={retain_acc_full:.4f} | Forget ACC={forget_acc_full:.4f}")

    rows = []

    mia_overall_full, _ = maybe_mia_hgat_overall(
        model_full, np.asarray(X_train), np.asarray(y_train), train_edges, args, device, member_mask=None
    )
    mia_deleted_full = maybe_mia_hgat_deleted(
        model_full, np.asarray(X_train), np.asarray(y_train), train_edges, deleted_idx_np, args, device
    )

    rows.append({
        "run": run_id,
        "seed": seed,
        "method": "Full@EditedHG",
        "K": 0,
        "edit_sec": edit_time,
        "update_sec": 0.0,
        "total_sec": edit_time,
        "test_acc": full_test_acc,
        "retain_acc": retain_acc_full,
        "forget_acc": forget_acc_full,
        "mia_overall": mia_overall_full,
        "mia_deleted": mia_deleted_full,
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
            forget_acc = float(
                (preds_train_orig[deleted_idx] == y_train_t[deleted_idx]).float().mean().item()
            )

        total = edit_time + update_time
        print(f"[FT-K] K={K:4d} | Test ACC={test_acc:.4f} | Retain ACC={retain_acc:.4f} | "
              f"Forget ACC={forget_acc:.4f} | time={total:.2f}s")

        mia_overall = None
        mia_deleted = None
        if args.run_mia:
            mia_overall, _ = maybe_mia_hgat_overall(
                m, np.asarray(X_train), np.asarray(y_train), train_edges, args, device, member_mask=None
            )
            mia_deleted = maybe_mia_hgat_deleted(
                m, np.asarray(X_train), np.asarray(y_train), train_edges, deleted_idx_np, args, device
            )

        rows.append({
            "run": run_id,
            "seed": seed,
            "method": "FT-K@EditedHG",
            "K": K,
            "edit_sec": edit_time,
            "update_sec": update_time,
            "total_sec": total,
            "test_acc": test_acc,
            "retain_acc": retain_acc,
            "forget_acc": forget_acc,
            "mia_overall": mia_overall,
            "mia_deleted": mia_deleted,
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
            forget_acc = float(
                (preds_train_orig[deleted_idx] == y_train_t[deleted_idx]).float().mean().item()
            )

        total = edit_time + update_time
        print(f"[FT-head] K={K:4d} | Test ACC={test_acc:.4f} | Retain ACC={retain_acc:.4f} | "
              f"Forget ACC={forget_acc:.4f} | time={total:.2f}s")

        mia_overall = None
        mia_deleted = None
        if args.run_mia:
            mia_overall, _ = maybe_mia_hgat_overall(
                m, np.asarray(X_train), np.asarray(y_train), train_edges, args, device, member_mask=None
            )
            mia_deleted = maybe_mia_hgat_deleted(
                m, np.asarray(X_train), np.asarray(y_train), train_edges, deleted_idx_np, args, device
            )

        rows.append({
            "run": run_id,
            "seed": seed,
            "method": "FT-head@EditedHG",
            "K": K,
            "edit_sec": edit_time,
            "update_sec": update_time,
            "total_sec": total,
            "test_acc": test_acc,
            "retain_acc": retain_acc,
            "forget_acc": forget_acc,
            "mia_overall": mia_overall,
            "mia_deleted": mia_deleted,
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
    p = argparse.ArgumentParser("HGAT FT baseline (row deletion on edited hypergraph)")

    # data
    p.add_argument("--train_csv", type=str, default="/root/autodl-tmp/TabHGIF/data/adult.data")
    p.add_argument("--test_csv", type=str, default="/root/autodl-tmp/TabHGIF/data/adult.test")
    p.add_argument("--cat_cols", type=str, nargs="+", default=[
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ])
    p.add_argument("--max_nodes_per_hyperedge", type=int, default=10000)

    # model/train
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=130)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--print_freq", type=int, default=20)

    # deletion
    p.add_argument("--remove_ratio", type=float, default=0.30)

    # FT
    p.add_argument("--ft_steps", type=int, nargs="+", default=[50, 100, 200])
    p.add_argument("--ft_lr", type=float, default=1e-3)
    p.add_argument("--ft_wd", type=float, default=0.0)

    # MIA (optional)
    p.add_argument("--run_mia", action="store_true", default=False)
    # keep these for compatibility with your MIA_HGAT if needed
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
    p.add_argument("--out_csv", type=str, default="ft_hgat_row_zero_results.csv")

    return p.parse_args()


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