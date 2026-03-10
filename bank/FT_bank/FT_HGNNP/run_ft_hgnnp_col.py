#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HGNNP FT baselines on edited hypergraph (BANK, COLUMN unlearning)

Methods:
  - Full@EditedHG    : evaluate original trained full model on edited train/test hypergraph after column deletion
  - FT-K@EditedHG    : warm-start from Full, finetune ALL params for K steps on edited HG
  - FT-head@EditedHG : warm-start from Full, finetune HEAD only for K steps on edited HG

Column unlearning setting:
  - Delete one column (default: education) from both train/test features and hyperedges
  - Finetuning and evaluation are performed on edited data/hypergraph
"""

import os
import time
import copy
import argparse
import random
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# ===== Bank column preprocessing / deletion =====
from bank.HGNNP.data_preprocessing_bank_col import (
    preprocess_node_features_bank,
    generate_hyperedge_dict_bank,
    delete_feature_column,
)

# ===== HGNNP model + helpers =====
from bank.HGNNP.HGNNP import HGNNP_implicit, build_incidence_matrix, compute_degree_vectors

# ===== Optional MIA =====
try:
    from MIA.MIA_utils import membership_inference
from paths import BANK_DATA
    _HAS_MIA = True
except Exception:
    _HAS_MIA = False

# -----------------------------
# Loss (same style as your HGNNP FT scripts)
# -----------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def to_sparse_tensor(H_sp, device):
    Hc = H_sp.tocoo()
    idx = torch.LongTensor(np.vstack((Hc.row, Hc.col)))
    val = torch.FloatTensor(Hc.data)
    return torch.sparse_coo_tensor(idx, val, Hc.shape).coalesce().to(device)

def rebuild_hgnnp_structure_from_hyperedges(hyperedges: dict, num_nodes: int, device):
    """Build HGNNP structure tensors from hyperedge dict."""
    H_sp = build_incidence_matrix(hyperedges, num_nodes)
    dv_inv_np, de_inv_np = compute_degree_vectors(H_sp)
    H_t = to_sparse_tensor(H_sp, device)
    dv_t = torch.tensor(dv_inv_np, dtype=torch.float32, device=device)
    de_t = torch.tensor(de_inv_np, dtype=torch.float32, device=device)
    return H_t, dv_t, de_t

@torch.no_grad()
def eval_acc(model, x, y, H, dv, de) -> float:
    model.eval()
    logits = model(x, H, dv, de)
    pred = logits.argmax(dim=1)
    return float((pred == y).float().mean().item())

def freeze_all_but_head(model: nn.Module):
    """
    Best-effort head-only finetune for HGNNP_implicit.
    Try common names first; fallback to last child.
    """
    for p in model.parameters():
        p.requires_grad = False

    cand_names = ["hgc2", "classifier", "fc", "lin", "out", "head"]
    for name in cand_names:
        if hasattr(model, name):
            m = getattr(model, name)
            if isinstance(m, nn.Module):
                for p in m.parameters():
                    p.requires_grad = True
                return

    children = list(model.children())
    if children:
        for p in children[-1].parameters():
            p.requires_grad = True

def train_full_model(
    model,
    criterion,
    optimizer,
    scheduler,
    x, y, H, dv, de,
    num_epochs=200,
    print_freq=10,
):
    best_state = copy.deepcopy(model.state_dict())
    best_acc = -1.0

    t0 = time.time()
    for ep in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        logits = model(x, H, dv, de)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            pred = logits.argmax(dim=1)
            train_acc = (pred == y).float().mean().item()

        if train_acc > best_acc:
            best_acc = train_acc
            best_state = copy.deepcopy(model.state_dict())

        if (ep == 1) or (ep % print_freq == 0) or (ep == num_epochs):
            print(f"[Train] ep {ep:4d}/{num_epochs} | loss={loss.item():.4f} | train_acc={train_acc:.4f}")

    train_time = time.time() - t0
    model.load_state_dict(best_state)
    print(f"Training complete in {train_time:.2f}s")
    print(f"Best Train Acc: {best_acc:.4f}")
    return model, train_time

def finetune_steps(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    H: torch.Tensor,
    dv: torch.Tensor,
    de: torch.Tensor,
    steps: int,
    lr: float,
    wd: float,
    use_focal: bool = True,
):
    if steps <= 0:
        return model, 0.0

    params = [p for p in model.parameters() if p.requires_grad]
    opt = optim.Adam(params, lr=lr, weight_decay=wd)
    crit = FocalLoss(gamma=2.0, reduction="mean") if use_focal else nn.CrossEntropyLoss()

    t0 = time.time()
    for _ in range(steps):
        model.train()
        opt.zero_grad(set_to_none=True)
        logits = model(x, H, dv, de)
        loss = crit(logits, y)
        loss.backward()
        opt.step()
    dt = time.time() - t0
    return model, dt

def maybe_mia(tag, model, X_tr_np, y_tr_np, hyperedges_edit, args, device):
    """
    Optional MIA. If your current MIA utility doesn't support column-unlearning semantics cleanly,
    you can keep it off (--run_mia False) and fill NA.
    """
    if (not args.run_mia) or (not _HAS_MIA):
        return None
    try:
        print(f"— MIA on {tag} —")
        _, (_, _), (auc_target, _f1_target) = membership_inference(
            X_train=X_tr_np,
            y_train=y_tr_np,
            hyperedges=hyperedges_edit,
            target_model=model,
            args=args,
            device=device
        )
        return float(auc_target)
    except Exception as e:
        print(f"[WARN] MIA failed on {tag}: {e}")
        return None

# -----------------------------
# One run
# -----------------------------
def run_one(args, run_id: int):
    seed = args.seed + run_id
    set_seed(seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device} | seed={seed}")

    # ===== 1) Load BANK csv and split =====
    df = pd.read_csv(args.data_csv, sep=';', header=0)
    assert 'y' in df.columns, "Bank CSV must contain label column 'y'."

    df_train, df_test = train_test_split(
        df,
        test_size=args.split_ratio,
        random_state=42,
        stratify=df['y']
    )
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    print(f"TRAIN samples: {len(df_train)}, TEST samples: {len(df_test)}")
    print("– TRAIN label dist:", Counter(df_train['y']))
    print("– TEST  label dist:", Counter(df_test['y']))

    # ===== 2) Preprocess & build original hypergraph =====
    X_tr, y_tr, df_tr_proc, transformer = preprocess_node_features_bank(df_train, is_test=False)
    X_te, y_te, df_te_proc, _ = preprocess_node_features_bank(df_test, is_test=True, transformer=transformer)

    hyperedges_tr = generate_hyperedge_dict_bank(
        df_tr_proc,
        args.cat_cols,
        args.cont_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        device=device
    )
    hyperedges_te = generate_hyperedge_dict_bank(
        df_te_proc,
        args.cat_cols,
        args.cont_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        device=device
    )

    print(f"一共生成的训练超边总数：{len(hyperedges_tr)}")
    print(f"一共生成的测试超边总数：{len(hyperedges_te)}")

    fts_tr = torch.FloatTensor(X_tr).to(device)
    lbls_tr = torch.LongTensor(y_tr).to(device)
    fts_te = torch.FloatTensor(X_te).to(device)
    lbls_te = torch.LongTensor(y_te).to(device)

    H_tr, dv_tr, de_tr = rebuild_hgnnp_structure_from_hyperedges(hyperedges_tr, fts_tr.shape[0], device)
    H_te, dv_te, de_te = rebuild_hgnnp_structure_from_hyperedges(hyperedges_te, fts_te.shape[0], device)

    # ===== 3) Train Full model on original HG =====
    num_classes = int(np.max(y_tr)) + 1
    model_full = HGNNP_implicit(
        in_ch=fts_tr.shape[1],
        n_class=num_classes,
        n_hid=args.hidden_dim,
        dropout=args.dropout
    ).to(device)

    optimizer = optim.Adam(model_full.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    criterion = FocalLoss(gamma=2.0, reduction="mean")

    print("== Train Full Model ==")
    model_full, full_train_time = train_full_model(
        model_full, criterion, optimizer, scheduler,
        fts_tr, lbls_tr, H_tr, dv_tr, de_tr,
        num_epochs=args.epochs, print_freq=args.print_freq
    )

    full_test_acc = eval_acc(model_full, fts_te, lbls_te, H_te, dv_te, de_te)
    print(f"[Full] Test ACC={full_test_acc:.4f} | train_time={full_train_time:.2f}s")

    # ===== 4) Column unlearning edit (train + test) =====
    col_to_remove = args.columns_to_unlearn[0]
    print(f"[Column Unlearning] columns_to_unlearn={args.columns_to_unlearn}")

    t_edit0 = time.time()

    # train side delete
    fts_tr_edit, _, hyperedges_tr_edit = delete_feature_column(
        fts_tr.clone(), transformer, col_to_remove,
        H_tr, hyperedges_tr,
        continuous_cols=args.cont_cols
    )

    # test side delete
    fts_te_edit, _, hyperedges_te_edit = delete_feature_column(
        fts_te.clone(), transformer, col_to_remove,
        H_te, hyperedges_te,
        continuous_cols=args.cont_cols
    )

    # rebuild HGNNP structures from edited hyperedges
    H_tr_edit, dv_tr_edit, de_tr_edit = rebuild_hgnnp_structure_from_hyperedges(
        hyperedges_tr_edit, fts_tr_edit.shape[0], device
    )
    H_te_edit, dv_te_edit, de_te_edit = rebuild_hgnnp_structure_from_hyperedges(
        hyperedges_te_edit, fts_te_edit.shape[0], device
    )

    edit_time = time.time() - t_edit0

    print(f"[EditedHG] train #hyperedges(orig)={len(hyperedges_tr)} -> #hyperedges(edit)={len(hyperedges_tr_edit)}")
    print(f"[EditedHG] test  #hyperedges(orig)={len(hyperedges_te)} -> #hyperedges(edit)={len(hyperedges_te_edit)}")

    # Full model evaluated on edited HG / edited features
    full_edit_train_acc = eval_acc(model_full, fts_tr_edit, lbls_tr, H_tr_edit, dv_tr_edit, de_tr_edit)
    full_edit_test_acc = eval_acc(model_full, fts_te_edit, lbls_te, H_te_edit, dv_te_edit, de_te_edit)
    print(f"[Full@EditedHG] Train ACC={full_edit_train_acc:.4f} | Test ACC={full_edit_test_acc:.4f}")

    rows = []
    mia_full = maybe_mia("Full@EditedHG", model_full, np.asarray(X_tr), np.asarray(y_tr), hyperedges_tr_edit, args, device)
    rows.append({
        "run": run_id,
        "seed": seed,
        "method": "Full@EditedHG",
        "K": 0,
        "edit_sec": edit_time,
        "update_sec": 0.0,
        "total_sec": edit_time,
        "test_acc": full_edit_test_acc,
        "train_acc": full_edit_train_acc,
        "mia_overall": mia_full,
    })

    # ===== 5) FT-K / FT-head on edited HG =====
    print("\n== FT-K (warm-start on EditedHG) ==")
    for K in args.ft_steps:
        m = copy.deepcopy(model_full)
        for p in m.parameters():
            p.requires_grad = True

        m, ft_update_time = finetune_steps(
            m,
            x=fts_tr_edit, y=lbls_tr,
            H=H_tr_edit, dv=dv_tr_edit, de=de_tr_edit,
            steps=K, lr=args.ft_lr, wd=args.ft_wd,
            use_focal=True
        )

        train_acc = eval_acc(m, fts_tr_edit, lbls_tr, H_tr_edit, dv_tr_edit, de_tr_edit)
        test_acc = eval_acc(m, fts_te_edit, lbls_te, H_te_edit, dv_te_edit, de_te_edit)
        total = edit_time + ft_update_time

        print(f"[FT-K] K={K:4d} | Train ACC={train_acc:.4f} | Test ACC={test_acc:.4f} "
              f"| edit={edit_time:.4f}s | update={ft_update_time:.4f}s | total={total:.4f}s")

        mia_o = maybe_mia(f"FT-K(K={K})", m, np.asarray(X_tr), np.asarray(y_tr), hyperedges_tr_edit, args, device)
        rows.append({
            "run": run_id,
            "seed": seed,
            "method": "FT-K@EditedHG",
            "K": K,
            "edit_sec": edit_time,
            "update_sec": ft_update_time,
            "total_sec": total,
            "test_acc": test_acc,
            "train_acc": train_acc,
            "mia_overall": mia_o,
        })

    print("\n== FT-head (only train last layer) ==")
    for K in args.ft_steps:
        m = copy.deepcopy(model_full)
        freeze_all_but_head(m)

        m, ft_update_time = finetune_steps(
            m,
            x=fts_tr_edit, y=lbls_tr,
            H=H_tr_edit, dv=dv_tr_edit, de=de_tr_edit,
            steps=K, lr=args.ft_lr, wd=args.ft_wd,
            use_focal=True
        )

        train_acc = eval_acc(m, fts_tr_edit, lbls_tr, H_tr_edit, dv_tr_edit, de_tr_edit)
        test_acc = eval_acc(m, fts_te_edit, lbls_te, H_te_edit, dv_te_edit, de_te_edit)
        total = edit_time + ft_update_time

        print(f"[FT-head] K={K:4d} | Train ACC={train_acc:.4f} | Test ACC={test_acc:.4f} "
              f"| edit={edit_time:.4f}s | update={ft_update_time:.4f}s | total={total:.4f}s")

        mia_o = maybe_mia(f"FT-head(K={K})", m, np.asarray(X_tr), np.asarray(y_tr), hyperedges_tr_edit, args, device)
        rows.append({
            "run": run_id,
            "seed": seed,
            "method": "FT-head@EditedHG",
            "K": K,
            "edit_sec": edit_time,
            "update_sec": ft_update_time,
            "total_sec": total,
            "test_acc": test_acc,
            "train_acc": train_acc,
            "mia_overall": mia_o,
        })

    return rows

# -----------------------------
# Summary / Save
# -----------------------------
def summarize_and_save(all_rows, out_csv):
    df = pd.DataFrame(all_rows)
    df.to_csv(out_csv, index=False)

    agg_cols = ["edit_sec", "update_sec", "total_sec", "test_acc", "train_acc", "mia_overall"]
    g = df.groupby(["method", "K"], dropna=False)[agg_cols].agg(["mean", "std"]).reset_index()

    print("\n== Summary (mean±std) ==")

    for _, r in g.iterrows():
        method = r[("method", "")]
        K = int(r[("K", "")])

        def fmt(col):
            m = r[(col, "mean")]
            s = r[(col, "std")]
            if pd.isna(m):
                return "NA"
            if pd.isna(s):
                return f"{m:.4f}"
            return f"{m:.4f}±{s:.4f}"

        print(
            f"{method:14s} K={K:4d} | "
            f"edit={fmt('edit_sec')} | update={fmt('update_sec')} | total={fmt('total_sec')} | "
            f"test_acc={fmt('test_acc')} | train_acc={fmt('train_acc')} | mia_overall={fmt('mia_overall')}"
        )

    print(f"\n[Saved] {out_csv}")

# -----------------------------
# Args
# -----------------------------
def build_parser():
    p = argparse.ArgumentParser("HGNNP FT baseline on edited hypergraph (BANK, column unlearning)")

    # ===== Data =====
    p.add_argument("--data_csv", type=str,
                   default=BANK_DATA)

    p.add_argument("--split_ratio", type=float, default=0.3)

    # ===== Bank hypergraph columns =====
    p.add_argument("--cat_cols", type=str, nargs="+", default=[
        "job", "marital", "education", "default", "housing",
        "loan", "contact", "month", "poutcome"
    ])
    p.add_argument("--cont_cols", type=str, nargs="+", default=[
        "age", "balance", "day", "duration", "campaign", "pdays", "previous"
    ])

    # column to unlearn
    p.add_argument("--columns_to_unlearn", type=str, nargs="+", default=["education"])

    # hyperedge build
    p.add_argument("--max_nodes_per_hyperedge", type=int, default=10000)

    # ===== Model =====
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.1)

    # ===== Full training =====
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--milestones", type=int, nargs="+", default=[100, 150])
    p.add_argument("--gamma", type=float, default=0.1)
    p.add_argument("--print_freq", type=int, default=10)

    # ===== FT =====
    p.add_argument("--ft_steps", type=int, nargs="+", default=[50, 100, 200])
    p.add_argument("--ft_lr", type=float, default=5e-3)
    p.add_argument("--ft_wd", type=float, default=0.0)

    # ===== Multi-run + misc =====
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--run_mia", action="store_true", default=False)
    p.add_argument("--out_csv", type=str, default="ft_hgnnp_bank_col_zero_results.csv")

    return p

def main():
    args = build_parser().parse_args()

    if not os.path.exists(args.data_csv):
        print(f"[WARN] data_csv not found: {args.data_csv}")

    all_rows = []
    for run_id in range(args.runs):
        print(f"\n================= RUN {run_id+1}/{args.runs} =================")
        rows = run_one(args, run_id)
        all_rows.extend(rows)

    summarize_and_save(all_rows, args.out_csv)

if __name__ == "__main__":
    main()