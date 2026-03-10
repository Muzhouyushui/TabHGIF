#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Credit + HGNN + Row Unlearning FT baselines (Edited Hypergraph)

Methods:
  - Full@EditedHG
  - FT-K@EditedHG
  - FT-head@EditedHG

Dataset:
  - Credit Approval (single file: crx.data)
  - internal train/test split

Output style:
  edit / update / total / test_acc / retain_acc / forget_acc / mia_overall / mia_deleted
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
from sklearn.model_selection import train_test_split

# ===== Credit HGNN modules =====
from Credit.HGNN.data_preprocessing_Credit import (
    preprocess_node_features,
    generate_hyperedge_dict,
)
from Credit.HGNN.HGNN import (
    HGNN_implicit,
    build_incidence_matrix,
    compute_degree_vectors,
)
from Credit.HGNN.GIF_HGNN_ROW_Credit import (
    rebuild_structure_after_node_deletion,   # expect: (hyperedges, deleted_idx, N, device) -> H,dv,de,edited_edges
    train_model,                             # expect HGNN train function
)

from Credit.HGNN.HGNN_utils import evaluate_test_acc, evaluate_test_f1

# Optional MIA
try:
    from Credit.HGNN.MIA_HGNN import membership_inference
from paths import CREDIT_DATA
    _HAS_MIA = True
except Exception:
    membership_inference = None
    _HAS_MIA = False

# =========================================================
# Utils
# =========================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device(device_str: str):
    if torch.cuda.is_available() and ("cuda" in device_str):
        return torch.device(device_str)
    return torch.device("cpu")

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(
            logits, targets, weight=self.weight, reduction="none"
        )
        pt = torch.exp(-ce_loss)
        loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

def build_masks(N: int, deleted_idx_np: np.ndarray, device: torch.device):
    del_mask = torch.zeros(N, dtype=torch.bool, device=device)
    if len(deleted_idx_np) > 0:
        del_mask[torch.as_tensor(deleted_idx_np, dtype=torch.long, device=device)] = True
    retain_mask = ~del_mask
    return retain_mask, del_mask

def to_hgnn_sparse_tensors_from_hyperedges(hyperedges: dict, num_nodes: int, device: torch.device):
    """
    Build H, dv_inv, de_inv tensors for HGNN from hyperedge dict.
    """
    H_sp = build_incidence_matrix(hyperedges, num_nodes)  # scipy sparse expected
    dv_np, de_np = compute_degree_vectors(H_sp)

    H_coo = H_sp.tocoo()
    idx = torch.LongTensor(np.vstack((H_coo.row, H_coo.col))).to(device)
    val = torch.FloatTensor(H_coo.data).to(device)
    H_t = torch.sparse_coo_tensor(idx, val, H_coo.shape).coalesce()

    dv_t = torch.FloatTensor(dv_np).to(device)
    de_t = torch.FloatTensor(de_np).to(device)
    return H_t, dv_t, de_t

@torch.no_grad()
def eval_acc_masked(model, x, y, mask, H, dv_inv, de_inv) -> float:
    model.eval()
    out = model(x, H, dv_inv, de_inv)   # log_softmax for HGNN impl (usually)
    pred = out.argmax(dim=1)
    return float((pred[mask] == y[mask]).float().mean().item())

def freeze_all_but_head_hgnn(model: nn.Module):
    """
    HGNN_implicit usually has hgc1 / hgc2. Treat hgc2 as head.
    """
    for p in model.parameters():
        p.requires_grad = False

    if hasattr(model, "hgc2"):
        for p in model.hgc2.parameters():
            p.requires_grad = True
        print("[FT-head] Unfreezing model.hgc2")
        return

    # Fallback: unfreeze last child
    children = list(model.children())
    if children:
        for p in children[-1].parameters():
            p.requires_grad = True
        print(f"[FT-head] Fallback unfreezing last child: {children[-1].__class__.__name__}")
        return

    # last fallback
    for p in model.parameters():
        p.requires_grad = True
    print("[FT-head] Warning: fallback to full-parameter finetune")

def finetune_steps_hgnn(
    model: nn.Module,
    x_after: torch.Tensor,      # zeroed deleted rows for FT input
    y: torch.Tensor,
    retain_mask: torch.Tensor,
    H_edit: torch.Tensor,
    dv_edit: torch.Tensor,
    de_edit: torch.Tensor,
    steps: int,
    lr: float,
    weight_decay: float,
):
    """
    Retain-only finetune for K steps on EditedHG.
    """
    if steps <= 0:
        return model, 0.0

    params = [p for p in model.parameters() if p.requires_grad]
    if len(params) == 0:
        raise RuntimeError("No trainable parameters in finetune_steps_hgnn.")

    model.train()
    opt = optim.Adam(params, lr=lr, weight_decay=weight_decay)
    loss_fn = nn.NLLLoss()

    t0 = time.time()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        out = model(x_after, H_edit, dv_edit, de_edit)  # log_softmax
        loss = loss_fn(out[retain_mask], y[retain_mask])
        loss.backward()
        opt.step()
    dt = time.time() - t0
    return model, dt

def maybe_mia_overall(model, X_np, y_np, hyperedges, args, device):
    if (not args.run_mia) or (not _HAS_MIA):
        return None
    try:
        _, (_, _), (auc_target, _f1_target) = membership_inference(
            X_train=X_np,
            y_train=y_np,
            hyperedges=hyperedges,
            target_model=model,
            args=args,
            device=device
        )
        return float(auc_target)
    except Exception as e:
        print(f"[WARN] MIA overall failed: {e}")
        return None

def maybe_mia_deleted(model, X_np, y_np, hyperedges_keepdel, member_mask, args, device):
    """
    Keep/Del MIA setting: member_mask=True for retained nodes, False for deleted nodes (or vice versa, depends on your tool).
    This matches your existing scripts' style.
    """
    if (not args.run_mia) or (not _HAS_MIA):
        return None
    try:
        _, (_, _), (auc_target, _f1_target) = membership_inference(
            X_train=X_np,
            y_train=y_np,
            hyperedges=hyperedges_keepdel,
            target_model=model,
            args=args,
            device=device,
            member_mask=member_mask
        )
        return float(auc_target)
    except Exception as e:
        print(f"[WARN] MIA deleted failed: {e}")
        return None

# =========================================================
# Data loading (Credit single-file)
# =========================================================
def load_credit_df(args):
    """
    Credit Approval dataset: crx.data
    Original 16 columns: A1..A15 + class
    Add y in {0,1}
    """
    df = pd.read_csv(
        args.data_csv,
        header=None,
        na_values="?",
        skipinitialspace=True
    )
    df.columns = [
        "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8",
        "A9", "A10", "A11", "A12", "A13", "A14", "A15", "class"
    ]

    # common label mapping in Credit Approval dataset
    # '+' -> 1, '-' -> 0
    df["y"] = df["class"].map({"+": 1, "-": 0})
    if df["y"].isna().any():
        # fallback: robust if labels already encoded / spaces etc.
        vals = sorted(df["class"].dropna().unique().tolist())
        mapping = {vals[0]: 0, vals[-1]: 1} if len(vals) >= 2 else {vals[0]: 0}
        df["y"] = df["class"].map(mapping)
    return df

def split_credit_df(df_full, args):
    df_tr, df_te = train_test_split(
        df_full,
        test_size=args.split_ratio,
        stratify=df_full["y"],
        random_state=args.split_seed
    )
    df_tr = df_tr.reset_index(drop=True)
    df_te = df_te.reset_index(drop=True)
    return df_tr, df_te

# =========================================================
# Main one run
# =========================================================
def run_one(args, run_id: int):
    seed = args.seed + run_id
    set_seed(seed)
    device = get_device(args.device)
    print(f"[Device] {device} | seed={seed}")

    # 1) Load & split Credit
    df_full = load_credit_df(args)
    df_tr_raw, df_te_raw = split_credit_df(df_full, args)
    print(f"Train: {len(df_tr_raw)} rows, Test: {len(df_te_raw)} rows")
    print("Train label dist:", Counter(df_tr_raw["y"]))
    print("Test  label dist:", Counter(df_te_raw["y"]))

    # 2) Preprocess train/test (fit transformer on train)
    # Your Credit HGNN preprocess function style appears to accept dataframe + transformer
    X_tr, y_tr_np, df_tr_proc, transformer = preprocess_node_features(df_tr_raw)
    X_te, y_te_np, df_te_proc, _ = preprocess_node_features(df_te_raw, transformer=transformer)

    X_tr = np.asarray(X_tr)
    y_tr_np = np.asarray(y_tr_np, dtype=np.int64)
    X_te = np.asarray(X_te)
    y_te_np = np.asarray(y_te_np, dtype=np.int64)

    N = X_tr.shape[0]
    C = int(y_tr_np.max()) + 1

    print(f"Processed train: X={X_tr.shape}, y={y_tr_np.shape}")
    print(f"Processed test : X={X_te.shape}, y={y_te_np.shape}")

    # 3) Build hypergraphs from processed raw df
    t_hg0 = time.time()
    try:
        # Credit HGNN version often needs feature_cols=cat+cont
        hyperedges_tr = generate_hyperedge_dict(
            df_tr_proc,
            feature_cols=args.cat_cols + args.cont_cols,
            max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
            device=device
        )
        hyperedges_te = generate_hyperedge_dict(
            df_te_proc,
            feature_cols=args.cat_cols + args.cont_cols,
            max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
            device=device
        )
    except TypeError:
        # fallback signatures
        hyperedges_tr = generate_hyperedge_dict(
            df_tr_proc,
            args.cat_cols,
            max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
            device=device
        )
        hyperedges_te = generate_hyperedge_dict(
            df_te_proc,
            args.cat_cols,
            max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
            device=device
        )

    H_tr, dv_tr, de_tr = to_hgnn_sparse_tensors_from_hyperedges(hyperedges_tr, N, device)
    H_te, dv_te, de_te = to_hgnn_sparse_tensors_from_hyperedges(hyperedges_te, X_te.shape[0], device)
    hg_build_time = time.time() - t_hg0

    # 4) Tensors
    fts_tr = torch.from_numpy(X_tr).float().to(device)
    lbls_tr = torch.from_numpy(y_tr_np).long().to(device)
    fts_te = torch.from_numpy(X_te).float().to(device)
    lbls_te = torch.from_numpy(y_te_np).long().to(device)

    # 5) Train full model on original train hypergraph
    model_full = HGNN_implicit(
        in_ch=fts_tr.shape[1],
        n_class=C,
        n_hid=args.hidden_dim,
        dropout=args.dropout
    ).to(device)

    optimizer = optim.Adam(model_full.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    criterion = nn.NLLLoss() if not args.use_focal_loss else FocalLoss(gamma=args.focal_gamma, reduction="mean")

    print("== Train Full Model ==")
    t0 = time.time()
    model_full = train_model(
        model_full, criterion, optimizer, scheduler,
        fts_tr, lbls_tr, H_tr, dv_tr, de_tr,
        num_epochs=args.epochs, print_freq=args.print_freq
    )
    full_train_time = time.time() - t0

    test_obj = {"x": fts_te, "y": lbls_te, "H": H_te, "dv_inv": dv_te, "de_inv": de_te}
    full_test_acc = float(evaluate_test_acc(model_full, test_obj))
    full_test_f1 = float(evaluate_test_f1(model_full, test_obj))
    print(f"[Full] Test ACC={full_test_acc:.4f} | F1={full_test_f1:.4f} | train_time={full_train_time + hg_build_time:.4f}s")

    # 6) Sample deleted rows
    n_del = max(1, int(N * args.remove_ratio))
    rng = np.random.default_rng(seed)
    deleted_idx_np = np.sort(rng.choice(np.arange(N), size=n_del, replace=False).astype(np.int64))
    deleted_t = torch.as_tensor(deleted_idx_np, dtype=torch.long, device=device)
    retain_mask, del_mask = build_masks(N, deleted_idx_np, device)
    print(f"[Delete] remove_ratio={args.remove_ratio}, deleted={len(deleted_idx_np)}/{N}")

    # 7) Build EditedHG structure by deleting nodes from hyperedges
    t_edit0 = time.time()
    H_edit, dv_edit, de_edit, hyperedges_edit = rebuild_structure_after_node_deletion(
        hyperedges_tr, deleted_idx_np, N, device
    )
    edit_time = time.time() - t_edit0
    print(f"[EditedHG] #hyperedges(orig)={len(hyperedges_tr)} -> #hyperedges(edit)={len(hyperedges_edit)}")

    # 8) Feature zeroing for FT input only (same style as your FT baselines)
    fts_tr_after = fts_tr.clone()
    fts_tr_after[deleted_t] = 0.0

    # Evaluate Full@EditedHG on retain/forget (use original fts_tr to measure behavior)
    retain_acc_full = eval_acc_masked(model_full, fts_tr, lbls_tr, retain_mask, H_edit, dv_edit, de_edit)
    forget_acc_full = eval_acc_masked(model_full, fts_tr, lbls_tr, del_mask, H_edit, dv_edit, de_edit)

    # MIA (optional)
    mia_overall_full = maybe_mia_overall(model_full, X_tr, y_tr_np, hyperedges_edit, args, device)

    # Build keep/del attack hyperedges for deleted-set MIA (optional)
    mia_deleted_full = None
    if args.run_mia and _HAS_MIA:
        try:
            keep_idx_np = np.where(~np.isin(np.arange(N), deleted_idx_np))[0]
            X_keep = X_tr[keep_idx_np]
            y_keep = y_tr_np[keep_idx_np]
            X_del = X_tr[deleted_idx_np]
            y_del = y_tr_np[deleted_idx_np]

            # Build keep/del hyperedges from processed dataframes to align with node indexing
            df_keep_proc = df_tr_proc.drop(index=deleted_idx_np).reset_index(drop=True)
            df_del_proc = df_tr_proc.iloc[deleted_idx_np].reset_index(drop=True)

            try:
                he_keep = generate_hyperedge_dict(
                    df_keep_proc,
                    feature_cols=args.cat_cols + args.cont_cols,
                    max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
                    device=device
                )
                he_del = generate_hyperedge_dict(
                    df_del_proc,
                    feature_cols=args.cat_cols + args.cont_cols,
                    max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
                    device=device
                )
            except TypeError:
                he_keep = generate_hyperedge_dict(
                    df_keep_proc, args.cat_cols,
                    max_nodes_per_hyperedge=args.max_nodes_per_hyperedge, device=device
                )
                he_del = generate_hyperedge_dict(
                    df_del_proc, args.cat_cols,
                    max_nodes_per_hyperedge=args.max_nodes_per_hyperedge, device=device
                )

            he_keepdel = {}
            eid = 0
            for nodes in he_keep.values():
                he_keepdel[eid] = list(nodes)
                eid += 1
            offset = X_keep.shape[0]
            for nodes in he_del.values():
                he_keepdel[eid] = [n + offset for n in nodes]
                eid += 1

            X_attack = np.vstack([X_keep, X_del])
            y_attack = np.hstack([y_keep, y_del])

            # retained as members=True, deleted as non-members=False
            member_mask = np.concatenate([
                np.ones(X_keep.shape[0], dtype=bool),
                np.zeros(X_del.shape[0], dtype=bool)
            ])

            mia_deleted_full = maybe_mia_deleted(
                model_full, X_attack, y_attack, he_keepdel, member_mask, args, device
            )
        except Exception as e:
            print(f"[WARN] Build deleted-MIA input failed: {e}")
            mia_deleted_full = None

    rows = []

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

    # 9) FT-K
    print("\n== FT-K (warm-start on EditedHG) ==")
    for K in args.ft_steps:
        m = copy.deepcopy(model_full)
        for p in m.parameters():
            p.requires_grad = True

        m, update_time = finetune_steps_hgnn(
            m,
            x_after=fts_tr_after,
            y=lbls_tr,
            retain_mask=retain_mask,
            H_edit=H_edit,
            dv_edit=dv_edit,
            de_edit=de_edit,
            steps=int(K),
            lr=args.ft_lr,
            weight_decay=args.ft_wd
        )

        test_acc = float(evaluate_test_acc(m, test_obj))
        retain_acc = eval_acc_masked(m, fts_tr, lbls_tr, retain_mask, H_edit, dv_edit, de_edit)
        forget_acc = eval_acc_masked(m, fts_tr, lbls_tr, del_mask, H_edit, dv_edit, de_edit)
        total = edit_time + update_time

        mia_overall = maybe_mia_overall(m, X_tr, y_tr_np, hyperedges_edit, args, device)
        mia_deleted = None  # 可按上面的 keep/del 构造再复用；为避免重复耗时这里默认不再重复构造
        if args.run_mia and _HAS_MIA and args.run_mia_deleted:
            try:
                # 复用上面那套构造（重新算一遍，确保每个模型单独评估）
                keep_idx_np = np.where(~np.isin(np.arange(N), deleted_idx_np))[0]
                X_keep = X_tr[keep_idx_np]
                y_keep = y_tr_np[keep_idx_np]
                X_del = X_tr[deleted_idx_np]
                y_del = y_tr_np[deleted_idx_np]
                df_keep_proc = df_tr_proc.drop(index=deleted_idx_np).reset_index(drop=True)
                df_del_proc = df_tr_proc.iloc[deleted_idx_np].reset_index(drop=True)

                try:
                    he_keep = generate_hyperedge_dict(df_keep_proc, feature_cols=args.cat_cols + args.cont_cols,
                                                      max_nodes_per_hyperedge=args.max_nodes_per_hyperedge, device=device)
                    he_del = generate_hyperedge_dict(df_del_proc, feature_cols=args.cat_cols + args.cont_cols,
                                                     max_nodes_per_hyperedge=args.max_nodes_per_hyperedge, device=device)
                except TypeError:
                    he_keep = generate_hyperedge_dict(df_keep_proc, args.cat_cols,
                                                      max_nodes_per_hyperedge=args.max_nodes_per_hyperedge, device=device)
                    he_del = generate_hyperedge_dict(df_del_proc, args.cat_cols,
                                                     max_nodes_per_hyperedge=args.max_nodes_per_hyperedge, device=device)

                he_keepdel = {}
                eid = 0
                for nodes in he_keep.values():
                    he_keepdel[eid] = list(nodes); eid += 1
                offset = X_keep.shape[0]
                for nodes in he_del.values():
                    he_keepdel[eid] = [n + offset for n in nodes]; eid += 1

                X_attack = np.vstack([X_keep, X_del])
                y_attack = np.hstack([y_keep, y_del])
                member_mask = np.concatenate([
                    np.ones(X_keep.shape[0], dtype=bool),
                    np.zeros(X_del.shape[0], dtype=bool)
                ])
                mia_deleted = maybe_mia_deleted(m, X_attack, y_attack, he_keepdel, member_mask, args, device)
            except Exception as e:
                print(f"[WARN] FT-K deleted-MIA failed: {e}")
                mia_deleted = None

        print(
            f"FT-K@EditedHG    K={int(K):4d} | edit={edit_time:.4f} | update={update_time:.4f} | total={total:.4f} | "
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

    # 10) FT-head
    print("\n== FT-head (only train last layer) ==")
    for K in args.ft_steps:
        m = copy.deepcopy(model_full)
        freeze_all_but_head_hgnn(m)

        m, update_time = finetune_steps_hgnn(
            m,
            x_after=fts_tr_after,
            y=lbls_tr,
            retain_mask=retain_mask,
            H_edit=H_edit,
            dv_edit=dv_edit,
            de_edit=de_edit,
            steps=int(K),
            lr=args.ft_lr,
            weight_decay=args.ft_wd
        )

        test_acc = float(evaluate_test_acc(m, test_obj))
        retain_acc = eval_acc_masked(m, fts_tr, lbls_tr, retain_mask, H_edit, dv_edit, de_edit)
        forget_acc = eval_acc_masked(m, fts_tr, lbls_tr, del_mask, H_edit, dv_edit, de_edit)
        total = edit_time + update_time

        mia_overall = maybe_mia_overall(m, X_tr, y_tr_np, hyperedges_edit, args, device)
        mia_deleted = None
        if args.run_mia and _HAS_MIA and args.run_mia_deleted:
            try:
                keep_idx_np = np.where(~np.isin(np.arange(N), deleted_idx_np))[0]
                X_keep = X_tr[keep_idx_np]
                y_keep = y_tr_np[keep_idx_np]
                X_del = X_tr[deleted_idx_np]
                y_del = y_tr_np[deleted_idx_np]
                df_keep_proc = df_tr_proc.drop(index=deleted_idx_np).reset_index(drop=True)
                df_del_proc = df_tr_proc.iloc[deleted_idx_np].reset_index(drop=True)

                try:
                    he_keep = generate_hyperedge_dict(df_keep_proc, feature_cols=args.cat_cols + args.cont_cols,
                                                      max_nodes_per_hyperedge=args.max_nodes_per_hyperedge, device=device)
                    he_del = generate_hyperedge_dict(df_del_proc, feature_cols=args.cat_cols + args.cont_cols,
                                                     max_nodes_per_hyperedge=args.max_nodes_per_hyperedge, device=device)
                except TypeError:
                    he_keep = generate_hyperedge_dict(df_keep_proc, args.cat_cols,
                                                      max_nodes_per_hyperedge=args.max_nodes_per_hyperedge, device=device)
                    he_del = generate_hyperedge_dict(df_del_proc, args.cat_cols,
                                                     max_nodes_per_hyperedge=args.max_nodes_per_hyperedge, device=device)

                he_keepdel = {}
                eid = 0
                for nodes in he_keep.values():
                    he_keepdel[eid] = list(nodes); eid += 1
                offset = X_keep.shape[0]
                for nodes in he_del.values():
                    he_keepdel[eid] = [n + offset for n in nodes]; eid += 1

                X_attack = np.vstack([X_keep, X_del])
                y_attack = np.hstack([y_keep, y_del])
                member_mask = np.concatenate([
                    np.ones(X_keep.shape[0], dtype=bool),
                    np.zeros(X_del.shape[0], dtype=bool)
                ])
                mia_deleted = maybe_mia_deleted(m, X_attack, y_attack, he_keepdel, member_mask, args, device)
            except Exception as e:
                print(f"[WARN] FT-head deleted-MIA failed: {e}")
                mia_deleted = None

        print(
            f"FT-head@EditedHG K={int(K):4d} | edit={edit_time:.4f} | update={update_time:.4f} | total={total:.4f} | "
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

# =========================================================
# Summary
# =========================================================
def summarize_and_save(all_rows, out_csv):
    df = pd.DataFrame(all_rows)
    df.to_csv(out_csv, index=False)

    num_cols = ["edit_sec", "update_sec", "total_sec", "test_acc", "retain_acc", "forget_acc", "mia_overall", "mia_deleted"]
    g = df.groupby(["method", "K"], dropna=False)[num_cols].agg(["mean", "std"]).reset_index()

    print("\n== Summary (mean±std) ==")

    def _fmt(m, s):
        if pd.isna(m):
            return "NA"
        if pd.isna(s):
            return f"{m:.4f}"
        return f"{m:.4f}±{s:.4f}"

    for _, r in g.iterrows():
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

# =========================================================
# Args
# =========================================================
def get_args():
    p = argparse.ArgumentParser("Credit HGNN FT baseline (row deletion on edited hypergraph)")

    # Data (Credit single-file)
    p.add_argument("--data_csv", type=str, default=CREDIT_DATA)
    p.add_argument("--split_ratio", type=float, default=0.2)
    p.add_argument("--split_seed", type=int, default=21)

    # Credit feature groups (match your HGNN script)
    p.add_argument("--cat_cols", type=str, nargs="+",
                   default=["A1", "A4", "A5", "A6", "A7", "A9", "A10", "A12", "A13"])
    p.add_argument("--cont_cols", type=str, nargs="+",
                   default=["A2", "A3", "A8", "A11", "A14", "A15"])

    # Hypergraph
    p.add_argument("--max_nodes_per_hyperedge", type=int, default=50)

    # Model/train
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--milestones", type=int, nargs="+", default=[100, 150])
    p.add_argument("--gamma", type=float, default=0.1)
    p.add_argument("--print_freq", type=int, default=10)

    # Loss choice
    p.add_argument("--use_focal_loss", action="store_true", default=False)
    p.add_argument("--focal_gamma", type=float, default=2.0)

    # Row deletion
    p.add_argument("--remove_ratio", type=float, default=0.2)

    # FT
    p.add_argument("--ft_steps", type=int, nargs="+", default=[50, 100, 200])
    p.add_argument("--ft_lr", type=float, default=5e-3)
    p.add_argument("--ft_wd", type=float, default=0.0)

    # MIA
    p.add_argument("--run_mia", action="store_true", default=True)
    p.add_argument("--run_mia_deleted", action="store_true", default=False)

    # misc
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--out_csv", type=str, default="ft_hgnn_row_credit_results.csv")
    return p.parse_args()

def main():
    args = get_args()

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