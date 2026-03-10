# -*- coding: utf-8 -*-
"""
Bank dataset - HGAT row-deletion FT baselines (FT-K / FT-head)
Author: adapted for TabHGIF baseline experiments

Notes:
1) This is an FT baseline (not GIF), and it will not call approx_gif
2) It uses the Bank HGAT data preprocessing and model code
3) Row unlearning = deleting a subset of nodes from the training set, then warm-start finetuning on the edited hypergraph
4) Output edit/update/total/test_acc/retain_acc/forget_acc (MIA fields are placeholders for now)
"""

import os
import time
import copy
import math
import argparse
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ====== Existing Bank-HGAT modules (according to your project paths) ======
from bank.HGAT.data_preprocessing_bank import (
    preprocess_node_features_bank,
    generate_hyperedge_dict_bank,
)
from bank.HGAT.HGAT_new import HGAT_JK
from utils.common_utils import evaluate_test_acc

# If you already have an MIA interface, you can enable and integrate it
# from bank.HGAT.MIA_HGAT import membership_inference_hgat


# ============================================================
# Utils
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(cuda=True):
    if cuda and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def build_incidence_matrix(hyperedges: dict, num_nodes: int, device=None) -> torch.Tensor:
    """
    Consistent with the style of your HGAT row bank unlearning script:
    hyperedges: dict(edge_id -> list[node_idx])
    Return: torch sparse COO, shape [num_edges, num_nodes]
    """
    n_edges = len(hyperedges)
    H = torch.zeros((n_edges, num_nodes), dtype=torch.float32, device=device)
    for i, nodes in enumerate(hyperedges.values()):
        if len(nodes) == 0:
            continue
        H[i, nodes] = 1.0
    return H.to_sparse()


def clone_model(model: nn.Module) -> nn.Module:
    return copy.deepcopy(model)


def try_get_last_linear_layer(model: nn.Module):
    """
    Try to locate the final classification layer of HGAT for FT-head.
    The naming in your project for HGAT_JK may be inconsistent, so several common candidates are checked.
    """
    candidates = [
        "classifier", "fc", "fc_out", "out_proj", "mlp", "lin", "linear", "predictor"
    ]
    for name in candidates:
        if hasattr(model, name):
            layer = getattr(model, name)
            if isinstance(layer, nn.Module):
                return name, layer

    # Fallback: search backward for the last Linear layer
    last_name, last_mod = None, None
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            last_name, last_mod = n, m
    return last_name, last_mod


def freeze_all_except(model: nn.Module, train_layer_name: str):
    """
    Only train the parameters of the module corresponding to train_layer_name; freeze all others.
    Supports nested module names, e.g. 'classifier.2'
    """
    for p in model.parameters():
        p.requires_grad = False

    # Get nested module
    mod = model
    for part in train_layer_name.split("."):
        if part.isdigit():
            mod = mod[int(part)]
        else:
            mod = getattr(mod, part)
    for p in mod.parameters():
        p.requires_grad = True


def count_trainable_params(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def to_tensors(X_np, y_np, device):
    x = torch.from_numpy(np.asarray(X_np)).float().to(device)
    y = torch.from_numpy(np.asarray(y_np)).long().to(device)
    return x, y


def hgat_forward_logits(model, x, H):
    """
    Compatible with different HGAT_JK forward signatures
    Common cases:
      - model(x, H)
      - model(x, H, something...)
    Here, the most common form model(x, H) is tried first
    """
    return model(x, H)


def train_hgat_model(
    model,
    x_train,
    y_train,
    H_train,
    epochs=100,
    lr=1e-3,
    weight_decay=5e-4,
    milestones=(50, 80),
    gamma=0.1,
    print_freq=10,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=list(milestones),
        gamma=gamma
    )

    model.train()
    t0 = time.time()
    best_acc = 0.0
    best_state = copy.deepcopy(model.state_dict())

    for ep in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits = hgat_forward_logits(model, x_train, H_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            pred = logits.argmax(dim=1)
            acc = (pred == y_train).float().mean().item()

        if acc > best_acc:
            best_acc = acc
            best_state = copy.deepcopy(model.state_dict())

        if (ep == 1) or (ep % print_freq == 0) or (ep == epochs):
            print(f"[Train] ep {ep:4d}/{epochs} | loss={loss.item():.4f} | train_acc={acc:.4f}")

    model.load_state_dict(best_state)
    t1 = time.time()
    print(f"Training complete in {t1 - t0:.2f}s")
    print(f"Best Train Acc: {best_acc:.4f}")
    return model, (t1 - t0)


@torch.no_grad()
def eval_acc_on_hgat(model, x, y, H):
    model.eval()
    logits = hgat_forward_logits(model, x, H)
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item()


def member_mask_from_retain(retain_mask: np.ndarray):
    """
    retained=1, deleted=0
    """
    return retain_mask.astype(np.int64)


def member_mask_from_deleted(deleted_idx: np.ndarray, N: int):
    """
    deleted=1, others=0
    """
    mm = np.zeros(N, dtype=np.int64)
    mm[np.asarray(deleted_idx, dtype=np.int64)] = 1
    return mm


def rebuild_structure_after_node_deletion(train_edges: dict, deleted_idx, N_old: int, device):
    """
    Edited hypergraph after row deletion (preserve original node numbering, no reindexing):
    - Remove deleted nodes from each hyperedge
    - Drop hyperedges with length < 2
    - Return the new hyperedges(dict) + sparse H
    """
    del_set = set(int(i) for i in np.asarray(deleted_idx).tolist())
    new_edges = {}
    eid = 0
    for _, nodes in train_edges.items():
        kept = [int(v) for v in nodes if int(v) not in del_set]
        if len(kept) >= 2:
            new_edges[eid] = kept
            eid += 1

    H_edit = build_incidence_matrix(new_edges, N_old, device=device)
    return new_edges, H_edit


import inspect
import torch

def make_hgat_model_from_args(args, in_dim, num_classes, device):
    """
    Automatically adapt parameter names according to the constructor of your local HGAT.HGAT_new.HGAT_JK.
    Priority is given to the parameter names already verified in your current project:
      in_dim, hidden_dim, out_dim, dropout, alpha, num_layers, use_jk
    """
    from HGAT.HGAT_new import HGAT_JK

    sig = inspect.signature(HGAT_JK.__init__)
    param_names = set(sig.parameters.keys())

    # Remove self
    param_names.discard("self")

    # Prepare candidate parameters first (consistent with your HGAT_Unlearning_row_nei.py)
    base_kwargs = {
        "in_dim": int(in_dim),
        "hidden_dim": int(getattr(args, "hidden_dim", 64)),
        "out_dim": int(num_classes),
        "dropout": float(getattr(args, "dropout", 0.5)),
        "alpha": float(getattr(args, "alpha", 0.5)),
        "num_layers": int(getattr(args, "num_layers", 2)),
        "use_jk": bool(getattr(args, "use_jk", False)),
    }

    # If the local version uses different parameter names, apply alias mapping (compatible with different HGAT implementations)
    alias_map = {
        "nfeat": base_kwargs["in_dim"],
        "nhid": base_kwargs["hidden_dim"],
        "nclass": base_kwargs["out_dim"],
        "dropout": base_kwargs["dropout"],
        "alpha": base_kwargs["alpha"],
        "nlayer": base_kwargs["num_layers"],
        "num_layers": base_kwargs["num_layers"],
        "jk": base_kwargs["use_jk"],
        "use_jk": base_kwargs["use_jk"],
        "in_dim": base_kwargs["in_dim"],
        "hidden_dim": base_kwargs["hidden_dim"],
        "out_dim": base_kwargs["out_dim"],
    }

    # Only pass parameters supported by this version of __init__
    kw = {}
    for k, v in alias_map.items():
        if k in param_names:
            kw[k] = v

    # Key field validation (at least input/hidden/output must be found)
    has_in = ("in_dim" in kw) or ("nfeat" in kw)
    has_hid = ("hidden_dim" in kw) or ("nhid" in kw)
    has_out = ("out_dim" in kw) or ("nclass" in kw)

    if not (has_in and has_hid and has_out):
        raise RuntimeError(
            f"Unable to automatically construct HGAT_JK. Detected __init__ parameters: {sorted(list(param_names))}\n"
            f"Currently auto-generated kwargs: {kw}\n"
            f"Please map them one by one according to the parameter names in your local HGAT_new.py HGAT_JK.__init__."
        )

    model = HGAT_JK(**kw).to(device)
    return model
def summarize_rows(rows, save_csv=None):
    """
    rows: list[dict]
    """
    df = pd.DataFrame(rows)

    # Sort by setting + K
    if "K" in df.columns:
        df = df.sort_values(by=["setting", "K"]).reset_index(drop=True)

    if save_csv is not None:
        os.makedirs(os.path.dirname(save_csv), exist_ok=True) if os.path.dirname(save_csv) else None
        df.to_csv(save_csv, index=False)
        print(f"\n[Saved] {save_csv}")

    print("\n== Summary (mean±std) ==")
    # Currently runs=1 in most cases; this is also compatible with multiple runs
    group_cols = ["setting", "K"]
    metric_cols = [
        "edit", "update", "total",
        "test_acc", "retain_acc", "forget_acc",
        "mia_overall", "mia_deleted"
    ]
    show_cols = [c for c in metric_cols if c in df.columns]

    for (setting, K), g in df.groupby(group_cols, sort=False):
        msg = f"{setting:<14} K={int(K):4d}"
        for c in show_cols:
            vals = g[c].dropna()
            if len(vals) == 0:
                val_str = "NA"
            elif len(vals) == 1:
                val_str = f"{vals.iloc[0]:.4f}"
            else:
                val_str = f"{vals.mean():.4f}±{vals.std(ddof=0):.4f}"
            msg += f" | {c}={val_str}"
        print(msg)


# ============================================================
# Core experiment (single run)
# ============================================================

def run_one(args, run_id=0):
    set_seed(args.seed + run_id)
    device = get_device(args.cuda)
    print(f"\n================= RUN {run_id + 1}/{args.runs} =================")
    print(f"[Device] {device}")

    # 1) Read the full Bank table and split
    df_full = pd.read_csv(args.data_csv, sep=';', skipinitialspace=True)
    label_col = args.label_col
    assert label_col in df_full.columns, f"Label column '{label_col}' not found in {args.data_csv}"

    df_train, df_test = train_test_split(
        df_full,
        test_size=args.split_ratio,
        stratify=df_full[label_col],
        random_state=args.data_split_seed,
    )
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    # 2) Preprocessing (Bank)
    X_train, y_train, df_train_proc, transformer = preprocess_node_features_bank(df_train, is_test=False)
    X_test, y_test, df_test_proc, _ = preprocess_node_features_bank(df_test, is_test=True, transformer=transformer)

    print(f"Train: {len(df_train)}, Test: {len(df_test)}")
    print(f"Label distribution (train) -> 0: {(np.asarray(y_train)==0).sum()}, 1: {(np.asarray(y_train)==1).sum()}")
    print(f"Label distribution (test)  -> 0: {(np.asarray(y_test)==0).sum()}, 1: {(np.asarray(y_test)==1).sum()}")

    # 3) Build the original hypergraph (train/test)
    t_edit0 = time.time()
    train_edges = generate_hyperedge_dict_bank(
        df_train_proc, args.cat_cols, args.cont_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        device=device
    )
    test_edges = generate_hyperedge_dict_bank(
        df_test_proc, args.cat_cols, args.cont_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        device=device
    )
    H_train = build_incidence_matrix(train_edges, X_train.shape[0], device=device)
    H_test = build_incidence_matrix(test_edges, X_test.shape[0], device=device)
    print(f"[OrigHG] train #hyperedges={len(train_edges)} | test #hyperedges={len(test_edges)}")
    _ = time.time() - t_edit0  # This is only original graph construction and is not counted in summary edit

    # 4) Full model training
    x_tr, y_tr = to_tensors(X_train, y_train, device)
    x_te, y_te = to_tensors(X_test, y_test, device)

    print("== Train Full Model ==")
    full_model = make_hgat_model_from_args(
        args=args,
        in_dim=X_train.shape[1],
        num_classes=int(np.max(y_train)) + 1,
        device=device
    )
    full_model, train_time = train_hgat_model(
        full_model, x_tr, y_tr, H_train,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        milestones=args.milestones,
        gamma=args.gamma,
        print_freq=args.log_every
    )
    full_test_acc = eval_acc_on_hgat(full_model, x_te, y_te, H_test)
    print(f"[Full] Test ACC={full_test_acc:.4f} | train_time={train_time:.2f}s")

    # 5) Randomly delete training nodes (row deletion)
    N = X_train.shape[0]
    n_del = int(N * args.remove_ratio)
    deleted_idx = np.random.choice(N, size=n_del, replace=False)
    deleted_idx = np.sort(deleted_idx)
    retain_mask = np.ones(N, dtype=bool)
    retain_mask[deleted_idx] = False
    print(f"[Delete] remove_ratio={args.remove_ratio}, deleted={len(deleted_idx)}")

    # 6) Build the edited hypergraph (delete nodes from the training graph; test graph remains unchanged)
    t_edit_start = time.time()
    edited_train_edges, H_train_edit = rebuild_structure_after_node_deletion(
        train_edges, deleted_idx, N_old=N, device=device
    )
    edit_time = time.time() - t_edit_start
    print(f"[EditedHG] train #hyperedges(orig)={len(train_edges)} -> #hyperedges(edit)={len(edited_train_edges)}")

    # 7) Directly evaluate Full on editedHG (without finetuning)
    full_on_edit_train_acc = eval_acc_on_hgat(full_model, x_tr, y_tr, H_train_edit)
    # Both retain / forget acc are computed on the training set (using the structure of editedHG)
    with torch.no_grad():
        full_model.eval()
        logits_edit = hgat_forward_logits(full_model, x_tr, H_train_edit)
        pred_edit = logits_edit.argmax(dim=1).detach().cpu().numpy()
        y_np = np.asarray(y_train)

    retain_acc_full_edit = float((pred_edit[retain_mask] == y_np[retain_mask]).mean()) if retain_mask.any() else np.nan
    forget_mask = ~retain_mask
    forget_acc_full_edit = float((pred_edit[forget_mask] == y_np[forget_mask]).mean()) if forget_mask.any() else np.nan
    full_test_acc_edit = eval_acc_on_hgat(full_model, x_te, y_te, H_test)
    print(f"[Full@EditedHG] Test ACC={full_test_acc_edit:.4f} | Retain ACC={retain_acc_full_edit:.4f} | Forget ACC={forget_acc_full_edit:.4f}")

    rows = []

    # Full@EditedHG as the baseline row
    rows.append(dict(
        run=run_id + 1,
        setting="Full@EditedHG",
        K=0,
        edit=edit_time,
        update=0.0,
        total=edit_time,
        test_acc=full_test_acc_edit,
        retain_acc=retain_acc_full_edit,
        forget_acc=forget_acc_full_edit,
        mia_overall=np.nan,
        mia_deleted=np.nan,
    ))

    # 8) FT-K (warm-start, all params)
    print("\n== FT-K (warm-start on EditedHG) ==")
    for K in args.ft_k_list:
        model_k = clone_model(full_model)
        for p in model_k.parameters():
            p.requires_grad = True

        t_upd = time.time()
        model_k, upd_time = train_hgat_model(
            model_k, x_tr, y_tr, H_train_edit,
            epochs=K,
            lr=args.ft_lr,
            weight_decay=args.ft_weight_decay,
            milestones=args.ft_milestones if len(args.ft_milestones) > 0 else (max(1, K // 2),),
            gamma=args.ft_gamma,
            print_freq=max(1, min(args.log_every, K))
        )
        # Note: train_hgat_model returns its internal total time; here upd_time is used uniformly
        _ = time.time() - t_upd

        test_acc = eval_acc_on_hgat(model_k, x_te, y_te, H_test)
        with torch.no_grad():
            logits = hgat_forward_logits(model_k, x_tr, H_train_edit)
            pred = logits.argmax(dim=1).detach().cpu().numpy()
        retain_acc = float((pred[retain_mask] == y_np[retain_mask]).mean()) if retain_mask.any() else np.nan
        forget_acc = float((pred[forget_mask] == y_np[forget_mask]).mean()) if forget_mask.any() else np.nan

        total_time = edit_time + upd_time
        print(f"[FT-K] K={K:4d} | Test ACC={test_acc:.4f} | Retain ACC={retain_acc:.4f} | Forget ACC={forget_acc:.4f} | edit={edit_time:.4f}s | update={upd_time:.4f}s | total={total_time:.4f}s")

        rows.append(dict(
            run=run_id + 1,
            setting="FT-K@EditedHG",
            K=int(K),
            edit=edit_time,
            update=upd_time,
            total=total_time,
            test_acc=test_acc,
            retain_acc=retain_acc,
            forget_acc=forget_acc,
            mia_overall=np.nan,
            mia_deleted=np.nan,
        ))

    # 9) FT-head (only last layer)
    print("\n== FT-head (only train last layer) ==")
    last_name, last_layer = try_get_last_linear_layer(full_model)
    if last_layer is None:
        print("[Warn] Final linear layer not found. FT-head will fall back to FT-K (full-parameter training).")
    else:
        print(f"[FT-head] trainable layer = {last_name} ({last_layer.__class__.__name__})")

    for K in args.ft_k_list:
        model_h = clone_model(full_model)
        if last_layer is None:
            for p in model_h.parameters():
                p.requires_grad = True
        else:
            freeze_all_except(model_h, last_name)
        print(f"[FT-head] K={K:4d} | trainable_params={count_trainable_params(model_h)}")

        model_h, upd_time = train_hgat_model(
            model_h, x_tr, y_tr, H_train_edit,
            epochs=K,
            lr=args.ft_head_lr,
            weight_decay=args.ft_head_weight_decay,
            milestones=args.ft_milestones if len(args.ft_milestones) > 0 else (max(1, K // 2),),
            gamma=args.ft_gamma,
            print_freq=max(1, min(args.log_every, K))
        )

        test_acc = eval_acc_on_hgat(model_h, x_te, y_te, H_test)
        with torch.no_grad():
            logits = hgat_forward_logits(model_h, x_tr, H_train_edit)
            pred = logits.argmax(dim=1).detach().cpu().numpy()
        retain_acc = float((pred[retain_mask] == y_np[retain_mask]).mean()) if retain_mask.any() else np.nan
        forget_acc = float((pred[forget_mask] == y_np[forget_mask]).mean()) if forget_mask.any() else np.nan

        total_time = edit_time + upd_time
        print(f"[FT-head] K={K:4d} | Test ACC={test_acc:.4f} | Retain ACC={retain_acc:.4f} | Forget ACC={forget_acc:.4f} | edit={edit_time:.4f}s | update={upd_time:.4f}s | total={total_time:.4f}s")

        rows.append(dict(
            run=run_id + 1,
            setting="FT-head@EditedHG",
            K=int(K),
            edit=edit_time,
            update=upd_time,
            total=total_time,
            test_acc=test_acc,
            retain_acc=retain_acc,
            forget_acc=forget_acc,
            mia_overall=np.nan,
            mia_deleted=np.nan,
        ))

    print("\nDone.")
    return rows


# ============================================================
# Args
# ============================================================

def build_parser():
    p = argparse.ArgumentParser("Bank-HGAT row FT baseline")

    # Data
    p.add_argument("--data_csv", type=str,
                   default="/root/autodl-tmp/TabHGIF/data_banking/bank/bank-full.csv")
    p.add_argument("--label_col", type=str, default="y")
    p.add_argument("--split_ratio", type=float, default=0.2)
    p.add_argument("--data_split_seed", type=int, default=21)

    # Bank column definitions (consistent with your existing bank script)
    p.add_argument("--cat_cols", nargs="+", type=str,
                   default=['job','marital','education','default','housing','loan','contact','month','poutcome'])
    p.add_argument("--cont_cols", nargs="+", type=str,
                   default=['age','balance','day','duration','campaign','pdays','previous'])
    p.add_argument("--max_nodes_per_hyperedge", type=int, default=50)

    # Deletion setting (row)
    p.add_argument("--remove_ratio", type=float, default=0.3)

    # General training
    p.add_argument("--cuda", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--log_every", type=int, default=20)

    # Full model hyperparameters (can be tuned according to your original HGAT row bank script)
    p.add_argument("--epochs", type=int, default=130)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--milestones", nargs="+", type=int, default=[80, 110])
    p.add_argument("--gamma", type=float, default=0.1)

    # HGAT model hyperparameters (if your local HGAT_JK initialization parameter names differ, modify make_hgat_model_from_args)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--jk_mode", type=str, default="cat")

    # FT baseline settings
    p.add_argument("--ft_k_list", nargs="+", type=int, default=[50, 100, 200])

    p.add_argument("--ft_lr", type=float, default=1e-3)
    p.add_argument("--ft_weight_decay", type=float, default=5e-4)

    p.add_argument("--ft_head_lr", type=float, default=1e-3)
    p.add_argument("--ft_head_weight_decay", type=float, default=5e-4)

    p.add_argument("--ft_milestones", nargs="+", type=int, default=[])
    p.add_argument("--ft_gamma", type=float, default=0.1)

    # Output
    p.add_argument("--save_csv", type=str, default="ft_hgat_row_bank_results.csv")

    return p


def main():
    args = build_parser().parse_args()

    all_rows = []
    for run_id in range(args.runs):
        rows = run_one(args, run_id=run_id)
        all_rows.extend(rows)

    summarize_rows(all_rows, save_csv=args.save_csv)


if __name__ == "__main__":
    main()