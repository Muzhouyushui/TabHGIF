# -*- coding: utf-8 -*-
"""
run_ft_hgcn_col_zero_bank.py

Bank dataset: FT baseline experiments for HGCN under column deletion (Column Unlearning)
- FT-K@EditedHG: full-parameter finetuning for K steps on the edited hypergraph
- FT-head@EditedHG: finetuning only the last layer for K steps on the edited hypergraph
- Full@EditedHG: only edit the structure without updating parameters (used as a reference)

Notes:
1) Does not depend on bank.HGCN.config.get_args() (there is no config.py on your side)
2) Uses --data_csv to read the single Bank file (bank-full.csv)
3) The training / column-deletion pipeline is aligned with the idea of HGCN_Unlearning_col_bank.py
4) The FT framework is aligned with the idea of your existing run_ft_hgcn_col_zero.py
"""

import os
import time
import copy
import csv
import argparse
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split


# =========================
# Modify the imports here according to your actual project paths
# =========================
# If your script is placed under /root/autodl-tmp/TabHGIF/bank/FT_bank/...,
# the working directory is usually already the TabHGIF root when running; otherwise you can manually add sys.path.
#
# Example (if needed):
# import sys
# sys.path.append("/root/autodl-tmp/TabHGIF")

from bank.HGCN.data_preprocessing_col_bank import (
    preprocess_node_features,
    generate_hyperedge_dict,
    delete_feature_columns_hgcn,
)
from bank.HGCN.HGCN_utils import evaluate_hgcn_acc, evaluate_hgcn_f1
from bank.HGCN.GIF_HGCN_COL_bank import train_model  # only reuse train_model here, GIF is not used
from bank.HGCN.HGCN import HyperGCN, laplacian


# =========================================================
#                         Argument definitions (independent of config.py)
# =========================================================
def get_args():
    p = argparse.ArgumentParser("Bank HGCN FT baseline (Column Unlearning on EditedHG)")

    # ===== Bank single-file data =====
    p.add_argument("--data_csv", type=str,
                   default="/root/autodl-tmp/TabHGIF/data_banking/bank/bank-full.csv",
                   help="CSV path of the Bank dataset (bank-full.csv)")

    # ===== Data split =====
    p.add_argument("--split_ratio", type=float, default=0.2,
                   help="Test set ratio (train/test split)")

    # ===== Column information (for the Bank dataset) =====
    p.add_argument("--label_col", type=str, default="y")

    # Common categorical columns in Bank (you can further adjust according to your own preprocessing function)
    p.add_argument("--cat_cols", type=str, nargs="+", default=[
        "job", "marital", "education", "default", "housing", "loan",
        "contact", "month", "poutcome"
    ])

    # Common numerical columns in Bank
    p.add_argument("--cont_cols", type=str, nargs="+", default=[
        "age", "balance", "day", "duration", "campaign", "pdays", "previous"
    ])

    # ===== Column deletion setting (col unlearning) =====
    p.add_argument("--columns_to_unlearn", type=str, nargs="+", default=["age"],
                   help="Column names to apply column-level unlearning to; multiple columns are allowed")

    # ===== Hypergraph construction =====
    p.add_argument("--max_nodes_per_hyperedge", type=int, default=10000)
    p.add_argument("--mediators", action="store_true",
                   help="mediators option for laplacian()")

    # ===== HGCN model / basic training =====
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--fast", action="store_true")
    p.add_argument("--cuda", action="store_true", default=True)

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--milestones", type=int, nargs="+", default=[100, 150])
    p.add_argument("--gamma", type=float, default=0.1)
    p.add_argument("--log_every", type=int, default=10)

    # ===== FT baselines =====
    p.add_argument("--ft_steps", type=int, nargs="+", default=[50, 100, 200],
                   help="K steps for FT-K / FT-head")
    p.add_argument("--ft_lr", type=float, default=1e-3)
    p.add_argument("--ft_wd", type=float, default=0.0)
    p.add_argument("--ft_milestones", type=int, nargs="*", default=[],
                   help="Optional: scheduler milestones during FT stage (empty by default)")
    p.add_argument("--ft_gamma", type=float, default=0.1)

    # ===== MIA (placeholder, can also be skipped) =====
    p.add_argument("--run_mia", action="store_true", default=False)

    # ===== Multiple runs =====
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--seed", type=int, default=1)

    # ===== Output =====
    p.add_argument("--out_csv", type=str, default="ft_hgcn_col_bank_results.csv")

    return p.parse_args()


# =========================================================
#                         Utility functions
# =========================================================
def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _forward_hgcn(model, x):
    out = model(x)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out


def _clone_model(model: nn.Module) -> nn.Module:
    return copy.deepcopy(model)


def _set_model_structure(model, A):
    # Compatible with the HyperGCN implementation: the structural matrix is usually stored in model.structure
    model.structure = A
    # In some implementations, layers have a reapproximate flag to prevent structure rebuilding during forward
    if hasattr(model, "layers"):
        for layer in model.layers:
            if hasattr(layer, "reapproximate"):
                layer.reapproximate = False


def _get_last_trainable_module(model: nn.Module):
    # Prioritize common names
    for name in ["classifier", "final", "out_layer", "fc", "linear"]:
        if hasattr(model, name):
            mod = getattr(model, name)
            if isinstance(mod, nn.Module):
                return mod

    # fallback: find the last Linear
    last_linear = None
    for _, m in model.named_modules():
        if isinstance(m, nn.Linear):
            last_linear = m
    if last_linear is not None:
        return last_linear

    # fallback: the last child module
    children = [m for m in model.children()]
    if len(children) > 0:
        return children[-1]

    return None


def _freeze_all_then_unfreeze(model: nn.Module, target_module: nn.Module):
    for p in model.parameters():
        p.requires_grad = False
    if target_module is not None:
        for p in target_module.parameters():
            p.requires_grad = True


def _safe_eval_acc(model, x, y):
    """
    Compatible with the input dict key names of different versions of evaluate_hgcn_acc
    """
    try:
        return evaluate_hgcn_acc(model, {"lap": x, "y": y})
    except Exception:
        try:
            return evaluate_hgcn_acc(model, {"x": x, "y": y})
        except Exception:
            # fallback: compute manually
            model.eval()
            with torch.no_grad():
                out = _forward_hgcn(model, x)
                pred = out.argmax(dim=1)
                return (pred == y).float().mean().item()


def _safe_eval_f1(model, x, y):
    try:
        return evaluate_hgcn_f1(model, {"lap": x, "y": y})
    except Exception:
        try:
            return evaluate_hgcn_f1(model, {"x": x, "y": y})
        except Exception:
            return float("nan")


def _maybe_mia_hgcn_bank(model, args, x_train, y_train, x_test, y_test):
    """
    Placeholder interface for MIA. Add your own bank version of MIA here later.
    Returns (mia_overall, mia_aux)
    """
    if not getattr(args, "run_mia", False):
        return None, None
    print("[MIA] run_mia=True but MIA function is not connected yet -> return NA.")
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
    for method, K in sorted(groups.keys(), key=lambda x: (x[0], x[1])):
        vals = groups[(method, K)]

        def mstd(field):
            arr = np.array([float(v[field]) for v in vals if v[field] is not None], dtype=float)
            if len(arr) == 0:
                return None, None
            return arr.mean(), arr.std(ddof=0)

        e_m, e_s = mstd("edit")
        u_m, u_s = mstd("update")
        t_m, t_s = mstd("total")
        acc_m, acc_s = mstd("test_acc")
        mia_m, mia_s = mstd("mia_overall")

        line = (
            f"{method:<15} K={K:4d} | "
            f"edit={_fmt_num(e_m)}" + (f"±{e_s:.4f}" if e_s is not None else "") + " | "
            f"update={_fmt_num(u_m)}" + (f"±{u_s:.4f}" if u_s is not None else "") + " | "
            f"total={_fmt_num(t_m)}" + (f"±{t_s:.4f}" if t_s is not None else "") + " | "
            f"test_acc={_fmt_num(acc_m)}" + (f"±{acc_s:.4f}" if acc_s is not None else "") + " | "
            f"mia_overall={_fmt_num(mia_m)}" + (f"±{mia_s:.4f}" if mia_s is not None else "")
        )
        print(line)


# =========================================================
#                    Training / finetuning
# =========================================================
def _train_full_model_hgcn_bank(model, fts_tr, lbls_tr, args):
    """
    Reuse your bank train_model (from GIF_HGCN_COL_bank.py)
    """
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=args.milestones,
        gamma=args.gamma
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
        print_freq=args.log_every
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
    model.train()
    criterion = nn.NLLLoss()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)

    scheduler = None
    if milestones is not None and len(milestones) > 0:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    t0 = time.time()
    for _ in range(int(K)):
        optimizer.zero_grad()
        out = _forward_hgcn(model, x_train)
        loss = criterion(out, y_train)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

    update_time = time.time() - t0
    return model, update_time


# =========================================================
#                    Single run (one seed)
# =========================================================
def run_one(args, run_id: int):
    seed = int(args.seed) + run_id
    seed_everything(seed)

    device = torch.device("cuda:0" if (torch.cuda.is_available() and args.cuda) else "cpu")
    print(f"\n================= RUN {run_id + 1}/{args.runs} =================")
    print(f"[Device] {device} | seed={seed}")

    # -------------------------------------------------
    # 1) Read Bank CSV (using data_csv here, not train_csv)
    # -------------------------------------------------
    df = pd.read_csv(args.data_csv, sep=';', header=0)
    assert args.label_col in df.columns, f"Label column not found in CSV: {args.label_col}"

    # train/test split (consistent with the bank single-file style)
    df_tr, df_te = train_test_split(
        df,
        test_size=float(args.split_ratio),
        random_state=42 + run_id,
        stratify=df[args.label_col]
    )
    df_tr = df_tr.reset_index(drop=True)
    df_te = df_te.reset_index(drop=True)

    print(f"TRAIN={len(df_tr)} samples, TEST={len(df_te)} samples")
    print("– TRAIN dist:", Counter(df_tr[args.label_col]))
    print("– TEST  dist:", Counter(df_te[args.label_col]))

    # -------------------------------------------------
    # 2) Preprocessing: train fit_transform, test transform
    # -------------------------------------------------
    X_tr, y_tr, df_tr_proc, transformer = preprocess_node_features(df_tr, is_test=False)
    X_te, y_te, df_te_proc, _ = preprocess_node_features(df_te, is_test=True, transformer=transformer)

    fts_tr = torch.from_numpy(X_tr).float().to(device)
    lbls_tr = torch.tensor(y_tr, dtype=torch.long, device=device)
    fts_te = torch.from_numpy(X_te).float().to(device)
    lbls_te = torch.tensor(y_te, dtype=torch.long, device=device)

    # Parameters required by HyperGCN (following the style of your original HGCN script)
    args.d = X_tr.shape[1]
    args.c = int(np.max(y_tr)) + 1

    # -------------------------------------------------
    # 3) Graph construction (original train/test hypergraphs)
    # -------------------------------------------------
    hyper_tr = generate_hyperedge_dict(
        df_tr_proc,
        args.cat_cols,
        args.cont_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        device=device
    )
    hyper_te = generate_hyperedge_dict(
        df_te_proc,
        args.cat_cols,
        args.cont_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        device=device
    )

    # Some implementations return dict, some may return list directly; handle both here
    hyper_tr_list = list(hyper_tr.values()) if isinstance(hyper_tr, dict) else hyper_tr
    hyper_te_list = list(hyper_te.values()) if isinstance(hyper_te, dict) else hyper_te

    A_tr_before = laplacian(hyper_tr_list, X_tr, args.mediators).to(device)
    A_te_before = laplacian(hyper_te_list, X_te, args.mediators).to(device)

    # -------------------------------------------------
    # 4) Train the original model (Full model on original HG)
    # -------------------------------------------------
    model = HyperGCN(X_tr.shape[0], hyper_tr_list, X_tr, args).to(device)
    _set_model_structure(model, A_tr_before)

    print("== Train Full Model ==")
    t_train0 = time.time()
    model = _train_full_model_hgcn_bank(model, fts_tr, lbls_tr, args)
    train_time = time.time() - t_train0

    _set_model_structure(model, A_te_before)
    test_acc_before = _safe_eval_acc(model, fts_te, lbls_te)
    test_f1_before = _safe_eval_f1(model, fts_te, lbls_te)

    print(f"[Full] Test ACC={test_acc_before:.4f} | Test F1={test_f1_before:.4f} | train_time={train_time:.2f}s")

    # -------------------------------------------------
    # 5) Column deletion (edit train/test graph structure + features)
    # -------------------------------------------------
    cols = args.columns_to_unlearn
    if isinstance(cols, str):
        cols = [cols]
    print(f"[Column Unlearning] columns_to_unlearn={cols}")

    t_edit0 = time.time()
    X_tr_zero, hyper_tr_zero, A_tr_zero = delete_feature_columns_hgcn(
        X_tensor=fts_tr,
        transformer=transformer,
        column_names=cols,
        hyperedges=hyper_tr,   # the original function usually takes dict
        mediators=args.mediators,
        use_cuda=(device.type == "cuda"),
    )
    edit_time = time.time() - t_edit0

    X_te_zero, hyper_te_zero, A_te_zero = delete_feature_columns_hgcn(
        X_tensor=fts_te,
        transformer=transformer,
        column_names=cols,
        hyperedges=hyper_te,
        mediators=args.mediators,
        use_cuda=(device.type == "cuda"),
    )

    # unify tensor/device
    if not torch.is_tensor(X_tr_zero):
        X_tr_zero = torch.from_numpy(X_tr_zero).float().to(device)
    else:
        X_tr_zero = X_tr_zero.float().to(device)

    if not torch.is_tensor(X_te_zero):
        X_te_zero = torch.from_numpy(X_te_zero).float().to(device)
    else:
        X_te_zero = X_te_zero.float().to(device)

    A_tr_zero = A_tr_zero.to(device)
    A_te_zero = A_te_zero.to(device)

    n_h_tr_orig = len(hyper_tr) if isinstance(hyper_tr, dict) else len(hyper_tr_list)
    n_h_te_orig = len(hyper_te) if isinstance(hyper_te, dict) else len(hyper_te_list)
    n_h_tr_edit = len(hyper_tr_zero) if hasattr(hyper_tr_zero, "__len__") else -1
    n_h_te_edit = len(hyper_te_zero) if hasattr(hyper_te_zero, "__len__") else -1

    print(f"[EditedHG] train #hyperedges(orig)={n_h_tr_orig} -> #hyperedges(edit)={n_h_tr_edit}")
    print(f"[EditedHG] test  #hyperedges(orig)={n_h_te_orig} -> #hyperedges(edit)={n_h_te_edit}")

    rows = []

    # -------------------------------------------------
    # 6) Full@EditedHG (only replace the structure, no parameter update)
    # -------------------------------------------------
    full_edit_model = _clone_model(model)
    _set_model_structure(full_edit_model, A_te_zero)

    full_test_acc = _safe_eval_acc(full_edit_model, X_te_zero, lbls_te)
    full_test_f1 = _safe_eval_f1(full_edit_model, X_te_zero, lbls_te)
    mia_overall, mia_aux = _maybe_mia_hgcn_bank(full_edit_model, args, X_tr_zero, lbls_tr, X_te_zero, lbls_te)

    print(f"[Full@EditedHG] Test ACC={full_test_acc:.4f} | Test F1={full_test_f1:.4f}")

    rows.append({
        "run_id": run_id + 1,
        "method": "Full@EditedHG",
        "K": 0,
        "edit": float(edit_time),
        "update": 0.0,
        "total": float(edit_time),
        "test_acc": float(full_test_acc),
        "test_f1": float(full_test_f1) if not np.isnan(full_test_f1) else None,
        "mia_overall": None if mia_overall is None else float(mia_overall),
        "mia_aux": None if mia_aux is None else float(mia_aux),
    })

    # -------------------------------------------------
    # 7) FT-K (full-parameter finetuning, trained on EditedHG-train)
    # -------------------------------------------------
    print("\n== FT-K (warm-start on EditedHG) ==")
    for K in args.ft_steps:
        m_ft = _clone_model(model)
        _set_model_structure(m_ft, A_tr_zero)

        # all parameters trainable
        for p in m_ft.parameters():
            p.requires_grad = True

        m_ft, update_time = _finetune_steps_hgcn(
            m_ft,
            x_train=X_tr_zero,
            y_train=lbls_tr,
            K=int(K),
            lr=float(args.ft_lr),
            weight_decay=float(args.ft_wd),
            milestones=list(args.ft_milestones),
            gamma=float(args.ft_gamma),
        )

        _set_model_structure(m_ft, A_te_zero)
        test_acc = _safe_eval_acc(m_ft, X_te_zero, lbls_te)
        test_f1 = _safe_eval_f1(m_ft, X_te_zero, lbls_te)
        mia_overall, mia_aux = _maybe_mia_hgcn_bank(m_ft, args, X_tr_zero, lbls_tr, X_te_zero, lbls_te)

        total_time = edit_time + update_time
        print(
            f"[FT-K] K={int(K):4d} | Test ACC={test_acc:.4f} | Test F1={test_f1:.4f} | "
            f"edit={edit_time:.4f}s | update={update_time:.4f}s | total={total_time:.4f}s"
        )

        rows.append({
            "run_id": run_id + 1,
            "method": "FT-K@EditedHG",
            "K": int(K),
            "edit": float(edit_time),
            "update": float(update_time),
            "total": float(total_time),
            "test_acc": float(test_acc),
            "test_f1": float(test_f1) if not np.isnan(test_f1) else None,
            "mia_overall": None if mia_overall is None else float(mia_overall),
            "mia_aux": None if mia_aux is None else float(mia_aux),
        })

    # -------------------------------------------------
    # 8) FT-head (finetuning only the last layer)
    # -------------------------------------------------
    print("\n== FT-head (only train last layer) ==")
    for K in args.ft_steps:
        m_head = _clone_model(model)

        head_module = _get_last_trainable_module(m_head)
        _freeze_all_then_unfreeze(m_head, head_module)

        _set_model_structure(m_head, A_tr_zero)

        m_head, update_time = _finetune_steps_hgcn(
            m_head,
            x_train=X_tr_zero,
            y_train=lbls_tr,
            K=int(K),
            lr=float(args.ft_lr),
            weight_decay=float(args.ft_wd),
            milestones=list(args.ft_milestones),
            gamma=float(args.ft_gamma),
        )

        _set_model_structure(m_head, A_te_zero)
        test_acc = _safe_eval_acc(m_head, X_te_zero, lbls_te)
        test_f1 = _safe_eval_f1(m_head, X_te_zero, lbls_te)
        mia_overall, mia_aux = _maybe_mia_hgcn_bank(m_head, args, X_tr_zero, lbls_tr, X_te_zero, lbls_te)

        total_time = edit_time + update_time
        print(
            f"[FT-head] K={int(K):4d} | Test ACC={test_acc:.4f} | Test F1={test_f1:.4f} | "
            f"edit={edit_time:.4f}s | update={update_time:.4f}s | total={total_time:.4f}s"
        )

        rows.append({
            "run_id": run_id + 1,
            "method": "FT-head@EditedHG",
            "K": int(K),
            "edit": float(edit_time),
            "update": float(update_time),
            "total": float(total_time),
            "test_acc": float(test_acc),
            "test_f1": float(test_f1) if not np.isnan(test_f1) else None,
            "mia_overall": None if mia_overall is None else float(mia_overall),
            "mia_aux": None if mia_aux is None else float(mia_aux),
        })

    return rows


# =========================================================
#                           Main function
# =========================================================
def main():
    args = get_args()

    print("==== Bank HGCN FT (Column Unlearning) Config ====")
    show_keys = [
        "data_csv", "split_ratio", "label_col",
        "cat_cols", "cont_cols", "columns_to_unlearn",
        "max_nodes_per_hyperedge", "mediators",
        "epochs", "lr", "weight_decay", "milestones", "gamma",
        "ft_steps", "ft_lr", "ft_wd", "ft_milestones", "ft_gamma",
        "runs", "seed", "run_mia", "out_csv"
    ]
    for k in show_keys:
        print(f"{k}: {getattr(args, k)}")
    print("===============================================")

    all_rows = []
    for run_id in range(int(args.runs)):
        rows = run_one(args, run_id)
        all_rows.extend(rows)

    _save_rows_csv(all_rows, args.out_csv)
    print(f"\n[Saved] {args.out_csv}")
    _print_summary(all_rows)


if __name__ == "__main__":
    main()