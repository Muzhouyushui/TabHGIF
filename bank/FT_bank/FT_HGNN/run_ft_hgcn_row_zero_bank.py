# =========================
# File: run_ft_hgnn_row_bank.py
# HGNN FT baseline for Bank dataset (ROW deletion, Edited Hypergraph)
# Built by combining:
#   1) HGNN_Unlearning_row_bank.py (Bank preprocessing / hypergraph pipeline)
#   2) run_ft_hgnn_row_zero.py (FT-K / FT-head baseline logic)
# =========================

import os
import time
import copy
import argparse
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# ===== Bank HGNN pipeline imports (from your bank codebase) =====
from bank.HGNN.data_preprocessing_bank import (
    preprocess_node_features_bank,
    generate_hyperedge_dict_bank,
)
from bank.HGNN.HGNN import (
    HGNN_implicit,
    build_incidence_matrix,
    compute_degree_vectors,
)
from bank.HGNN.GIF_HGNN_ROW import (
    rebuild_structure_after_node_deletion,   # reuse only structure rebuild + train_model
    train_model,
)

# (Optional) MIA: enable it if your project has the corresponding implementation
try:
    from MIA.MIA_utils import membership_inference
    _HAS_MIA = True
except Exception:
    _HAS_MIA = False


# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_masks(N: int, deleted: torch.Tensor, device: torch.device):
    del_mask = torch.zeros(N, dtype=torch.bool, device=device)
    del_mask[deleted] = True
    retain_mask = ~del_mask
    return retain_mask, del_mask


def freeze_all_but_head_hgnn(model: nn.Module):
    """
    HGNN_implicit usually has hgc1 and hgc2; treat hgc2 as the head.
    """
    for p in model.parameters():
        p.requires_grad = False

    if hasattr(model, "hgc2"):
        for p in model.hgc2.parameters():
            p.requires_grad = True
    else:
        # fallback: unfreeze the last child
        children = list(model.children())
        if len(children) > 0:
            for p in children[-1].parameters():
                p.requires_grad = True


def finetune_steps_hgnn(
    model: nn.Module,
    x_after: torch.Tensor,          # feature-zero version for deleted rows (FT input only)
    y: torch.Tensor,
    retain_mask: torch.Tensor,
    H_edit: torch.Tensor,
    dv_edit: torch.Tensor,
    de_edit: torch.Tensor,
    steps: int,
    lr: float,
    weight_decay: float
):
    """
    Blind retain-only finetuning for K steps:
      loss = CE/NLL on retained nodes only, using the edited hypergraph + zeroed deleted-node features.
    """
    model.train()
    opt = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay
    )

    # HGNN_implicit output in your codebase is usually log_softmax
    loss_fn = nn.NLLLoss()

    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        out = model(x_after, H_edit, dv_edit, de_edit)
        loss = loss_fn(out[retain_mask], y[retain_mask])
        loss.backward()
        opt.step()

    return model


@torch.no_grad()
def eval_acc_masked(model, x, y, mask, H, dv_inv, de_inv):
    model.eval()
    out = model(x, H, dv_inv, de_inv)
    pred = out.argmax(dim=1)
    if mask.sum().item() == 0:
        return float("nan")
    return float((pred[mask] == y[mask]).float().mean().item())


def _clone_model(model: nn.Module):
    return copy.deepcopy(model)


def _maybe_mia_overall_and_deleted(
    tag,
    model,
    X_train_np,
    y_train_np,
    df_train_raw,
    deleted_idx_np,
    args,
    device
):
    """
    Returns:
      mia_overall, mia_deleted
    If no MIA implementation is available or args.run_mia=False -> (None, None)

    Notes:
    - 'overall' here follows the style of your row FT script: run MIA on the current training graph/data.
    - 'deleted' here is a Keep-vs-Deleted style attack set (if membership_inference supports member_mask).
      If not supported by your implementation, fall back to None.
    """
    if (not args.run_mia) or (not _HAS_MIA):
        return None, None

    try:
        # ----- overall MIA on the original training set hypergraph -----
        cat_cols = [
            "job", "marital", "education", "default",
            "housing", "loan", "contact", "month", "poutcome"
        ]
        cont_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

        hyperedges_full = generate_hyperedge_dict_bank(
            df_train_raw,
            cat_cols,
            cont_cols,
            max_nodes_per_hyperedge=getattr(args, "max_nodes_per_hyperedge_train", 50),
            device=device
        )

        print(f"— MIA on {tag} (overall) —")
        _, (_, _), (auc_overall, _) = membership_inference(
            X_train=X_train_np,
            y_train=y_train_np,
            hyperedges=hyperedges_full,
            target_model=model,
            args=args,
            device=device
        )
        mia_overall = float(auc_overall)

    except Exception as e:
        print(f"[WARN] overall MIA failed on {tag}: {e}")
        mia_overall = None

    # ----- deleted-only / keep-vs-deleted MIA -----
    mia_deleted = None
    try:
        # If your membership_inference supports member_mask, construct a keep+deleted attack set
        all_idx = np.arange(len(X_train_np))
        keep_idx = np.setdiff1d(all_idx, deleted_idx_np)

        X_keep, y_keep = X_train_np[keep_idx], y_train_np[keep_idx]
        X_del,  y_del  = X_train_np[deleted_idx_np], y_train_np[deleted_idx_np]

        df_keep = df_train_raw.iloc[keep_idx].reset_index(drop=True)
        df_del  = df_train_raw.iloc[deleted_idx_np].reset_index(drop=True)

        cat_cols = [
            "job", "marital", "education", "default",
            "housing", "loan", "contact", "month", "poutcome"
        ]
        cont_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

        he_keep = generate_hyperedge_dict_bank(
            df_keep, cat_cols, cont_cols,
            max_nodes_per_hyperedge=getattr(args, "max_nodes_per_hyperedge_train", 50),
            device=device
        )
        he_del = generate_hyperedge_dict_bank(
            df_del, cat_cols, cont_cols,
            max_nodes_per_hyperedge=getattr(args, "max_nodes_per_hyperedge_train", 50),
            device=device
        )

        # merge hyperedges with offset for the deleted subset
        he_attack = {}
        eid = 0
        for nodes in he_keep.values():
            he_attack[eid] = nodes
            eid += 1
        offset = len(X_keep)
        for nodes in he_del.values():
            he_attack[eid] = [n + offset for n in nodes]
            eid += 1

        X_attack = np.vstack([X_keep, X_del])
        y_attack = np.hstack([y_keep, y_del])
        member_mask = np.concatenate([
            np.ones(len(X_keep), dtype=bool),
            np.zeros(len(X_del), dtype=bool)
        ])

        print(f"— MIA on {tag} (keep-vs-deleted) —")
        _, _, (auc_deleted, _) = membership_inference(
            X_train=X_attack,
            y_train=y_attack,
            hyperedges=he_attack,
            target_model=model,
            args=args,
            device=device,
            member_mask=member_mask
        )
        mia_deleted = float(auc_deleted)

    except TypeError:
        # membership_inference may not support member_mask
        mia_deleted = None
    except Exception as e:
        print(f"[WARN] deleted MIA failed on {tag}: {e}")
        mia_deleted = None

    return mia_overall, mia_deleted


# -----------------------------
# Bank data loading helpers
# -----------------------------
def split_bank_train_test(df_full: pd.DataFrame, test_ratio: float, seed: int):
    """
    Match the style of HGNN_Unlearning_row_bank.py: split from one bank-full.csv file.
    """
    df_tr, df_te = train_test_split(
        df_full,
        test_size=test_ratio,
        random_state=seed,
        stratify=df_full["y"]
    )
    df_tr = df_tr.reset_index(drop=True)
    df_te = df_te.reset_index(drop=True)
    return df_tr, df_te


def print_label_stats_bank(df_tr, df_te):
    tr_counter = Counter(df_tr["y"].tolist())
    te_counter = Counter(df_te["y"].tolist())
    print(f"Training label distribution: {dict(tr_counter)}")
    print(f"Test label distribution: {dict(te_counter)}")


# -----------------------------
# One run
# -----------------------------
def run_one(args, run_id: int):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    seed = args.seed + run_id
    set_seed(seed)

    # ===== Load Bank CSV and split (single-file dataset style) =====
    df_full = pd.read_csv(args.data_csv, sep=";", skipinitialspace=True)
    df_tr_raw, df_te_raw = split_bank_train_test(df_full, args.split_ratio, seed=42)  # fixed split like many scripts
    print(f"Training set: {len(df_tr_raw)} samples, Test set: {len(df_te_raw)} samples")
    print_label_stats_bank(df_tr_raw, df_te_raw)

    # ===== Preprocess (Bank pipeline) =====
    # training set: fit transformer
    X_tr, y_tr_np, df_tr_proc, transformer = preprocess_node_features_bank(df_tr_raw, is_test=False)
    # test set: use the same transformer
    X_te, y_te_np, df_te_proc, _ = preprocess_node_features_bank(df_te_raw, is_test=True, transformer=transformer)

    X_tr = np.asarray(X_tr)
    y_tr_np = np.asarray(y_tr_np)
    X_te = np.asarray(X_te)
    y_te_np = np.asarray(y_te_np)

    # ===== Build hyperedges on RAW dataframes (important) =====
    cat_cols = [
        "job", "marital", "education", "default",
        "housing", "loan", "contact", "month", "poutcome"
    ]
    cont_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

    hyperedges_tr = generate_hyperedge_dict_bank(
        df_tr_raw,
        cat_cols,
        cont_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge_train,
        device=device
    )
    hyperedges_te = generate_hyperedge_dict_bank(
        df_te_raw,
        cat_cols,
        cont_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge_test,
        device=device
    )

    # ===== Tensors =====
    fts_tr = torch.from_numpy(X_tr).float().to(device)
    lbls_tr = torch.from_numpy(y_tr_np).long().to(device)

    fts_te = torch.from_numpy(X_te).float().to(device)
    lbls_te = torch.from_numpy(y_te_np).long().to(device)

    N = X_tr.shape[0]
    C = int(y_tr_np.max()) + 1

    # ===== Build ORIGINAL train/test hypergraph structure =====
    H_tr, dv_tr, de_tr, _ = rebuild_structure_after_node_deletion(
        hyperedges_tr, np.array([], dtype=np.int64), N, device
    )
    H_te, dv_te, de_te, _ = rebuild_structure_after_node_deletion(
        hyperedges_te, np.array([], dtype=np.int64), X_te.shape[0], device
    )

    # ===== Train full model on the ORIGINAL training graph =====
    model_full = HGNN_implicit(
        in_ch=fts_tr.shape[1],
        n_class=C,
        n_hid=args.hidden_dim,
        dropout=args.dropout
    ).to(device)

    optimizer = optim.Adam(model_full.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=args.milestones,
        gamma=args.gamma
    )
    criterion = nn.NLLLoss()

    print("== Train Full Model ==")
    t0 = time.time()
    model_full = train_model(
        model_full, criterion, optimizer, scheduler,
        fts_tr, lbls_tr, H_tr, dv_tr, de_tr,
        num_epochs=args.epochs,
        print_freq=args.print_freq
    )
    full_train_time = time.time() - t0

    acc_test = eval_acc_masked(
        model_full, fts_te, lbls_te,
        torch.ones(X_te.shape[0], dtype=torch.bool, device=device),
        H_te, dv_te, de_te
    )
    print(f"[Full] Test ACC={acc_test:.4f} | train_time={full_train_time:.2f}s")

    # ===== Random ROW deletion =====
    n_del = int(N * args.remove_ratio)
    rng = np.random.default_rng(seed)
    deleted_idx = rng.choice(np.arange(N), size=n_del, replace=False)
    deleted_idx = np.sort(deleted_idx)
    deleted = torch.tensor(deleted_idx, dtype=torch.long, device=device)
    print(f"[Delete] remove_ratio={args.remove_ratio}, deleted={len(deleted)}")

    retain_mask, del_mask = build_masks(N, deleted, device)

    # ===== Edited hypergraph after row deletion =====
    t_edit0 = time.time()
    H_edit, dv_edit, de_edit, hyperedges_edit = rebuild_structure_after_node_deletion(
        hyperedges_tr, deleted_idx.astype(np.int64), N, device
    )
    edit_time = time.time() - t_edit0

    print(f"[EditedHG] #hyperedges(orig)={len(hyperedges_tr)} -> #hyperedges(edit)={len(hyperedges_edit)}")

    # ===== Feature-zero for deleted rows (FT input only) =====
    # (aligned with your row-zero FT setting)
    fts_tr_after = fts_tr.clone()
    fts_tr_after[deleted] = 0.0

    # ===== Baseline before FT: Full model evaluated on EditedHG =====
    acc_ret = eval_acc_masked(model_full, fts_tr, lbls_tr, retain_mask, H_edit, dv_edit, de_edit)
    acc_for = eval_acc_masked(model_full, fts_tr, lbls_tr, del_mask,    H_edit, dv_edit, de_edit)
    print(f"[Full@EditedHG] Retain ACC={acc_ret:.4f} | Forget ACC={acc_for:.4f}")

    results = []

    # Full@EditedHG record
    mia_o, mia_d = _maybe_mia_overall_and_deleted(
        "Full@EditedHG",
        model_full,
        X_tr, y_tr_np,
        df_tr_raw,
        deleted_idx,
        args,
        device
    )
    results.append({
        "run": run_id,
        "method": "Full@EditedHG",
        "K": 0,
        "edit_sec": edit_time,
        "update_sec": 0.0,
        "total_sec": edit_time,
        "test_acc": acc_test,
        "retain_acc": acc_ret,
        "forget_acc": acc_for,
        "mia_overall": mia_o,
        "mia_deleted": mia_d,
        "seed": seed
    })

    # ===== FT-K =====
    print("\n== FT-K (warm-start on EditedHG) ==")
    for K in args.ft_steps:
        m = _clone_model(model_full)
        for p in m.parameters():
            p.requires_grad = True

        t1 = time.time()
        m = finetune_steps_hgnn(
            m,
            x_after=fts_tr_after,
            y=lbls_tr,
            retain_mask=retain_mask,
            H_edit=H_edit, dv_edit=dv_edit, de_edit=de_edit,
            steps=K,
            lr=args.ft_lr,
            weight_decay=args.ft_wd
        )
        update_time = time.time() - t1
        total_time = edit_time + update_time

        acc_test_k = eval_acc_masked(
            m, fts_te, lbls_te,
            torch.ones(X_te.shape[0], dtype=torch.bool, device=device),
            H_te, dv_te, de_te
        )
        acc_ret_k = eval_acc_masked(m, fts_tr, lbls_tr, retain_mask, H_edit, dv_edit, de_edit)
        acc_for_k = eval_acc_masked(m, fts_tr, lbls_tr, del_mask,    H_edit, dv_edit, de_edit)

        print(f"[FT-K] K={K:4d} | Test ACC={acc_test_k:.4f} | Retain ACC={acc_ret_k:.4f} | Forget ACC={acc_for_k:.4f} "
              f"| edit={edit_time:.4f}s | update={update_time:.4f}s | total={total_time:.4f}s")

        mia_o, mia_d = _maybe_mia_overall_and_deleted(
            f"FT-K(K={K})",
            m,
            X_tr, y_tr_np,
            df_tr_raw,
            deleted_idx,
            args,
            device
        )

        results.append({
            "run": run_id,
            "method": "FT-K@EditedHG",
            "K": K,
            "edit_sec": edit_time,
            "update_sec": update_time,
            "total_sec": total_time,
            "test_acc": acc_test_k,
            "retain_acc": acc_ret_k,
            "forget_acc": acc_for_k,
            "mia_overall": mia_o,
            "mia_deleted": mia_d,
            "seed": seed
        })

    # ===== FT-head =====
    print("\n== FT-head (only train last layer) ==")
    for K in args.ft_steps:
        m = _clone_model(model_full)
        freeze_all_but_head_hgnn(m)

        t1 = time.time()
        m = finetune_steps_hgnn(
            m,
            x_after=fts_tr_after,
            y=lbls_tr,
            retain_mask=retain_mask,
            H_edit=H_edit, dv_edit=dv_edit, de_edit=de_edit,
            steps=K,
            lr=args.ft_lr,
            weight_decay=args.ft_wd
        )
        update_time = time.time() - t1
        total_time = edit_time + update_time

        acc_test_k = eval_acc_masked(
            m, fts_te, lbls_te,
            torch.ones(X_te.shape[0], dtype=torch.bool, device=device),
            H_te, dv_te, de_te
        )
        acc_ret_k = eval_acc_masked(m, fts_tr, lbls_tr, retain_mask, H_edit, dv_edit, de_edit)
        acc_for_k = eval_acc_masked(m, fts_tr, lbls_tr, del_mask,    H_edit, dv_edit, de_edit)

        print(f"[FT-head] K={K:4d} | Test ACC={acc_test_k:.4f} | Retain ACC={acc_ret_k:.4f} | Forget ACC={acc_for_k:.4f} "
              f"| edit={edit_time:.4f}s | update={update_time:.4f}s | total={total_time:.4f}s")

        mia_o, mia_d = _maybe_mia_overall_and_deleted(
            f"FT-head(K={K})",
            m,
            X_tr, y_tr_np,
            df_tr_raw,
            deleted_idx,
            args,
            device
        )

        results.append({
            "run": run_id,
            "method": "FT-head@EditedHG",
            "K": K,
            "edit_sec": edit_time,
            "update_sec": update_time,
            "total_sec": total_time,
            "test_acc": acc_test_k,
            "retain_acc": acc_ret_k,
            "forget_acc": acc_for_k,
            "mia_overall": mia_o,
            "mia_deleted": mia_d,
            "seed": seed
        })

    print("\nDone.\n")
    return results


# -----------------------------
# Summary / Save
# -----------------------------
def summarize_and_save(all_rows, out_csv):
    df = pd.DataFrame(all_rows)
    df.to_csv(out_csv, index=False)

    agg_cols = [
        "edit_sec", "update_sec", "total_sec",
        "test_acc", "retain_acc", "forget_acc",
        "mia_overall", "mia_deleted"
    ]
    g = df.groupby(["method", "K"], dropna=False)[agg_cols].agg(["mean", "std"]).reset_index()

    print("== Summary (mean±std) ==")
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
            f"{method:14s} K={int(K):4d} | "
            f"edit={fmt('edit_sec')} | update={fmt('update_sec')} | total={fmt('total_sec')} | "
            f"test_acc={fmt('test_acc')} | retain_acc={fmt('retain_acc')} | forget_acc={fmt('forget_acc')} | "
            f"mia_overall={fmt('mia_overall')} | mia_deleted={fmt('mia_deleted')}"
        )

    print(f"\n[Saved] {out_csv}")


# -----------------------------
# Args
# -----------------------------
def get_args():
    p = argparse.ArgumentParser("Bank HGNN FT baseline (ROW deletion) on edited hypergraph")

    # Bank single-file dataset
    p.add_argument(
        "--data_csv", type=str,
        default="/root/autodl-tmp/TabHGIF/data_banking/bank/bank-full.csv",
        help="Bank Marketing CSV path (; separated)"
    )
    p.add_argument("--split_ratio", type=float, default=0.2, help="test split ratio")

    # hypergraph build
    p.add_argument("--max_nodes_per_hyperedge_train", type=int, default=50)
    p.add_argument("--max_nodes_per_hyperedge_test",  type=int, default=50)

    # model/train
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--milestones", type=int, nargs="*", default=[100, 150])
    p.add_argument("--gamma", type=float, default=0.1)
    p.add_argument("--print_freq", type=int, default=10)

    # deletion + runs
    p.add_argument("--remove_ratio", type=float, default=0.10)  # bank row script often uses 10%
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--seed", type=int, default=1)

    # finetune
    p.add_argument("--ft_steps", type=int, nargs="+", default=[50, 100, 200])
    p.add_argument("--ft_lr", type=float, default=1e-3)
    p.add_argument("--ft_wd", type=float, default=0.0)

    # MIA (optional)
    p.add_argument("--run_mia", action="store_true", default=True)

    # misc
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--out_csv", type=str, default="ft_hgnn_row_bank_results.csv")

    return p.parse_args()


# -----------------------------
# Main
# -----------------------------
def main():
    args = get_args()

    if not os.path.exists(args.data_csv):
        raise FileNotFoundError(f"Bank CSV not found: {args.data_csv}")

    all_rows = []
    for run_id in range(args.runs):
        print(f"\n================= RUN {run_id+1}/{args.runs} =================")
        rows = run_one(args, run_id)
        all_rows.extend(rows)

    summarize_and_save(all_rows, args.out_csv)


if __name__ == "__main__":
    main()