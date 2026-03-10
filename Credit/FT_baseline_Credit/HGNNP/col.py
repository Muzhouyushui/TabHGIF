
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

# =========================================================
# Credit column pipeline (reuse the same as HGNN version)
# =========================================================
from Credit.HGNN.data_preprocessing_Credit_col import (
    generate_hyperedge_dict,
    preprocess_node_features,
    delete_feature_columns,   # Credit COL uses plural in your HGNN script
)

# =========================================================
# HGNNP model & sparse utils
# =========================================================
# Prefer project path: Credit/HGNNP/HGNNP.py (you gave an example module HGNNP.py)
try:
    from Credit.HGNNP.HGNNP import (
        HGNNP_implicit,
        build_incidence_matrix,
        compute_degree_vectors,
    )
except Exception:
    # fallback (if you run this script beside HGNNP.py)
    from Credit.HGNNP.HGNNP import (
        HGNNP_implicit,
        build_incidence_matrix,
        compute_degree_vectors,
    )

# =========================================================
# train/eval utils (keep consistent with your HGNN code style)
# =========================================================
try:
    from utils.common_utils import evaluate_test_acc, evaluate_test_f1
except Exception:
    try:
        from Credit.HGNN.HGNN_utils import evaluate_test_acc, evaluate_test_f1
    except Exception:
        evaluate_test_acc, evaluate_test_f1 = None, None

# =========================================================
# Optional MIA
# =========================================================
_HAS_MIA = False
_mia_fn = None
try:
    from MIA.MIA_utils import membership_inference_attack as _mia_fn
    _HAS_MIA = True
except Exception:
    try:
        from MIA.MIA_utils import membership_inference as _mia_fn
from paths import CREDIT_DATA
        _HAS_MIA = True
    except Exception:
        _HAS_MIA = False
        _mia_fn = None

# =========================================================
# Basic utils
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

def scipy_sparse_to_torch_sparse(H_sparse, device):
    H_coo = H_sparse.tocoo()
    indices = np.vstack((H_coo.row, H_coo.col)).astype(np.int64)
    values = H_coo.data.astype(np.float32)
    H_tensor = torch.sparse_coo_tensor(
        torch.from_numpy(indices),
        torch.from_numpy(values),
        size=H_coo.shape,
        dtype=torch.float32,
    ).coalesce().to(device)
    return H_tensor

def rebuild_hgnnp_struct_from_hyperedges(hyperedges_dict, num_nodes, device):
    """
    hyperedges_dict -> scipy sparse H -> degree vectors -> torch sparse H
    For HGNNP_implicit forward: (x, H, dv_inv, de_inv)
    """
    H_sparse = build_incidence_matrix(hyperedges_dict, num_nodes)
    dv_np, de_np = compute_degree_vectors(H_sparse)

    H_tensor = scipy_sparse_to_torch_sparse(H_sparse, device)
    dv_inv = torch.tensor(dv_np, dtype=torch.float32, device=device)
    de_inv = torch.tensor(de_np, dtype=torch.float32, device=device)
    return H_sparse, H_tensor, dv_inv, de_inv

# =========================================================
# Train / Eval
# =========================================================
def train_full_model_hgnnp(model, fts, lbls, H_tensor, dv_inv, de_inv, args):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.milestones, gamma=args.gamma
    )

    best_state = copy.deepcopy(model.state_dict())
    best_acc = -1.0
    t0 = time.time()

    print("== Train Full Model ==")
    for ep in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        out = model(fts, H_tensor, dv_inv, de_inv)  # HGNNP returns log_softmax
        loss = criterion(out, lbls)
        loss.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            pred = out.argmax(dim=1)
            acc = (pred == lbls).float().mean().item()

        if acc > best_acc:
            best_acc = acc
            best_state = copy.deepcopy(model.state_dict())

        if (ep + 1) % max(1, args.print_freq) == 0 or ep == 0:
            print(f"[Train] ep {ep+1:4d}/{args.epochs} | loss={loss.item():.4f} | train_acc={acc:.4f}")

    model.load_state_dict(best_state)
    train_time = time.time() - t0
    print(f"Training complete in {train_time:.2f}s")
    print(f"Best Train Acc: {best_acc:.4f}")
    return model, train_time

@torch.no_grad()
def eval_acc_hgnnp(model, x, y, H_tensor, dv_inv, de_inv):
    model.eval()
    out = model(x, H_tensor, dv_inv, de_inv)
    pred = out.argmax(dim=1)
    return float((pred == y).float().mean().item())

def freeze_all_but_head_hgnnp(model: nn.Module):
    """
    HGNNP_implicit typically has hgc1/hgc2; we treat hgc2 as head.
    """
    for p in model.parameters():
        p.requires_grad = False

    if hasattr(model, "hgc2"):
        for p in model.hgc2.parameters():
            p.requires_grad = True
        print("[FT-head] Unfreezing model.hgc2")
        return

    children = list(model.children())
    if len(children) > 0:
        for p in children[-1].parameters():
            p.requires_grad = True
        print(f"[FT-head] Fallback unfreezing last child: {children[-1].__class__.__name__}")
        return

    # fallback
    for p in model.parameters():
        p.requires_grad = True
    print("[FT-head] Warning: fallback to full finetune")

def finetune_steps_hgnnp(model, x, y, H_tensor, dv_inv, de_inv, steps, lr, wd):
    params = [p for p in model.parameters() if p.requires_grad]
    if len(params) == 0:
        raise RuntimeError("No trainable parameters for finetune.")

    criterion = nn.CrossEntropyLoss()
    opt = optim.Adam(params, lr=lr, weight_decay=wd)

    model.train()
    for _ in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        out = model(x, H_tensor, dv_inv, de_inv)
        loss = criterion(out, y)
        loss.backward()
        opt.step()
    return model

# =========================================================
# Optional MIA wrapper (same as your HGNN script)
# =========================================================
def maybe_run_mia(tag, model, args, X_train_np=None, y_train_np=None, hyperedges=None, device=None):
    if not args.run_mia:
        return None, None
    if (not _HAS_MIA) or (_mia_fn is None):
        print(f"[MIA] skipped for {tag} (MIA module not available)")
        return None, None

    print(f"— MIA on {tag} —")
    try:
        if _mia_fn.__name__ == "membership_inference":
            _, (_, _), (auc_target, f1_target) = _mia_fn(
                X_train=X_train_np,
                y_train=y_train_np,
                hyperedges=hyperedges,
                target_model=model,
                args=args,
                device=device
            )
            return float(auc_target), None
        else:
            out = _mia_fn(model=model, args=args)
            if isinstance(out, dict):
                return out.get("mia_overall", None), out.get("mia_deleted", None)
            if isinstance(out, (tuple, list)) and len(out) >= 2:
                return out[0], out[1]
            if isinstance(out, (float, int)):
                return float(out), None
            return None, None
    except Exception as e:
        print(f"[MIA] failed for {tag}: {e}")
        return None, None

# =========================================================
# Credit single-file load/split
# =========================================================
def load_credit_df(args):
    """
    Credit Approval dataset: crx.data
    Raw 16 cols: A1..A15 + class
    """
    df = pd.read_csv(
        args.data_csv,
        header=None,
        na_values="?",
        skipinitialspace=True,
    )
    df.columns = [f"A{i}" for i in range(1, 16)] + ["class"]

    # label mapping
    df["y"] = df["class"].map({"+": 1, "-": 0})
    if df["y"].isna().any():
        vals = sorted(df["class"].dropna().unique().tolist())
        if len(vals) >= 2:
            mapping = {vals[0]: 0, vals[-1]: 1}
            df["y"] = df["class"].map(mapping)
        else:
            raise ValueError("Cannot infer label mapping from Credit class column.")
    return df

def split_credit_df(df, args):
    df_train, df_test = train_test_split(
        df,
        test_size=args.split_ratio,
        random_state=args.split_seed,
        stratify=df["y"],
    )
    return df_train.reset_index(drop=True), df_test.reset_index(drop=True)

def parse_columns_to_unlearn(cols_arg):
    if isinstance(cols_arg, (list, tuple)):
        return [str(c).strip() for c in cols_arg if str(c).strip()]
    s = str(cols_arg).strip()
    if "," in s:
        return [x.strip() for x in s.split(",") if x.strip()]
    return [s]

# =========================================================
# One run
# =========================================================
def run_one(args, run_id):
    seed = args.seed + run_id
    set_seed(seed)

    device = get_device(args.device)
    print(f"[Device] {device} | seed={seed}")

    # 1) Load Credit & split
    df_full = load_credit_df(args)
    df_train_raw, df_test_raw = split_credit_df(df_full, args)

    print(f"训练集样本数: {len(df_train_raw)}, 测试集样本数: {len(df_test_raw)}")
    print("– TRAIN label dist:", Counter(df_train_raw["y"]))
    print("– TEST  label dist:", Counter(df_test_raw["y"]))

    # 2) Preprocess train/test with same transformer
    X_train_np, y_train_np, df_train_proc, transformer = preprocess_node_features(
        df_train_raw, transformer=None
    )
    X_test_np, y_test_np, df_test_proc, _ = preprocess_node_features(
        df_test_raw, transformer=transformer
    )

    X_train_np = np.asarray(X_train_np)
    y_train_np = np.asarray(y_train_np).astype(np.int64)
    X_test_np = np.asarray(X_test_np)
    y_test_np = np.asarray(y_test_np).astype(np.int64)

    print(f"预处理后 TRAIN shape: {X_train_np.shape}, labels: {Counter(y_train_np)}")
    print(f"预处理后 TEST  shape: {X_test_np.shape}, labels: {Counter(y_test_np)}")

    # Credit Approval continuous/categorical split (same as HGNN script)
    cont_cols = ["A2", "A3", "A8", "A11", "A14", "A15"]
    cat_cols = [f"A{i}" for i in range(1, 16) if f"A{i}" not in cont_cols]

    # 3) Build hypergraphs
    hyperedges_train = generate_hyperedge_dict(
        df_train_proc,
        feature_cols=cat_cols + cont_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge_train,
        device=device,
    )
    hyperedges_test = generate_hyperedge_dict(
        df_test_proc,
        feature_cols=cat_cols + cont_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge_test,
        device=device,
    )

    fts_train = torch.tensor(X_train_np, dtype=torch.float32, device=device)
    lbls_train = torch.tensor(y_train_np, dtype=torch.long, device=device)
    fts_test = torch.tensor(X_test_np, dtype=torch.float32, device=device)
    lbls_test = torch.tensor(y_test_np, dtype=torch.long, device=device)

    _, H_train, dv_train, de_train = rebuild_hgnnp_struct_from_hyperedges(
        hyperedges_train, X_train_np.shape[0], device
    )
    _, H_test, dv_test, de_test = rebuild_hgnnp_struct_from_hyperedges(
        hyperedges_test, X_test_np.shape[0], device
    )

    # 4) Train full model on original HG
    n_features = fts_train.shape[1]
    n_classes = int(np.max(y_train_np)) + 1

    model_full = HGNNP_implicit(
        in_ch=n_features,
        n_class=n_classes,
        n_hid=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    model_full, full_train_time = train_full_model_hgnnp(
        model_full, fts_train, lbls_train, H_train, dv_train, de_train, args
    )

    # original test eval (optional)
    if evaluate_test_acc is not None:
        test_obj_raw = {"x": fts_test, "y": lbls_test, "H": H_test, "dv_inv": dv_test, "de_inv": de_test}
        full_test_acc_raw = float(evaluate_test_acc(model_full, test_obj_raw))
        try:
            full_test_f1_raw = float(evaluate_test_f1(model_full, test_obj_raw))
        except Exception:
            full_test_f1_raw = None
    else:
        full_test_acc_raw = eval_acc_hgnnp(model_full, fts_test, lbls_test, H_test, dv_test, de_test)
        full_test_f1_raw = None

    print(f"[Full] Test ACC={full_test_acc_raw:.4f}"
          + (f" | F1={full_test_f1_raw:.4f}" if full_test_f1_raw is not None else "")
          + f" | train_time={full_train_time:.2f}s")

    # 5) Column unlearning on EditedHG (train + test)
    cols_to_unlearn = parse_columns_to_unlearn(args.columns_to_unlearn)
    print(f"[Column Unlearning] columns_to_unlearn={cols_to_unlearn}")

    t_edit0 = time.time()

    # train edited
    fts_train_edit = fts_train.clone()
    hyperedges_train_edit = copy.deepcopy(hyperedges_train)

    for col in cols_to_unlearn:
        out = delete_feature_columns(
            fts_train_edit,
            transformer,
            [col],  # Credit col deletion function expects list-like
            hyperedges_train_edit,
            df_train_proc,
            max_nodes_per_hyperedge=args.max_nodes_per_hyperedge_train,
            device=device,
        )
        if not (isinstance(out, (tuple, list)) and len(out) >= 2):
            raise RuntimeError("delete_feature_columns return format not recognized.")
        fts_train_edit = out[0]
        hyperedges_train_edit = out[1]

    _, H_train_edit, dv_train_edit, de_train_edit = rebuild_hgnnp_struct_from_hyperedges(
        hyperedges_train_edit, fts_train_edit.shape[0], device
    )

    # test edited
    fts_test_edit = fts_test.clone()
    hyperedges_test_edit = copy.deepcopy(hyperedges_test)

    for col in cols_to_unlearn:
        out = delete_feature_columns(
            fts_test_edit,
            transformer,
            [col],
            hyperedges_test_edit,
            df_test_proc,
            max_nodes_per_hyperedge=args.max_nodes_per_hyperedge_test,
            device=device,
        )
        if not (isinstance(out, (tuple, list)) and len(out) >= 2):
            raise RuntimeError("delete_feature_columns return format not recognized.")
        fts_test_edit = out[0]
        hyperedges_test_edit = out[1]

    _, H_test_edit, dv_test_edit, de_test_edit = rebuild_hgnnp_struct_from_hyperedges(
        hyperedges_test_edit, fts_test_edit.shape[0], device
    )

    edit_time = time.time() - t_edit0

    print(f"[EditedHG] train #hyperedges(orig)={len(hyperedges_train)} -> #hyperedges(edit)={len(hyperedges_train_edit)}")
    print(f"[EditedHG] test  #hyperedges(orig)={len(hyperedges_test)} -> #hyperedges(edit)={len(hyperedges_test_edit)}")

    # 6) Full@EditedHG
    full_edit_train_acc = eval_acc_hgnnp(model_full, fts_train_edit, lbls_train, H_train_edit, dv_train_edit, de_train_edit)
    full_edit_test_acc = eval_acc_hgnnp(model_full, fts_test_edit, lbls_test, H_test_edit, dv_test_edit, de_test_edit)
    print(f"[Full@EditedHG] Train ACC={full_edit_train_acc:.4f} | Test ACC={full_edit_test_acc:.4f}")

    rows = []

    mia_overall, mia_deleted = maybe_run_mia(
        "Full@EditedHG", model_full, args,
        X_train_np=X_train_np, y_train_np=y_train_np, hyperedges=hyperedges_train_edit, device=device
    )

    print(
        f"Full@EditedHG    K={0:4d} | edit={edit_time:.4f} | update={0.0:.4f} | total={edit_time:.4f} | "
        f"test_acc={full_edit_test_acc:.4f} | train_acc={full_edit_train_acc:.4f} | "
        f"mia_overall={'NA' if mia_overall is None else f'{mia_overall:.4f}'} | "
        f"mia_deleted={'NA' if mia_deleted is None else f'{mia_deleted:.4f}'}"
    )

    rows.append({
        "run": run_id,
        "seed": seed,
        "method": "Full@EditedHG",
        "K": 0,
        "edit": float(edit_time),
        "update": 0.0,
        "total": float(edit_time),
        "test_acc": float(full_edit_test_acc),
        "train_acc": float(full_edit_train_acc),
        "mia_overall": None if mia_overall is None else float(mia_overall),
        "mia_deleted": None if mia_deleted is None else float(mia_deleted),
    })

    # 7) FT-K
    print("\n== FT-K (warm-start on EditedHG) ==")
    for K in args.ft_steps:
        m = copy.deepcopy(model_full)
        for p in m.parameters():
            p.requires_grad = True

        t_up = time.time()
        m = finetune_steps_hgnnp(
            m, fts_train_edit, lbls_train, H_train_edit, dv_train_edit, de_train_edit,
            steps=K, lr=args.ft_lr, wd=args.ft_wd
        )
        update_time = time.time() - t_up

        train_acc = eval_acc_hgnnp(m, fts_train_edit, lbls_train, H_train_edit, dv_train_edit, de_train_edit)
        test_acc = eval_acc_hgnnp(m, fts_test_edit, lbls_test, H_test_edit, dv_test_edit, de_test_edit)
        total_time = edit_time + update_time

        mia_overall, mia_deleted = maybe_run_mia(
            f"FT-K(K={K})", m, args,
            X_train_np=X_train_np, y_train_np=y_train_np, hyperedges=hyperedges_train_edit, device=device
        )

        print(
            f"FT-K@EditedHG    K={K:4d} | edit={edit_time:.4f} | update={update_time:.4f} | total={total_time:.4f} | "
            f"test_acc={test_acc:.4f} | train_acc={train_acc:.4f} | "
            f"mia_overall={'NA' if mia_overall is None else f'{mia_overall:.4f}'} | "
            f"mia_deleted={'NA' if mia_deleted is None else f'{mia_deleted:.4f}'}"
        )

        rows.append({
            "run": run_id,
            "seed": seed,
            "method": "FT-K@EditedHG",
            "K": int(K),
            "edit": float(edit_time),
            "update": float(update_time),
            "total": float(total_time),
            "test_acc": float(test_acc),
            "train_acc": float(train_acc),
            "mia_overall": None if mia_overall is None else float(mia_overall),
            "mia_deleted": None if mia_deleted is None else float(mia_deleted),
        })

    # 8) FT-head
    print("\n== FT-head (only train last layer) ==")
    for K in args.ft_steps:
        m = copy.deepcopy(model_full)
        freeze_all_but_head_hgnnp(m)

        t_up = time.time()
        m = finetune_steps_hgnnp(
            m, fts_train_edit, lbls_train, H_train_edit, dv_train_edit, de_train_edit,
            steps=K, lr=args.ft_lr, wd=args.ft_wd
        )
        update_time = time.time() - t_up

        train_acc = eval_acc_hgnnp(m, fts_train_edit, lbls_train, H_train_edit, dv_train_edit, de_train_edit)
        test_acc = eval_acc_hgnnp(m, fts_test_edit, lbls_test, H_test_edit, dv_test_edit, de_test_edit)
        total_time = edit_time + update_time

        mia_overall, mia_deleted = maybe_run_mia(
            f"FT-head(K={K})", m, args,
            X_train_np=X_train_np, y_train_np=y_train_np, hyperedges=hyperedges_train_edit, device=device
        )

        print(
            f"FT-head@EditedHG K={K:4d} | edit={edit_time:.4f} | update={update_time:.4f} | total={total_time:.4f} | "
            f"test_acc={test_acc:.4f} | train_acc={train_acc:.4f} | "
            f"mia_overall={'NA' if mia_overall is None else f'{mia_overall:.4f}'} | "
            f"mia_deleted={'NA' if mia_deleted is None else f'{mia_deleted:.4f}'}"
        )

        rows.append({
            "run": run_id,
            "seed": seed,
            "method": "FT-head@EditedHG",
            "K": int(K),
            "edit": float(edit_time),
            "update": float(update_time),
            "total": float(total_time),
            "test_acc": float(test_acc),
            "train_acc": float(train_acc),
            "mia_overall": None if mia_overall is None else float(mia_overall),
            "mia_deleted": None if mia_deleted is None else float(mia_deleted),
        })

    return rows

# =========================================================
# Summary
# =========================================================
def format_mean_std(mean_val, std_val):
    if pd.isna(mean_val):
        return "NA"
    if pd.isna(std_val):
        return f"{mean_val:.4f}"
    return f"{mean_val:.4f}±{std_val:.4f}"

def summarize_and_save(all_rows, out_csv):
    df = pd.DataFrame(all_rows)
    df.to_csv(out_csv, index=False)

    agg_cols = ["edit", "update", "total", "test_acc", "train_acc", "mia_overall", "mia_deleted"]
    summary = df.groupby(["method", "K"], dropna=False)[agg_cols].agg(["mean", "std"]).reset_index()

    print("\n== Summary (mean±std) ==")
    for _, r in summary.iterrows():
        method = r[("method", "")]
        K = int(r[("K", "")])

        edit_s = format_mean_std(r[("edit", "mean")], r[("edit", "std")])
        upd_s = format_mean_std(r[("update", "mean")], r[("update", "std")])
        total_s = format_mean_std(r[("total", "mean")], r[("total", "std")])
        test_s = format_mean_std(r[("test_acc", "mean")], r[("test_acc", "std")])
        train_s = format_mean_std(r[("train_acc", "mean")], r[("train_acc", "std")])
        miao_s = format_mean_std(r[("mia_overall", "mean")], r[("mia_overall", "std")])
        miad_s = format_mean_std(r[("mia_deleted", "mean")], r[("mia_deleted", "std")])

        print(
            f"{method:14s} K={K:4d} | edit={edit_s} | update={upd_s} | total={total_s} | "
            f"test_acc={test_s} | train_acc={train_s} | mia_overall={miao_s} | mia_deleted={miad_s}"
        )

    print(f"\n[Saved] {out_csv}")

# =========================================================
# Args
# =========================================================
def get_args():
    p = argparse.ArgumentParser("Credit HGNNP FT baselines on edited hypergraph (column unlearning)")

    # ===== Data =====
    p.add_argument("--data_csv", type=str,
                   default=CREDIT_DATA,
                   help="Credit Approval crx.data path")

    # internal split
    p.add_argument("--split_ratio", type=float, default=0.2)
    p.add_argument("--split_seed", type=int, default=42)

    # ===== Hypergraph build =====
    p.add_argument("--max_nodes_per_hyperedge_train", type=int, default=50)
    p.add_argument("--max_nodes_per_hyperedge_test", type=int, default=50)

    # ===== Column unlearning =====
    p.add_argument("--columns_to_unlearn", type=str, default="A5",
                   help="要遗忘的原始列名；多个列用逗号分隔，例如 A5,A9")

    # ===== Model / Full train =====
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--milestones", type=int, nargs="*", default=[100, 150])
    p.add_argument("--gamma", type=float, default=0.1)
    p.add_argument("--print_freq", type=int, default=10)

    # ===== FT baselines =====
    p.add_argument("--ft_steps", type=int, nargs="+", default=[50, 100, 200])
    p.add_argument("--ft_lr", type=float, default=1e-3)
    p.add_argument("--ft_wd", type=float, default=0.0)

    # ===== MIA =====
    p.add_argument("--run_mia", action="store_true", help="是否运行MIA（默认不跑）")

    # ===== Runs / misc =====
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--out_csv", type=str, default="ft_hgnnp_col_credit_results.csv")

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
