# =========================
# File: run_ft_hgnn_col_zero.py
# HGNN FT baselines on Edited Hypergraph (Column Unlearning)
# Aligned with HGNN_Unlearning_col.py logic:
#   - use preprocess_node_features_HGNNcol / generate_hyperedge_dict / delete_feature_column
#   - build_incidence_matrix(hyperedges_dict, N) -> scipy sparse
#   - compute_degree_vectors(H_sparse)
#   - convert scipy sparse to torch sparse_coo for HGNN forward
# =========================

import os
import time
import copy
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# ====== Column-unlearning preprocessing (HGNN col pipeline) ======
from database.data_preprocessing.data_preprocessing_column import (
    preprocess_node_features_HGNNcol,
    generate_hyperedge_dict,
    delete_feature_column,
)

# ====== HGNN model & sparse utils ======
from HGNN.HGNN_2 import (
    HGNN_implicit,
    build_incidence_matrix,
    compute_degree_vectors,
)

# ====== Optional MIA (keep optional, no hard dependency) ======
_HAS_MIA = False
try:
    # 如果你项目里函数名不同，按你自己的MIA接口改这里
    from MIA.MIA_utils import membership_inference_attack as _mia_fn
    _HAS_MIA = True
except Exception:
    try:
        from MIA.MIA_utils import membership_inference as _mia_fn
from paths import ACI_TEST, ACI_TRAIN
        _HAS_MIA = True
    except Exception:
        _mia_fn = None
        _HAS_MIA = False

# -----------------------------
# Basic utils
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

# -----------------------------
# Sparse conversion / HGNN structure rebuild
# -----------------------------
def scipy_sparse_to_torch_sparse(H_sparse, device):
    """
    H_sparse: scipy sparse matrix (coo/csr/csc)
    return: torch.sparse_coo_tensor on device
    """
    H_coo = H_sparse.tocoo()
    indices = np.vstack((H_coo.row, H_coo.col)).astype(np.int64)
    values = H_coo.data.astype(np.float32)

    H_tensor = torch.sparse_coo_tensor(
        torch.from_numpy(indices),
        torch.from_numpy(values),
        size=H_coo.shape,
        dtype=torch.float32
    ).coalesce().to(device)
    return H_tensor

def rebuild_hgnn_struct_from_hyperedges(hyperedges_dict, num_nodes, device):
    """
    Strictly follow HGNN_Unlearning_col logic:
      hyperedges_dict -> build_incidence_matrix (scipy sparse)
      -> compute_degree_vectors
      -> torch sparse H_tensor
    """
    H_sparse = build_incidence_matrix(hyperedges_dict, num_nodes)   # IMPORTANT: dict input
    dv_np, de_np = compute_degree_vectors(H_sparse)

    H_tensor = scipy_sparse_to_torch_sparse(H_sparse, device)
    dv_inv = torch.tensor(dv_np, dtype=torch.float32, device=device)
    de_inv = torch.tensor(de_np, dtype=torch.float32, device=device)
    return H_sparse, H_tensor, dv_inv, de_inv

# -----------------------------
# Training / Eval
# -----------------------------
def train_full_model_hgnn(model, fts, lbls, H_tensor, dv_inv, de_inv, args):
    """
    Local train loop (NO GIF dependency)
    """
    # HGNN_implicit in your project通常输出 logits；与 HGNN_Unlearning_col 对齐用 CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.milestones, gamma=args.gamma
    )

    best_state = copy.deepcopy(model.state_dict())
    best_acc = -1.0

    print("== Train Full Model ==")
    t0 = time.time()

    for ep in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        out = model(fts, H_tensor, dv_inv, de_inv)
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
def eval_acc_hgnn(model, x, y, H_tensor, dv_inv, de_inv):
    model.eval()
    out = model(x, H_tensor, dv_inv, de_inv)
    pred = out.argmax(dim=1)
    return float((pred == y).float().mean().item())

def clone_model(model):
    return copy.deepcopy(model)

def freeze_all_but_head_hgnn(model: nn.Module):
    """
    HGNN_implicit commonly has hgc1/hgc2, treat hgc2 as head.
    Fallback: unfreeze the last child module.
    """
    for p in model.parameters():
        p.requires_grad = False

    if hasattr(model, "hgc2"):
        for p in model.hgc2.parameters():
            p.requires_grad = True
        return

    children = list(model.children())
    if len(children) > 0:
        for p in children[-1].parameters():
            p.requires_grad = True

def finetune_steps_hgnn(model, x, y, H_tensor, dv_inv, de_inv, steps, lr, wd):
    criterion = nn.CrossEntropyLoss()
    opt = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=wd)

    model.train()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        out = model(x, H_tensor, dv_inv, de_inv)
        loss = criterion(out, y)
        loss.backward()
        opt.step()
    return model

# -----------------------------
# Optional MIA wrapper
# -----------------------------
def maybe_run_mia(tag, model, args, X_train_np=None, y_train_np=None, hyperedges=None, device=None):
    if not args.run_mia:
        return None, None
    if not _HAS_MIA or _mia_fn is None:
        print(f"[MIA] skipped for {tag} (MIA module not available)")
        return None, None

    print(f"— MIA on {tag} —")
    try:
        # 这里兼容你项目里两种常见接口；如果不匹配，直接回NA
        if _mia_fn.__name__ == "membership_inference":
            # MIA_utils.membership_inference style (example)
            _, (_, _), (auc_target, f1_target) = _mia_fn(
                X_train=X_train_np,
                y_train=y_train_np,
                hyperedges=hyperedges,
                target_model=model,
                args=args,
                device=device
            )
            # 这里的 deleted-specific MIA 如果你接口没有，就先返回同值/None
            return float(auc_target), None
        else:
            # custom MIA_HGNN style - 你可按自己的接口改
            out = _mia_fn(model=model, args=args)
            if isinstance(out, dict):
                return out.get("mia_overall", None), out.get("mia_deleted", None)
            if isinstance(out, (list, tuple)) and len(out) >= 2:
                return out[0], out[1]
            if isinstance(out, (float, int)):
                return float(out), None
            return None, None
    except Exception as e:
        print(f"[MIA] failed for {tag}: {e}")
        return None, None

# -----------------------------
# Arg helpers
# -----------------------------
def parse_columns_to_unlearn(cols_arg):
    if isinstance(cols_arg, (list, tuple)):
        return [str(c).strip() for c in cols_arg if str(c).strip()]
    s = str(cols_arg).strip()
    if "," in s:
        return [x.strip() for x in s.split(",") if x.strip()]
    return [s]

def format_mean_std(mean_val, std_val):
    if pd.isna(mean_val):
        return "NA"
    if pd.isna(std_val):
        return f"{mean_val:.4f}"
    return f"{mean_val:.4f}±{std_val:.4f}"

# -----------------------------
# One run
# -----------------------------
def run_one(args, run_id):
    seed = args.seed + run_id
    set_seed(seed)

    device = torch.device(args.device if (torch.cuda.is_available() and "cuda" in args.device) else "cpu")
    print(f"[Device] {device}")

    # ==========================================================
    # 1) Load TRAIN / preprocess (HGNN column pipeline)
    # ==========================================================
    X_train_np, y_train_np, df_train, transformer = preprocess_node_features_HGNNcol(
        args.train_csv, is_test=False
    )
    X_train_np = to_numpy(X_train_np)
    y_train_np = to_numpy(y_train_np).astype(np.int64)

    # Build TRAIN hyperedges
    hyperedges_train = generate_hyperedge_dict(
        df_train,
        args.cat_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge_train,
        device=device
    )

    fts_train = torch.tensor(X_train_np, dtype=torch.float32, device=device)
    lbls_train = torch.tensor(y_train_np, dtype=torch.long, device=device)

    _, H_train, dv_train, de_train = rebuild_hgnn_struct_from_hyperedges(
        hyperedges_train, X_train_np.shape[0], device
    )

    # ==========================================================
    # 2) Load TEST / preprocess with same transformer
    # ==========================================================
    X_test_np, y_test_np, df_test, _ = preprocess_node_features_HGNNcol(
        args.test_csv, is_test=True, transformer=transformer
    )
    X_test_np = to_numpy(X_test_np)
    y_test_np = to_numpy(y_test_np).astype(np.int64)

    hyperedges_test = generate_hyperedge_dict(
        df_test,
        args.cat_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge_test,
        device=device
    )

    fts_test = torch.tensor(X_test_np, dtype=torch.float32, device=device)
    lbls_test = torch.tensor(y_test_np, dtype=torch.long, device=device)

    _, H_test, dv_test, de_test = rebuild_hgnn_struct_from_hyperedges(
        hyperedges_test, X_test_np.shape[0], device
    )

    # ==========================================================
    # 3) Train Full model on ORIGINAL train HG
    # ==========================================================
    n_features = fts_train.shape[1]
    n_classes = int(np.max(y_train_np)) + 1

    model_full = HGNN_implicit(
        in_ch=n_features,
        n_class=n_classes,
        n_hid=args.hidden_dim,
        dropout=args.dropout
    ).to(device)

    model_full, full_train_time = train_full_model_hgnn(
        model_full, fts_train, lbls_train, H_train, dv_train, de_train, args
    )

    full_test_acc = eval_acc_hgnn(model_full, fts_test, lbls_test, H_test, dv_test, de_test)
    print(f"[Full] Test ACC={full_test_acc:.4f} | train_time={full_train_time:.2f}s")

    # ==========================================================
    # 4) Column Unlearning (EditedHG) -- STRICTLY use delete_feature_column
    # ==========================================================
    cols_to_unlearn = parse_columns_to_unlearn(args.columns_to_unlearn)
    print(f"[Column Unlearning] columns_to_unlearn={cols_to_unlearn}")

    # 当前 delete_feature_column 通常一次处理一列；多列则串行应用
    fts_train_edit = fts_train.clone()
    hyperedges_train_edit = copy.deepcopy(hyperedges_train)
    H_train_edit = H_train

    t_edit0 = time.time()
    for col in cols_to_unlearn:
        out = delete_feature_column(
            fts_train_edit,
            transformer,
            col,
            H_train_edit,
            hyperedges_train_edit,
            continuous_cols=args.continuous_cols
        )
        # 兼容不同返回格式：常见是 (fts_new, H_new, hyperedges_new)
        if isinstance(out, (tuple, list)) and len(out) == 3:
            fts_train_edit, H_train_edit, hyperedges_train_edit = out
        else:
            raise RuntimeError("delete_feature_column return format not recognized (expected 3 values).")

    edit_time = time.time() - t_edit0

    # HGNN col逻辑：用新hyperedges重建 H_sparse/dv/de（train）
    _, H_train_edit, dv_train_edit, de_train_edit = rebuild_hgnn_struct_from_hyperedges(
        hyperedges_train_edit, fts_train_edit.shape[0], device
    )

    # test set do the same column deletion
    fts_test_edit = fts_test.clone()
    hyperedges_test_edit = copy.deepcopy(hyperedges_test)
    H_test_edit = H_test
    for col in cols_to_unlearn:
        out = delete_feature_column(
            fts_test_edit,
            transformer,
            col,
            H_test_edit,
            hyperedges_test_edit,
            continuous_cols=args.continuous_cols
        )
        if isinstance(out, (tuple, list)) and len(out) == 3:
            fts_test_edit, H_test_edit, hyperedges_test_edit = out
        else:
            raise RuntimeError("delete_feature_column return format not recognized (expected 3 values).")

    _, H_test_edit, dv_test_edit, de_test_edit = rebuild_hgnn_struct_from_hyperedges(
        hyperedges_test_edit, fts_test_edit.shape[0], device
    )

    print(f"[EditedHG] train #hyperedges(orig)={len(hyperedges_train)} -> #hyperedges(edit)={len(hyperedges_train_edit)}")
    print(f"[EditedHG] test  #hyperedges(orig)={len(hyperedges_test)} -> #hyperedges(edit)={len(hyperedges_test_edit)}")

    # ==========================================================
    # 5) Full@EditedHG evaluation (no update)
    # ==========================================================
    full_edit_train_acc = eval_acc_hgnn(model_full, fts_train_edit, lbls_train, H_train_edit, dv_train_edit, de_train_edit)
    full_edit_test_acc  = eval_acc_hgnn(model_full, fts_test_edit,  lbls_test,  H_test_edit,  dv_test_edit,  de_test_edit)
    print(f"[Full@EditedHG] Train ACC={full_edit_train_acc:.4f} | Test ACC={full_edit_test_acc:.4f}")

    rows = []
    mia_overall, mia_deleted = maybe_run_mia(
        "Full@EditedHG", model_full, args,
        X_train_np=X_train_np, y_train_np=y_train_np, hyperedges=hyperedges_train_edit, device=device
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

    # ==========================================================
    # 6) FT-K (warm-start, all params trainable) on EditedHG
    # ==========================================================
    print("\n== FT-K (warm-start on EditedHG) ==")
    for K in args.ft_steps:
        m = clone_model(model_full)
        for p in m.parameters():
            p.requires_grad = True

        t_up = time.time()
        m = finetune_steps_hgnn(
            m, fts_train_edit, lbls_train, H_train_edit, dv_train_edit, de_train_edit,
            steps=K, lr=args.ft_lr, wd=args.ft_wd
        )
        update_time = time.time() - t_up

        train_acc = eval_acc_hgnn(m, fts_train_edit, lbls_train, H_train_edit, dv_train_edit, de_train_edit)
        test_acc = eval_acc_hgnn(m, fts_test_edit, lbls_test, H_test_edit, dv_test_edit, de_test_edit)
        total_time = edit_time + update_time

        print(f"[FT-K] K={K:4d} | Train ACC={train_acc:.4f} | Test ACC={test_acc:.4f} "
              f"| edit={edit_time:.4f}s | update={update_time:.4f}s | total={total_time:.4f}s")

        mia_overall, mia_deleted = maybe_run_mia(
            f"FT-K(K={K})", m, args,
            X_train_np=X_train_np, y_train_np=y_train_np, hyperedges=hyperedges_train_edit, device=device
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

    # ==========================================================
    # 7) FT-head (only last layer) on EditedHG
    # ==========================================================
    print("\n== FT-head (only train last layer) ==")
    for K in args.ft_steps:
        m = clone_model(model_full)
        freeze_all_but_head_hgnn(m)

        t_up = time.time()
        m = finetune_steps_hgnn(
            m, fts_train_edit, lbls_train, H_train_edit, dv_train_edit, de_train_edit,
            steps=K, lr=args.ft_lr, wd=args.ft_wd
        )
        update_time = time.time() - t_up

        train_acc = eval_acc_hgnn(m, fts_train_edit, lbls_train, H_train_edit, dv_train_edit, de_train_edit)
        test_acc = eval_acc_hgnn(m, fts_test_edit, lbls_test, H_test_edit, dv_test_edit, de_test_edit)
        total_time = edit_time + update_time

        print(f"[FT-head] K={K:4d} | Train ACC={train_acc:.4f} | Test ACC={test_acc:.4f} "
              f"| edit={edit_time:.4f}s | update={update_time:.4f}s | total={total_time:.4f}s")

        mia_overall, mia_deleted = maybe_run_mia(
            f"FT-head(K={K})", m, args,
            X_train_np=X_train_np, y_train_np=y_train_np, hyperedges=hyperedges_train_edit, device=device
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

    print("\nDone.")
    return rows

# -----------------------------
# Summary + save
# -----------------------------
def summarize_and_save(all_rows, out_csv):
    df = pd.DataFrame(all_rows)
    df.to_csv(out_csv, index=False)

    agg_cols = ["edit", "update", "total", "test_acc", "train_acc", "mia_overall", "mia_deleted"]
    summary = df.groupby(["method", "K"], dropna=False)[agg_cols].agg(["mean", "std"]).reset_index()

    print("\n== Summary (mean±std) ==")
    for _, r in summary.iterrows():
        method = r[("method", "")]
        K = int(r[("K", "")])

        edit_s   = format_mean_std(r[("edit", "mean")],       r[("edit", "std")])
        upd_s    = format_mean_std(r[("update", "mean")],     r[("update", "std")])
        total_s  = format_mean_std(r[("total", "mean")],      r[("total", "std")])
        test_s   = format_mean_std(r[("test_acc", "mean")],   r[("test_acc", "std")])
        train_s  = format_mean_std(r[("train_acc", "mean")],  r[("train_acc", "std")])
        miao_s   = format_mean_std(r[("mia_overall", "mean")], r[("mia_overall", "std")])
        miad_s   = format_mean_std(r[("mia_deleted", "mean")], r[("mia_deleted", "std")])

        print(
            f"{method:14s} K={K:4d} | edit={edit_s} | update={upd_s} | total={total_s} "
            f"| test_acc={test_s} | train_acc={train_s} | mia_overall={miao_s} | mia_deleted={miad_s}"
        )

    print(f"\n[Saved] {out_csv}")

# -----------------------------
# Args
# -----------------------------
def get_args():
    p = argparse.ArgumentParser("HGNN FT baselines on edited hypergraph (column unlearning)")

    # ===== Data =====
    p.add_argument("--train_csv", type=str,
                   default=ACI_TRAIN,
                   help="训练数据路径（adult.data）")
    p.add_argument("--test_csv", type=str,
                   default=ACI_TEST,
                   help="测试数据路径（adult.test）")

    # ===== Hypergraph build (match HGNN col pipeline) =====
    p.add_argument("--cat_cols", type=str, nargs="+", default=[
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ])
    # continuous_cols: 需要和 data_preprocessing_column.delete_feature_column 接口一致
    p.add_argument("--continuous_cols", type=str, nargs="*", default=[
        "age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"
    ])
    p.add_argument("--max_nodes_per_hyperedge_train", type=int, default=10000)
    p.add_argument("--max_nodes_per_hyperedge_test",  type=int, default=10000)

    # ===== Column unlearning =====
    p.add_argument("--columns_to_unlearn", type=str, default="education",
                   help="要遗忘的原始列名；多个列用逗号分隔，例如 education,occupation")

    # ===== Model / Full train =====
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--milestones", type=int, nargs="*", default=[100, 150])
    p.add_argument("--gamma", type=float, default=0.1)
    p.add_argument("--print_freq", type=int, default=10)

    # ===== FT baselines =====
    p.add_argument("--ft_steps", type=int, nargs="+", default=[50, 100, 200])
    p.add_argument("--ft_lr", type=float, default=1e-3)
    p.add_argument("--ft_wd", type=float, default=0.0)

    # ===== Runs / misc =====
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--run_mia", action="store_true", help="是否运行MIA（默认不跑）")
    p.add_argument("--out_csv", type=str, default="ft_hgnn_col_zero_results.csv")

    return p.parse_args()

# -----------------------------
# Main
# -----------------------------
def main():
    args = get_args()

    if not os.path.exists(args.train_csv):
        print(f"[WARN] train_csv not found: {args.train_csv}")
    if not os.path.exists(args.test_csv):
        print(f"[WARN] test_csv not found: {args.test_csv}")

    all_rows = []
    for run_id in range(args.runs):
        print(f"\n================= RUN {run_id+1}/{args.runs} =================")
        rows = run_one(args, run_id)
        all_rows.extend(rows)

    summarize_and_save(all_rows, args.out_csv)

if __name__ == "__main__":
    main()