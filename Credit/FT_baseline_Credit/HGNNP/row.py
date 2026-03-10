
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
# Credit preprocessing + hyperedge construction
# (reuse the same dataset-specific functions as your Credit COL pipeline)
# =========================================================
from Credit.HGNN.data_preprocessing_Credit_col import (
    preprocess_node_features,
    generate_hyperedge_dict,
)

# =========================================================
# HGNNP model + incidence helpers
# =========================================================
try:
    from Credit.HGNNP.HGNNP import HGNNP_implicit, build_incidence_matrix, compute_degree_vectors
except Exception:
    # fallback (if you run beside HGNNP.py)
    from Credit.HGNNP import HGNNP_implicit, build_incidence_matrix, compute_degree_vectors

# =========================================================
# Rebuild structure after node deletion + train loop (project util)
# (same as your HGNNP row template)
# =========================================================
try:
    from GIF.GIF_HGNNP_ROW_NEI import rebuild_structure_after_node_deletion, train_model
except Exception:
    # fallback (if placed elsewhere in your repo)
    from Credit.HGNNP import rebuild_structure_after_node_deletion, train_model

# =========================================================
# Optional MIA (keep interface; enable with --run_mia)
# =========================================================
try:
    from MIA.MIA_HGNNP import membership_inference
from paths import CREDIT_DATA
    _HAS_MIA = True
except Exception:
    _HAS_MIA = False

# -----------------------------
# Loss (same as your template: FocalLoss)
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
# Utils
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device(device_str: str):
    if torch.cuda.is_available() and ("cuda" in device_str):
        return torch.device(device_str)
    return torch.device("cpu")

def to_sparse_tensor(H_sp, device):
    Hc = H_sp.tocoo()
    idx = torch.LongTensor([Hc.row, Hc.col]).to(device)
    val = torch.FloatTensor(Hc.data).to(device)
    return torch.sparse_coo_tensor(idx, val, Hc.shape).coalesce().to(device)

def build_structure_from_hyperedges(hyperedges: dict, num_nodes: int, device):
    """
    Build (H, dv_inv, de_inv) for a given hyperedge dict.
    """
    H_sp = build_incidence_matrix(hyperedges, num_nodes)
    dv_inv, de_inv = compute_degree_vectors(H_sp)
    H_t = to_sparse_tensor(H_sp, device)
    dv_t = torch.tensor(dv_inv, dtype=torch.float32, device=device)
    de_t = torch.tensor(de_inv, dtype=torch.float32, device=device)
    return H_t, dv_t, de_t

@torch.no_grad()
def eval_acc_masked(model, x, y, mask, H, dv, de) -> float:
    model.eval()
    logits = model(x, H, dv, de)  # HGNNP_implicit returns logits/log-softmax depending on your impl
    pred = logits.argmax(dim=1)
    if mask.sum().item() == 0:
        return float("nan")
    return float((pred[mask] == y[mask]).float().mean().item())

def freeze_all_but_head(model: nn.Module):
    """
    Best-effort head-only finetune for HGNNP_implicit.
    Prefer attribute 'hgc2' (common in HGNN/HGNNP implicit variants).
    """
    for p in model.parameters():
        p.requires_grad = False

    if hasattr(model, "hgc2") and isinstance(getattr(model, "hgc2"), nn.Module):
        for p in model.hgc2.parameters():
            p.requires_grad = True
        return

    # common candidates
    for name in ["classifier", "fc", "lin", "out", "head"]:
        if hasattr(model, name) and isinstance(getattr(model, name), nn.Module):
            for p in getattr(model, name).parameters():
                p.requires_grad = True
            return

    # fallback: unfreeze last child
    children = list(model.children())
    if children:
        for p in children[-1].parameters():
            p.requires_grad = True
        return

    # last fallback: full train
    for p in model.parameters():
        p.requires_grad = True

def finetune_steps(
    model: nn.Module,
    x_after: torch.Tensor,       # deleted node features zeroed (used ONLY for FT forward)
    y: torch.Tensor,
    retain_mask: torch.Tensor,
    H_edit: torch.Tensor,
    dv_edit: torch.Tensor,
    de_edit: torch.Tensor,
    steps: int,
    lr: float,
    wd: float,
    use_focal: bool = True,
):
    """
    Blind retain-only finetune on edited HG.
    """
    if steps <= 0:
        return model

    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    if len(params) == 0:
        raise RuntimeError("No trainable parameters for finetune.")

    opt = optim.Adam(params, lr=lr, weight_decay=wd)
    crit = FocalLoss(gamma=2.0, reduction="mean") if use_focal else nn.CrossEntropyLoss()

    for _ in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        logits = model(x_after, H_edit, dv_edit, de_edit)
        loss = crit(logits[retain_mask], y[retain_mask])
        loss.backward()
        opt.step()

    return model

def maybe_mia(tag, model, X_tr_np, y_tr_np, hyperedges_edit, args, device):
    if (not args.run_mia) or (not _HAS_MIA):
        return None, None
    print(f"— MIA on {tag} —")
    _, (_, _), (auc_target, f1_target) = membership_inference(
        X_train=X_tr_np,
        y_train=y_tr_np,
        hyperedges=hyperedges_edit,
        target_model=model,
        args=args,
        device=device
    )
    # Keep two columns for downstream tables; if you later add deleted-only MIA, map it there.
    return float(auc_target), None

# =========================================================
# Credit single-file load/split
# =========================================================
def load_credit_df(data_csv: str):
    df = pd.read_csv(
        data_csv,
        header=None,
        na_values="?",
        skipinitialspace=True,
    )
    df.columns = [f"A{i}" for i in range(1, 16)] + ["class"]

    df["y"] = df["class"].map({"+": 1, "-": 0})
    if df["y"].isna().any():
        vals = sorted(df["class"].dropna().unique().tolist())
        if len(vals) >= 2:
            mapping = {vals[0]: 0, vals[-1]: 1}
            df["y"] = df["class"].map(mapping)
        else:
            raise ValueError("Cannot infer label mapping from Credit class column.")
    return df

def split_credit_df(df, split_ratio: float, split_seed: int):
    df_train, df_test = train_test_split(
        df,
        test_size=split_ratio,
        random_state=split_seed,
        stratify=df["y"],
    )
    return df_train.reset_index(drop=True), df_test.reset_index(drop=True)

# =========================================================
# One run
# =========================================================
def run_one(args, run_id: int):
    device = get_device(args.device)
    seed = args.seed + run_id
    set_seed(seed)
    print(f"[Device] {device} | seed={seed}")

    # ----- load & split -----
    df_full = load_credit_df(args.data_csv)
    df_tr_raw, df_te_raw = split_credit_df(df_full, args.split_ratio, args.split_seed)

    print(f"训练集样本数: {len(df_tr_raw)}, 测试集样本数: {len(df_te_raw)}")
    print("– TRAIN label dist:", Counter(df_tr_raw["y"]))
    print("– TEST  label dist:", Counter(df_te_raw["y"]))

    # ----- preprocess (same transformer for train/test) -----
    X_tr, y_tr, df_tr_proc, transformer = preprocess_node_features(df_tr_raw, transformer=None)
    X_te, y_te, df_te_proc, _ = preprocess_node_features(df_te_raw, transformer=transformer)

    X_tr = np.asarray(X_tr)
    y_tr = np.asarray(y_tr).astype(np.int64)
    X_te = np.asarray(X_te)
    y_te = np.asarray(y_te).astype(np.int64)

    N = X_tr.shape[0]
    C = int(y_tr.max()) + 1

    # ----- hyperedges (Credit: use all cols A1..A15) -----
    cont_cols = ["A2", "A3", "A8", "A11", "A14", "A15"]
    cat_cols = [f"A{i}" for i in range(1, 16) if f"A{i}" not in cont_cols]
    feature_cols = cat_cols + cont_cols

    hyperedges_tr = generate_hyperedge_dict(
        df_tr_proc,
        feature_cols=feature_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge_train,
        device=device,
    )
    hyperedges_te = generate_hyperedge_dict(
        df_te_proc,
        feature_cols=feature_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge_test,
        device=device,
    )

    # ----- tensors -----
    fts_tr = torch.from_numpy(X_tr).float().to(device)
    lbls_tr = torch.from_numpy(y_tr).long().to(device)
    fts_te = torch.from_numpy(X_te).float().to(device)
    lbls_te = torch.from_numpy(y_te).long().to(device)

    # ----- original train HG structure (no deletion) -----
    H_tr, dv_tr, de_tr, _ = rebuild_structure_after_node_deletion(
        hyperedges_tr, np.array([], dtype=np.int64), N, device
    )

    # ----- test HG structure -----
    H_te, dv_te, de_te = build_structure_from_hyperedges(hyperedges_te, X_te.shape[0], device)

    # ----- train full model -----
    model_full = HGNNP_implicit(
        in_ch=fts_tr.shape[1],
        n_class=C,
        n_hid=args.hidden_dim,
        dropout=args.dropout
    ).to(device)

    optimizer = optim.Adam(model_full.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    criterion = FocalLoss(gamma=2.0, reduction="mean")

    t0 = time.time()
    model_full = train_model(
        model_full, criterion, optimizer, scheduler,
        fts_tr, lbls_tr, H_tr, dv_tr, de_tr,
        num_epochs=args.epochs, print_freq=args.print_freq
    )
    full_time = time.time() - t0

    test_mask_te = torch.ones(X_te.shape[0], dtype=torch.bool, device=device)
    test_acc_full = eval_acc_masked(model_full, fts_te, lbls_te, test_mask_te, H_te, dv_te, de_te)
    print(f"[Full] train_time={full_time:.4f}s | test_acc={test_acc_full:.4f}")

    # ----- sample deleted nodes -----
    n_del = int(N * args.remove_ratio)
    rng = np.random.default_rng(seed)
    deleted_idx = rng.choice(np.arange(N), size=n_del, replace=False).astype(np.int64)
    deleted_idx.sort()

    del_mask = torch.zeros(N, dtype=torch.bool, device=device)
    del_mask[torch.from_numpy(deleted_idx).to(device)] = True
    retain_mask = ~del_mask
    print(f"[Delete] remove_ratio={args.remove_ratio} deleted={n_del} / N={N}")

    # ----- edited HG structure (train) -----
    t_edit = time.time()
    H_edit, dv_edit, de_edit, hyperedges_edit = rebuild_structure_after_node_deletion(
        hyperedges_tr, deleted_idx, N, device
    )
    edit_time = time.time() - t_edit

    # ----- zero-out ONLY for FT input -----
    fts_after = fts_tr.clone()
    fts_after[del_mask] = 0.0

    # ----- evaluate Full@EditedHG (IMPORTANT: use ORIGINAL features fts_tr) -----
    retain_acc_before = eval_acc_masked(model_full, fts_tr, lbls_tr, retain_mask, H_edit, dv_edit, de_edit)
    forget_acc_before = eval_acc_masked(model_full, fts_tr, lbls_tr, del_mask,    H_edit, dv_edit, de_edit)
    print(f"[Full@EditedHG] retain_acc={retain_acc_before:.4f} | forget_acc={forget_acc_before:.4f}")

    rows = []
    mia_o, mia_d = maybe_mia("Full@EditedHG", model_full, X_tr, y_tr, hyperedges_edit, args, device)

    # keep a consistent output style (edit/update/total)
    rows.append({
        "run": run_id,
        "seed": seed,
        "method": "Full@EditedHG",
        "K": 0,
        "edit": float(edit_time),
        "update": 0.0,
        "total": float(edit_time),
        "test_acc": float(test_acc_full),
        "retain_acc": float(retain_acc_before),
        "forget_acc": float(forget_acc_before),
        "mia_overall": None if mia_o is None else float(mia_o),
        "mia_deleted": None if mia_d is None else float(mia_d),
    })

    print(
        f"Full@EditedHG    K={0:4d} | edit={edit_time:.4f} | update={0.0:.4f} | total={edit_time:.4f} | "
        f"test_acc={test_acc_full:.4f} | retain_acc={retain_acc_before:.4f} | forget_acc={forget_acc_before:.4f} | "
        f"mia_overall={'NA' if mia_o is None else f'{mia_o:.4f}'} | mia_deleted={'NA' if mia_d is None else f'{mia_d:.4f}'}"
    )

    # ===== FT-K and FT-head =====
    for K in args.ft_steps:
        # ---- FT-K (all params) ----
        m = copy.deepcopy(model_full)
        for p in m.parameters():
            p.requires_grad = True

        t1 = time.time()
        m = finetune_steps(
            m,
            x_after=fts_after,
            y=lbls_tr,
            retain_mask=retain_mask,
            H_edit=H_edit, dv_edit=dv_edit, de_edit=de_edit,
            steps=K,
            lr=args.ft_lr,
            wd=args.ft_wd,
            use_focal=True
        )
        update_time = time.time() - t1

        test_acc = eval_acc_masked(m, fts_te, lbls_te, test_mask_te, H_te, dv_te, de_te)
        retain_acc = eval_acc_masked(m, fts_tr, lbls_tr, retain_mask, H_edit, dv_edit, de_edit)
        forget_acc = eval_acc_masked(m, fts_tr, lbls_tr, del_mask,    H_edit, dv_edit, de_edit)

        mia_o, mia_d = maybe_mia(f"FT-K(K={K})", m, X_tr, y_tr, hyperedges_edit, args, device)

        total_time = edit_time + update_time
        print(
            f"FT-K@EditedHG    K={K:4d} | edit={edit_time:.4f} | update={update_time:.4f} | total={total_time:.4f} | "
            f"test_acc={test_acc:.4f} | retain_acc={retain_acc:.4f} | forget_acc={forget_acc:.4f} | "
            f"mia_overall={'NA' if mia_o is None else f'{mia_o:.4f}'} | mia_deleted={'NA' if mia_d is None else f'{mia_d:.4f}'}"
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
            "retain_acc": float(retain_acc),
            "forget_acc": float(forget_acc),
            "mia_overall": None if mia_o is None else float(mia_o),
            "mia_deleted": None if mia_d is None else float(mia_d),
        })

        # ---- FT-head (head only) ----
        m = copy.deepcopy(model_full)
        freeze_all_but_head(m)

        t1 = time.time()
        m = finetune_steps(
            m,
            x_after=fts_after,
            y=lbls_tr,
            retain_mask=retain_mask,
            H_edit=H_edit, dv_edit=dv_edit, de_edit=de_edit,
            steps=K,
            lr=args.ft_lr,
            wd=args.ft_wd,
            use_focal=True
        )
        update_time = time.time() - t1

        test_acc = eval_acc_masked(m, fts_te, lbls_te, test_mask_te, H_te, dv_te, de_te)
        retain_acc = eval_acc_masked(m, fts_tr, lbls_tr, retain_mask, H_edit, dv_edit, de_edit)
        forget_acc = eval_acc_masked(m, fts_tr, lbls_tr, del_mask,    H_edit, dv_edit, de_edit)

        mia_o, mia_d = maybe_mia(f"FT-head(K={K})", m, X_tr, y_tr, hyperedges_edit, args, device)

        total_time = edit_time + update_time
        print(
            f"FT-head@EditedHG K={K:4d} | edit={edit_time:.4f} | update={update_time:.4f} | total={total_time:.4f} | "
            f"test_acc={test_acc:.4f} | retain_acc={retain_acc:.4f} | forget_acc={forget_acc:.4f} | "
            f"mia_overall={'NA' if mia_o is None else f'{mia_o:.4f}'} | mia_deleted={'NA' if mia_d is None else f'{mia_d:.4f}'}"
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
            "retain_acc": float(retain_acc),
            "forget_acc": float(forget_acc),
            "mia_overall": None if mia_o is None else float(mia_o),
            "mia_deleted": None if mia_d is None else float(mia_d),
        })

    return rows

# =========================================================
# Summary + CSV
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

    agg_cols = ["edit", "update", "total", "test_acc", "retain_acc", "forget_acc", "mia_overall", "mia_deleted"]
    summary = df.groupby(["method", "K"], dropna=False)[agg_cols].agg(["mean", "std"]).reset_index()

    print("\n== Summary (mean±std) ==")
    for _, r in summary.iterrows():
        method = r[("method", "")]
        K = int(r[("K", "")])

        edit_s = format_mean_std(r[("edit", "mean")], r[("edit", "std")])
        upd_s = format_mean_std(r[("update", "mean")], r[("update", "std")])
        total_s = format_mean_std(r[("total", "mean")], r[("total", "std")])
        test_s = format_mean_std(r[("test_acc", "mean")], r[("test_acc", "std")])
        ret_s = format_mean_std(r[("retain_acc", "mean")], r[("retain_acc", "std")])
        fog_s = format_mean_std(r[("forget_acc", "mean")], r[("forget_acc", "std")])
        miao_s = format_mean_std(r[("mia_overall", "mean")], r[("mia_overall", "std")])
        miad_s = format_mean_std(r[("mia_deleted", "mean")], r[("mia_deleted", "std")])

        print(
            f"{method:14s} K={K:4d} | edit={edit_s} | update={upd_s} | total={total_s} | "
            f"test_acc={test_s} | retain_acc={ret_s} | forget_acc={fog_s} | "
            f"mia_overall={miao_s} | mia_deleted={miad_s}"
        )

    print(f"\n[Saved] {out_csv}")

# =========================================================
# Args
# =========================================================
def get_args():
    p = argparse.ArgumentParser("Credit HGNNP FT baselines on edited hypergraph (row deletion, feature-zero)")

    # Data
    p.add_argument("--data_csv", type=str,
                   default=CREDIT_DATA,
                   help="Credit Approval crx.data path")
    p.add_argument("--split_ratio", type=float, default=0.2)
    p.add_argument("--split_seed", type=int, default=42)

    # Hypergraph
    p.add_argument("--max_nodes_per_hyperedge_train", type=int, default=50)
    p.add_argument("--max_nodes_per_hyperedge_test", type=int, default=50)

    # Model
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)

    # Full training
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--milestones", type=int, nargs="*", default=[100, 150])
    p.add_argument("--gamma", type=float, default=0.1)
    p.add_argument("--print_freq", type=int, default=10)

    # Deletion
    p.add_argument("--remove_ratio", type=float, default=0.30)

    # FT
    p.add_argument("--ft_steps", type=int, nargs="+", default=[50, 100, 200])
    p.add_argument("--ft_lr", type=float, default=5e-3)
    p.add_argument("--ft_wd", type=float, default=0.0)

    # Runs / misc
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--run_mia", action="store_true", help="是否运行MIA（默认不跑）")
    p.add_argument("--out_csv", type=str, default="ft_hgnnp_row_credit_zero_results.csv")

    return p.parse_args()

def main():
    args = get_args()

    if not os.path.exists(args.data_csv):
        print(f"[WARN] data_csv not found: {args.data_csv}")

    all_rows = []
    for run_id in range(args.runs):
        print(f"\n================= RUN {run_id+1}/{args.runs} =================")
        all_rows.extend(run_one(args, run_id))

    summarize_and_save(all_rows, args.out_csv)

if __name__ == "__main__":
    main()
