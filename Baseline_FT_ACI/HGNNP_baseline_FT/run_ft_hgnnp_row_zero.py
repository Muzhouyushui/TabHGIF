# =========================
# File: run_ft_hgnnp_row_zero.py
# =========================
"""
HGNNP FT baselines on edited hypergraph (row deletion, feature-zero), in the SAME style as HGNN FT script.

Methods:
  - Full@EditedHG    : train full model on original training HG, evaluate on edited HG (retain/forget)
  - FT-K@EditedHG    : warm-start from Full, finetune ALL params for K steps on edited HG (retain-only loss)
  - FT-head@EditedHG : warm-start from Full, finetune HEAD only for K steps on edited HG (retain-only loss)

Blind setting:
  - Finetune uses x_after (deleted node features are zeroed)
  - Loss computed ONLY on retained nodes (retain_mask)
  - No deleted labels/features used as training signal

Evaluation semantics (IMPORTANT, matches your HGCN/HGNN FT):
  - retain_acc / forget_acc are evaluated using ORIGINAL features x_before (not zeroed),
    but on EDITED hypergraph structure (H_edit).
"""

import os
import time
import copy
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from database.data_preprocessing.data_preprocessing_K import (
    preprocess_node_features,
    generate_hyperedge_dict,
)

# HGNNP model + incidence helpers (same as your HGNNP_Unlearning_row_nei.py)
from HGNNs_Model.HGNNP import HGNNP_implicit, build_incidence_matrix, compute_degree_vectors

# GIF util (must provide rebuild_structure_after_node_deletion + train_model)
from GIF.GIF_HGNNP_ROW_NEI import (
    rebuild_structure_after_node_deletion,
    train_model,
)

# Optional MIA (keep interface; you can turn on with --run_mia)
try:
    from MIA.MIA_utils import membership_inference
from paths import ACI_TEST, ACI_TRAIN
    _HAS_MIA = True
except Exception:
    _HAS_MIA = False

# -----------------------------
# Loss (follow your HGNNP script: FocalLoss)
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
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def to_sparse_tensor(H_sp, device):
    Hc = H_sp.tocoo()
    idx = torch.LongTensor([Hc.row, Hc.col]).to(device)
    val = torch.FloatTensor(Hc.data).to(device)
    return torch.sparse_coo_tensor(idx, val, Hc.shape).coalesce().to(device)

def build_structure_from_hyperedges(hyperedges: dict, num_nodes: int, device):
    """
    For test HG construction (not using rebuild_structure_after_node_deletion).
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
    logits = model(x, H, dv, de)  # HGNNP returns logits
    pred = logits.argmax(dim=1)
    return float((pred[mask] == y[mask]).float().mean().item())

def freeze_all_but_head(model: nn.Module):
    """
    Make a best-effort "head-only" finetune for HGNNP_implicit.
    We try common attribute names; if not found, unfreeze the last child module.
    """
    for p in model.parameters():
        p.requires_grad = False

    # common candidates
    cand_names = ["hgc2", "classifier", "fc", "lin", "out", "head"]
    for name in cand_names:
        if hasattr(model, name):
            m = getattr(model, name)
            if isinstance(m, nn.Module):
                for p in m.parameters():
                    p.requires_grad = True
                return

    # fallback: unfreeze last child module
    children = list(model.children())
    if children:
        for p in children[-1].parameters():
            p.requires_grad = True

def finetune_steps(
    model: nn.Module,
    x_after: torch.Tensor,       # deleted features zeroed (used ONLY for FT forward)
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
    opt = optim.Adam(params, lr=lr, weight_decay=wd)

    if use_focal:
        crit = FocalLoss(gamma=2.0, reduction="mean")
    else:
        crit = nn.CrossEntropyLoss()

    for _ in range(steps):
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
    return float(auc_target), float(auc_target)  # keep both columns; adapt if you have deleted-only MIA later

# -----------------------------
# One run
# -----------------------------
def run_one(args, run_id: int):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    seed = args.seed + run_id
    set_seed(seed)
    print(f"[Device] {device} | seed={seed}")

    # ----- data (same pipeline as your HGNNP script) -----
    X_tr, y_tr, df_tr, transformer = preprocess_node_features(args.train_csv, is_test=False, transformer=None)
    X_te, y_te, df_te, _ = preprocess_node_features(args.test_csv, is_test=True, transformer=transformer)

    X_tr = np.asarray(X_tr)
    y_tr = np.asarray(y_tr)
    X_te = np.asarray(X_te)
    y_te = np.asarray(y_te)

    N = X_tr.shape[0]
    C = int(y_tr.max()) + 1

    # ----- build hyperedges from df (IMPORTANT) -----
    hyperedges_tr = generate_hyperedge_dict(
        df_tr,
        args.cat_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge_train,
        device=device,
    )
    hyperedges_te = generate_hyperedge_dict(
        df_te,
        args.cat_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge_test,
        device=device,
    )

    # ----- tensors -----
    fts_tr = torch.from_numpy(X_tr).float().to(device)
    lbls_tr = torch.from_numpy(y_tr).long().to(device)

    fts_te = torch.from_numpy(X_te).float().to(device)
    lbls_te = torch.from_numpy(y_te).long().to(device)

    # ----- original train HG structure (no deletion) -----
    # use your rebuild helper for consistency
    H_tr, dv_tr, de_tr, _ = rebuild_structure_after_node_deletion(
        hyperedges_tr, np.array([], dtype=np.int64), N, device
    )

    # ----- test HG structure -----
    # (if your rebuild supports it, you can also use rebuild; here use explicit build)
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

    # ----- edited HG structure -----
    H_edit, dv_edit, de_edit, hyperedges_edit = rebuild_structure_after_node_deletion(
        hyperedges_tr, deleted_idx, N, device
    )

    # ----- zero-out ONLY for FT input -----
    fts_after = fts_tr.clone()
    fts_after[del_mask] = 0.0

    # ----- evaluate Full@EditedHG (IMPORTANT: use ORIGINAL features fts_tr) -----
    retain_acc_before = eval_acc_masked(model_full, fts_tr, lbls_tr, retain_mask, H_edit, dv_edit, de_edit)
    forget_acc_before = eval_acc_masked(model_full, fts_tr, lbls_tr, del_mask,    H_edit, dv_edit, de_edit)
    print(f"[Full@EditedHG] retain_acc={retain_acc_before:.4f} forget_acc={forget_acc_before:.4f}")

    rows = []
    mia_o, mia_d = maybe_mia("Full@EditedHG", model_full, X_tr, y_tr, hyperedges_edit, args, device)
    rows.append({
        "run": run_id,
        "seed": seed,
        "method": "Full@EditedHG",
        "K": 0,
        "time_sec": full_time,
        "test_acc": test_acc_full,
        "retain_acc": retain_acc_before,
        "forget_acc": forget_acc_before,
        "mia_overall": mia_o,
        "mia_deleted": mia_d
    })

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
        ft_time = time.time() - t1

        test_acc = eval_acc_masked(m, fts_te, lbls_te, test_mask_te, H_te, dv_te, de_te)
        retain_acc = eval_acc_masked(m, fts_tr, lbls_tr, retain_mask, H_edit, dv_edit, de_edit)
        forget_acc = eval_acc_masked(m, fts_tr, lbls_tr, del_mask,    H_edit, dv_edit, de_edit)

        mia_o, mia_d = maybe_mia(f"FT-K(K={K})", m, X_tr, y_tr, hyperedges_edit, args, device)
        rows.append({
            "run": run_id,
            "seed": seed,
            "method": "FT-K@EditedHG",
            "K": K,
            "time_sec": ft_time,
            "test_acc": test_acc,
            "retain_acc": retain_acc,
            "forget_acc": forget_acc,
            "mia_overall": mia_o,
            "mia_deleted": mia_d
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
        ft_time = time.time() - t1

        test_acc = eval_acc_masked(m, fts_te, lbls_te, test_mask_te, H_te, dv_te, de_te)
        retain_acc = eval_acc_masked(m, fts_tr, lbls_tr, retain_mask, H_edit, dv_edit, de_edit)
        forget_acc = eval_acc_masked(m, fts_tr, lbls_tr, del_mask,    H_edit, dv_edit, de_edit)

        mia_o, mia_d = maybe_mia(f"FT-head(K={K})", m, X_tr, y_tr, hyperedges_edit, args, device)
        rows.append({
            "run": run_id,
            "seed": seed,
            "method": "FT-head@EditedHG",
            "K": K,
            "time_sec": ft_time,
            "test_acc": test_acc,
            "retain_acc": retain_acc,
            "forget_acc": forget_acc,
            "mia_overall": mia_o,
            "mia_deleted": mia_d
        })

    return rows

# -----------------------------
# Summary + CSV
# -----------------------------
def summarize_and_save(all_rows, out_csv):
    df = pd.DataFrame(all_rows)
    df.to_csv(out_csv, index=False)

    agg_cols = ["time_sec", "test_acc", "retain_acc", "forget_acc", "mia_overall", "mia_deleted"]
    g = df.groupby(["method", "K"], dropna=False)[agg_cols].agg(["mean", "std"]).reset_index()

    print("== Summary (mean±std) ==")
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
            f"{method:14s} K={K:4d} | time={fmt('time_sec')} | test_acc={fmt('test_acc')} "
            f"| retain_acc={fmt('retain_acc')} | forget_acc={fmt('forget_acc')} "
            f"| mia_overall={fmt('mia_overall')} | mia_deleted={fmt('mia_deleted')}"
        )

    print(f"\n[Saved] {out_csv}")

# -----------------------------
# Args
# -----------------------------
def build_parser():
    p = argparse.ArgumentParser("HGNNP FT baseline (HGCN/HGNN-style)")

    # data
    p.add_argument("--train_csv", type=str,
                        default=ACI_TRAIN,
                        help="训练数据路径（adult.data）")
    p.add_argument("--test_csv", type=str,
                        default=ACI_TEST,
                        help="测试数据路径（adult.test）")

    # hyperedges
    p.add_argument("--cat_cols", type=str, nargs="+", default=[
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ])
    p.add_argument("--max_nodes_per_hyperedge_train", type=int, default=10000)
    p.add_argument("--max_nodes_per_hyperedge_test",  type=int, default=10000)

    # model
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.1)

    # full training
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--milestones", type=int, nargs="+", default=[100, 150])
    p.add_argument("--gamma", type=float, default=0.1)
    p.add_argument("--print_freq", type=int, default=10)

    # deletion + runs
    p.add_argument("--remove_ratio", type=float, default=0.30)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)

    # finetune
    p.add_argument("--ft_steps", type=int, nargs="+", default=[50, 100, 200])
    p.add_argument("--ft_lr", type=float, default=5e-3)
    p.add_argument("--ft_wd", type=float, default=0.0)

    # misc
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--run_mia", action="store_true", default=True)
    p.add_argument("--out_csv", type=str, default="ft_hgnnp_row_zero_results.csv")

    return p

def main():
    args = build_parser().parse_args()

    if not os.path.exists(args.train_csv):
        print(f"[WARN] train_csv not found: {args.train_csv}")
    if not os.path.exists(args.test_csv):
        print(f"[WARN] test_csv not found: {args.test_csv}")

    all_rows = []
    for run_id in range(args.runs):
        print(f"\n================= RUN {run_id+1}/{args.runs} =================")
        all_rows.extend(run_one(args, run_id))

    summarize_and_save(all_rows, args.out_csv)

if __name__ == "__main__":
    main()
