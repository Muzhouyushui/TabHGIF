# =========================
# File: run_ft_hgnn_row_zero.py
# =========================
import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from database.data_preprocessing.data_preprocessing_K import (
    preprocess_node_features,
    generate_hyperedge_dict,
)

from HGNN.HGNN_2 import HGNN_implicit
from GIF.GIF_HGNN_ROW_NEI import rebuild_structure_after_node_deletion, train_model

# (可选) 你有 MIA 的话就打开
try:
    from MIA.MIA_utils import membership_inference
    _HAS_MIA = True
except Exception:
    _HAS_MIA = False

# -----------------------------
# Utilities (match HGCN-FT style)
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
    # HGNN_implicit: hgc1, hgc2 (treat hgc2 as head)
    for p in model.parameters():
        p.requires_grad = False
    if hasattr(model, "hgc2"):
        for p in model.hgc2.parameters():
            p.requires_grad = True
    else:
        # fallback: unfreeze last child
        children = list(model.children())
        if children:
            for p in children[-1].parameters():
                p.requires_grad = True

def finetune_steps_hgnn(
    model: nn.Module,
    x_after: torch.Tensor,          # zero-out features used ONLY for FT input
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
    retain-only finetune for K steps (blind):
      loss = CE on retained nodes only.
    """
    model.train()
    opt = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay
    )
    loss_fn = nn.NLLLoss()

    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        out = model(x_after, H_edit, dv_edit, de_edit)   # log_softmax
        loss = loss_fn(out[retain_mask], y[retain_mask])
        loss.backward()
        opt.step()
    return model

@torch.no_grad()
def eval_acc_masked(model, x, y, mask, H, dv_inv, de_inv):
    model.eval()
    out = model(x, H, dv_inv, de_inv)
    pred = out.argmax(dim=1)
    return float((pred[mask] == y[mask]).float().mean().item())

def _clone_model(model: nn.Module):
    return copy.deepcopy(model)

def _maybe_mia(tag, model, X_tr, y_tr_np, hyperedges, args, device):
    if (not args.run_mia) or (not _HAS_MIA):
        return None, None

    # overall MIA / deleted-only MIA 的口径你可以按你项目的定义再对齐
    # 这里给一个最简：直接跑 target AUC
    print(f"— MIA on {tag} —")
    _, (_, _), (auc_target, f1_target) = membership_inference(
        X_train=X_tr,
        y_train=y_tr_np,
        hyperedges=hyperedges,
        target_model=model,
        args=args,
        device=device
    )
    return float(auc_target), float(f1_target)

# -----------------------------
# One run (match run_ft_row_zero.py structure)
# -----------------------------
def run_one(args, run_id: int):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    seed = args.seed + run_id
    set_seed(seed)

    # ===== Load train/test (same as your HGIF/HGNN pipeline) =====
    X_tr, y_tr_np, df_tr, transformer = preprocess_node_features(
        args.train_csv, is_test=False, transformer=None
    )
    X_te, y_te_np, df_te, _ = preprocess_node_features(
        args.test_csv, is_test=True, transformer=transformer
    )

    X_tr = np.asarray(X_tr)
    y_tr_np = np.asarray(y_tr_np)
    X_te = np.asarray(X_te)
    y_te_np = np.asarray(y_te_np)

    # ===== Hyperedges from raw df (IMPORTANT: same as your HGNN code) =====
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

    # ===== Torch tensors =====
    fts_tr = torch.from_numpy(X_tr).float().to(device)
    lbls_tr = torch.from_numpy(y_tr_np).long().to(device)

    fts_te = torch.from_numpy(X_te).float().to(device)
    lbls_te = torch.from_numpy(y_te_np).long().to(device)

    N = X_tr.shape[0]
    C = int(y_tr_np.max()) + 1

    # ===== Train full model on ORIGINAL training hypergraph =====
    model_full = HGNN_implicit(
        in_ch=fts_tr.shape[1],
        n_class=C,
        n_hid=args.hidden_dim,
        dropout=args.dropout
    ).to(device)

    optimizer = optim.Adam(model_full.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    criterion = nn.NLLLoss()

    print("== Train Full Model ==")
    t0 = time.time()
    model_full = train_model(
        model_full, criterion, optimizer, scheduler,
        fts_tr, lbls_tr,
        *rebuild_structure_after_node_deletion(hyperedges_tr, np.array([], dtype=np.int64), N, device)[:3],  # H_tr,dv_tr,de_tr
        num_epochs=args.epochs, print_freq=args.print_freq
    )
    full_train_time = time.time() - t0

    # test on test hypergraph
    H_te, dv_te, de_te, _ = rebuild_structure_after_node_deletion(hyperedges_te, np.array([], dtype=np.int64), X_te.shape[0], device)
    acc_test = eval_acc_masked(model_full, fts_te, lbls_te, torch.ones(X_te.shape[0], dtype=torch.bool, device=device),
                               H_te, dv_te, de_te)
    print(f"[Full] Test ACC={acc_test:.4f} | train_time={full_train_time:.2f}s")

    # ===== Sample deleted rows (same as HGCN FT remove_ratio) =====
    n_del = int(N * args.remove_ratio)
    rng = np.random.default_rng(seed)
    deleted_idx = rng.choice(np.arange(N), size=n_del, replace=False)
    deleted = torch.tensor(deleted_idx, dtype=torch.long, device=device)
    print(f"[Delete] remove_ratio={args.remove_ratio}, deleted={len(deleted)}")

    retain_mask, del_mask = build_masks(N, deleted, device)

    # ===== Build EDITED hypergraph structure =====
    # 你已有这个工具函数：会删除节点并返回 (H_edit, dv_edit, de_edit, hyperedges_edit)
    H_edit, dv_edit, de_edit, hyperedges_edit = rebuild_structure_after_node_deletion(
        hyperedges_tr, deleted_idx.astype(np.int64), N, device
    )
    print(f"[EditedHG] #hyperedges(orig)={len(hyperedges_tr)} -> #hyperedges(edit)={len(hyperedges_edit)}")

    # ===== Feature-zero ONLY for finetune input (match HGCN FT behavior) =====
    fts_tr_after = fts_tr.clone()
    fts_tr_after[deleted] = 0.0

    # ===== Evaluate retain/forget BEFORE FT (IMPORTANT: use ORIGINAL fts_tr like HGCN FT) =====
    acc_ret = eval_acc_masked(model_full, fts_tr, lbls_tr, retain_mask, H_edit, dv_edit, de_edit)
    acc_for = eval_acc_masked(model_full, fts_tr, lbls_tr, del_mask,    H_edit, dv_edit, de_edit)
    print(f"[Full@EditedHG] Retain ACC={acc_ret:.4f} | Forget ACC={acc_for:.4f}")

    results = []

    # ===== Record Full@EditedHG (for summary table) =====
    mia_o, mia_d = _maybe_mia("Full@EditedHG", model_full, X_tr, y_tr_np, hyperedges_edit, args, device)
    results.append({
        "run": run_id,
        "method": "Full@EditedHG",
        "K": 0,
        "time_sec": full_train_time,
        "test_acc": acc_test,
        "retain_acc": acc_ret,
        "forget_acc": acc_for,
        "mia_overall": mia_o,
        "mia_deleted": mia_d,
        "seed": seed
    })

    # ===== FT-K =====
    print("\n== FT-K (warm-start) ==")
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
        ft_time = time.time() - t1

        acc_test_k = eval_acc_masked(m, fts_te, lbls_te, torch.ones(X_te.shape[0], dtype=torch.bool, device=device),
                                     H_te, dv_te, de_te)
        acc_ret_k = eval_acc_masked(m, fts_tr, lbls_tr, retain_mask, H_edit, dv_edit, de_edit)
        acc_for_k = eval_acc_masked(m, fts_tr, lbls_tr, del_mask,    H_edit, dv_edit, de_edit)

        print(f"[FT-K] K={K:4d} | Test ACC={acc_test_k:.4f} | Retain ACC={acc_ret_k:.4f} | Forget ACC={acc_for_k:.4f} | time={ft_time:.2f}s")

        mia_o, mia_d = _maybe_mia(f"FT-K(K={K})", m, X_tr, y_tr_np, hyperedges_edit, args, device)
        results.append({
            "run": run_id,
            "method": "FT-K@EditedHG",
            "K": K,
            "time_sec": ft_time,
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
        ft_time = time.time() - t1

        acc_test_k = eval_acc_masked(m, fts_te, lbls_te, torch.ones(X_te.shape[0], dtype=torch.bool, device=device),
                                     H_te, dv_te, de_te)
        acc_ret_k = eval_acc_masked(m, fts_tr, lbls_tr, retain_mask, H_edit, dv_edit, de_edit)
        acc_for_k = eval_acc_masked(m, fts_tr, lbls_tr, del_mask,    H_edit, dv_edit, de_edit)

        print(f"[FT-head] K={K:4d} | Test ACC={acc_test_k:.4f} | Retain ACC={acc_ret_k:.4f} | Forget ACC={acc_for_k:.4f} | time={ft_time:.2f}s")

        mia_o, mia_d = _maybe_mia(f"FT-head(K={K})", m, X_tr, y_tr_np, hyperedges_edit, args, device)
        results.append({
            "run": run_id,
            "method": "FT-head@EditedHG",
            "K": K,
            "time_sec": ft_time,
            "test_acc": acc_test_k,
            "retain_acc": acc_ret_k,
            "forget_acc": acc_for_k,
            "mia_overall": mia_o,
            "mia_deleted": mia_d,
            "seed": seed
        })

    print("\nDone.\n")
    return results

def summarize_and_save(all_rows, out_csv):
    import pandas as pd
from paths import ACI_TEST, ACI_TRAIN

    df = pd.DataFrame(all_rows)
    df.to_csv(out_csv, index=False)

    agg_cols = ["time_sec", "test_acc", "retain_acc", "forget_acc", "mia_overall", "mia_deleted"]
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

        print(f"{method:14s} K={int(K):4d} | time={fmt('time_sec')} | test_acc={fmt('test_acc')} "
              f"| retain_acc={fmt('retain_acc')} | forget_acc={fmt('forget_acc')} "
              f"| mia_overall={fmt('mia_overall')} | mia_deleted={fmt('mia_deleted')}")
    print(f"\n[Saved] {out_csv}")

def get_args():
    p = argparse.ArgumentParser("HGNN FT baseline (HGCN-style) on edited hypergraph")
    p.add_argument("--train_csv", type=str,
                        default=ACI_TRAIN,
                        help="训练数据路径（adult.data）")
    p.add_argument("--test_csv", type=str,
                        default=ACI_TEST,
                        help="测试数据路径（adult.test）")

    # hypergraph build
    p.add_argument("--cat_cols", type=str, nargs="+", default=[
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ])
    p.add_argument("--max_nodes_per_hyperedge_train", type=int, default=10000)
    p.add_argument("--max_nodes_per_hyperedge_test",  type=int, default=10000)

    # model/train
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--milestones", type=int, nargs="*", default=[100, 150])
    p.add_argument("--gamma", type=float, default=0.1)
    p.add_argument("--print_freq", type=int, default=10)

    # deletion + runs
    p.add_argument("--remove_ratio", type=float, default=0.30)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--seed", type=int, default=1)

    # finetune
    p.add_argument("--ft_steps", type=int, nargs="+", default=[50, 100, 200])
    p.add_argument("--ft_lr", type=float, default=5e-3)
    p.add_argument("--ft_wd", type=float, default=0.0)

    # misc
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--run_mia", action="store_true",default=True)
    p.add_argument("--out_csv", type=str, default="ft_hgnn_row_zero_results.csv")
    return p.parse_args()

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
