# run_ft_hgcn_bank_row_zero.py
# -*- coding: utf-8 -*-

import os
import time
import copy
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split

# ===== Bank 数据预处理 & 超图构建（按你现有 Bank HGCN 管线）=====
from bank.HGNNP.data_preprocessing_bank import (
    preprocess_node_features_bank,
    generate_hyperedge_dict_bank,
)

# ===== HGCN 模型 & Laplacian =====
from bank.HGCN.HGCN import HyperGCN, laplacian

# ===== 训练 / 评估 / MIA（按你现有 Bank HGCN 管线）=====
from bank.HGCN.GIF_HGCN_ROW_bank import train_model
from bank.HGCN.HGCN_utils import evaluate_hgcn_acc, evaluate_hgcn_f1
from bank.HGCN.MIA_HGCN import membership_inference_hgcn


# ============================================================
# Utils
# ============================================================

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _device(args):
    if args.cuda and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _clone_model(model):
    return copy.deepcopy(model)


def _build_masks(num_nodes: int, deleted: torch.LongTensor, device):
    retain_mask = torch.ones(num_nodes, dtype=torch.bool, device=device)
    retain_mask[deleted] = False
    del_mask = ~retain_mask
    return retain_mask, del_mask


def _member_mask_from_retain(retain_mask: torch.Tensor):
    # retained=1, deleted=0
    return retain_mask.detach().cpu().numpy().astype(np.int64)


def _member_mask_from_deleted(deleted: torch.LongTensor, N: int):
    # deleted=1, others=0
    mm = np.zeros(N, dtype=np.int64)
    mm[deleted.detach().cpu().numpy()] = 1
    return mm


def _remove_nodes_from_hyperedges(hyperedges, deleted_nodes):
    """

    """
    del_set = set(int(x) for x in deleted_nodes)
    new_edges = []
    for e in hyperedges:
        e2 = [v for v in e if int(v) not in del_set]
        if len(e2) >= 2:
            new_edges.append(e2)
    return new_edges


def _new_model_hgcn_bank(X_np, hyperedges, args, device):
    """

    """
    cfg = lambda: None
    cfg.d = X_np.shape[1]
    cfg.c = 2  # bank 二分类
    cfg.depth = args.depth
    cfg.hidden = args.hidden_dim
    cfg.dropout = args.dropout
    cfg.fast = args.fast
    cfg.mediators = args.mediators
    cfg.cuda = args.cuda
    cfg.dataset = "bank"

    model = HyperGCN(
        X_np.shape[0],
        hyperedges,
        X_np,
        cfg
    ).to(device)

    A = laplacian(hyperedges, X_np, args.mediators).to(device)
    model.structure = A
    for layer in model.layers:
        layer.reapproximate = False
    return model, A


def eval_split_hgcn(model, x_tensor, y_tensor, A_tensor, mask=None):
    """

    """
    model.structure = A_tensor
    for l in model.layers:
        l.reapproximate = False

    data = {"lap": x_tensor, "y": y_tensor}
    if mask is not None:
        data["train_mask"] = mask

    f1 = evaluate_hgcn_f1(model, data)
    acc = evaluate_hgcn_acc(model, data)
    return f1, acc


def freeze_all_but_head_hgcn(model):
    """

    """
    for p in model.parameters():
        p.requires_grad = False

    if hasattr(model, "layers") and len(model.layers) > 0:
        for p in model.layers[-1].parameters():
            p.requires_grad = True
        return

    # fallback（保守）
    for n, p in model.named_parameters():
        if any(k in n.lower() for k in ["out", "classifier", "final", "last"]):
            p.requires_grad = True


def finetune_steps_hgcn(
    model,
    x_after: torch.Tensor,
    y: torch.Tensor,
    retain_mask: torch.Tensor,
    steps: int,
    lr: float,
    weight_decay: float = 0.0,
    print_freq: int = 0,
):
    """
    """
    trainable = [p for p in model.parameters() if p.requires_grad]
    if len(trainable) == 0:
        raise RuntimeError("No trainable parameters found in finetune_steps_hgcn().")

    opt = optim.Adam(trainable, lr=lr, weight_decay=weight_decay)
    criterion = nn.NLLLoss()

    model.train()
    for step in range(1, steps + 1):
        opt.zero_grad()

        out = model(x_after)
        loss = criterion(out[retain_mask], y[retain_mask])

        loss.backward()
        opt.step()

        if print_freq > 0 and (step == 1 or step % print_freq == 0 or step == steps):
            with torch.no_grad():
                pred = out.argmax(dim=1)
                acc = (pred[retain_mask] == y[retain_mask]).float().mean().item()
            print(f"[FT] step {step:4d}/{steps} | loss={loss.item():.4f} | retain_acc={acc:.4f}")

    return model


# ============================================================
# Core
# ============================================================

def get_args():
    p = argparse.ArgumentParser("HGCN FT baselines on Bank (row deletion, feature-zero)")

    # ===== Data =====
    p.add_argument("--data_csv", type=str,
                   default="/root/autodl-tmp/TabHGIF/data_banking/bank/bank-full.csv")
    p.add_argument("--split_ratio", type=float, default=0.2)

    p.add_argument("--cat_cols", type=str, nargs="+", default=[
        'job', 'marital', 'education', 'default', 'housing',
        'loan', 'contact', 'month', 'poutcome'
    ])
    p.add_argument("--cont_cols", type=str, nargs="+", default=[
        'age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous'
    ])

    # ===== Hypergraph =====
    p.add_argument("--max_nodes_per_hyperedge", type=int, default=50)
    p.add_argument("--mediators", action="store_true")

    # ===== Model / Full Train =====
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.01)
    p.add_argument("--fast", action="store_true")
    p.add_argument("--cuda", action="store_true", default=True)

    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight_decay", type=float, default=0.001)
    p.add_argument("--milestones", type=int, nargs="+", default=[100, 150])
    p.add_argument("--gamma", type=float, default=0.1)
    p.add_argument("--log_every", type=int, default=10)

    # ===== Deletion =====
    p.add_argument("--remove_ratio", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=1)

    # ===== FT baselines =====
    p.add_argument("--ft_steps", type=int, nargs="+", default=[50, 100, 200])
    p.add_argument("--ft_lr", type=float, default=1e-3)
    p.add_argument("--ft_wd", type=float, default=0.0)
    p.add_argument("--ft_print_freq", type=int, default=0)

    # ===== MIA =====
    p.add_argument("--run_mia", action="store_true", default=True)

    # ===== Multi-run + save =====
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--out_csv", type=str, default="ft_hgcn_bank_row_zero_results.csv")

    return p.parse_args()


def _maybe_mia_bank(tag, model, A_used, hedges_used, X_train_np, y_train_np, args, device,
                    retain_mask=None, deleted=None):
    mia_overall_auc = None
    mia_deleted_auc = None

    if not args.run_mia:
        return mia_overall_auc, mia_deleted_auc

    model.structure = A_used
    for l in model.layers:
        l.reapproximate = False

    N = len(y_train_np)

    # overall: retained=1, deleted=0
    if retain_mask is not None:
        mm_retain = _member_mask_from_retain(retain_mask)
        print(f"— MIA on {tag} (overall/retain-membership) —")
        _, _, tgt = membership_inference_hgcn(
            X_train=X_train_np,
            y_train=y_train_np,
            hyperedges=hedges_used,
            target_model=model,
            args=args,
            device=device,
            member_mask=mm_retain
        )
        if tgt is not None:
            mia_overall_auc = float(tgt[0])

    # deleted-only: deleted=1, others=0
    if deleted is not None:
        mm_deleted = _member_mask_from_deleted(deleted, N)
        print(f"— MIA on {tag} (deleted-only membership) —")
        _, _, tgt2 = membership_inference_hgcn(
            X_train=X_train_np,
            y_train=y_train_np,
            hyperedges=hedges_used,
            target_model=model,
            args=args,
            device=device,
            member_mask=mm_deleted
        )
        if tgt2 is not None:
            mia_deleted_auc = float(tgt2[0])

    return mia_overall_auc, mia_deleted_auc


def run_one(args, run_id: int):
    device = _device(args)
    seed = args.seed + run_id
    set_seed(seed)
    print(f"[Device] {device} | seed={seed}")

    df_full = pd.read_csv(args.data_csv, sep=';', skipinitialspace=True)
    df_tr, df_te = train_test_split(
        df_full,
        test_size=args.split_ratio,
        stratify=df_full["y"],
        random_state=42 + run_id
    )
    df_tr = df_tr.reset_index(drop=True)
    df_te = df_te.reset_index(drop=True)
    print(f"Train={len(df_tr)}, Test={len(df_te)}")

    # ===== 2) 预处理 =====
    X_tr, y_tr_np, df_tr_proc, transformer = preprocess_node_features_bank(df_tr, is_test=False)
    X_te, y_te_np, df_te_proc, _ = preprocess_node_features_bank(df_te, is_test=True, transformer=transformer)

    # ===== 3) 原始超图 =====
    hedges_tr = list(generate_hyperedge_dict_bank(
        df_tr_proc, args.cat_cols, args.cont_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        device=device
    ).values())
    hedges_te = list(generate_hyperedge_dict_bank(
        df_te_proc, args.cat_cols, args.cont_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        device=device
    ).values())

    # ===== 4)  =====
    fts_tr = torch.from_numpy(X_tr).float().to(device)
    lbls_tr = torch.tensor(y_tr_np, dtype=torch.long, device=device)
    fts_te = torch.from_numpy(X_te).float().to(device)
    lbls_te = torch.tensor(y_te_np, dtype=torch.long, device=device)

    model_full, A_tr = _new_model_hgcn_bank(X_tr, hedges_tr, args, device)
    A_te = laplacian(hedges_te, X_te, args.mediators).to(device)

    # ===== 6) Full train =====
    print("== Train Full Model ==")
    t0 = time.time()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model_full.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

    model_full = train_model(
        model_full, criterion, optimizer, scheduler,
        fts_tr, lbls_tr,
        num_epochs=args.epochs,
        print_freq=args.log_every
    )
    full_train_time = time.time() - t0

    f1_test_full, acc_test_full = eval_split_hgcn(model_full, fts_te, lbls_te, A_te, mask=None)
    print(f"[Full] Test ACC={acc_test_full:.4f} | F1={f1_test_full:.4f} | train_time={full_train_time:.4f}s")

    N = X_tr.shape[0]
    n_del = int(N * args.remove_ratio)
    rng = np.random.default_rng(seed)
    deleted_idx = rng.choice(np.arange(N), size=n_del, replace=False)
    deleted = torch.tensor(deleted_idx, dtype=torch.long, device=device)
    print(f"[Delete] remove_ratio={args.remove_ratio}, deleted={len(deleted)}")

    retain_mask, del_mask = _build_masks(N, deleted, device)

    # ===== 8) Feature-zero (row deletion approximation) =====
    fts_tr_after = fts_tr.clone()
    fts_tr_after[deleted] = 0.0

    # ===== 9) Edited hypergraph + edited Laplacian 计时 =====
    t_edit0 = time.time()
    hedges_tr_edit = _remove_nodes_from_hyperedges(hedges_tr, deleted.detach().cpu().tolist())
    A_tr_edit = laplacian(hedges_tr_edit, X_tr, args.mediators).to(device)
    edit_time_sec = time.time() - t_edit0
    print(f"[EditedHG] #hyperedges(orig)={len(hedges_tr)} -> #hyperedges(edit)={len(hedges_tr_edit)} | edit={edit_time_sec:.4f}s")

    # ===== 10) Full@EditedHG 评估（模型未更新，只切换结构） =====
    f1_ret, acc_ret = eval_split_hgcn(model_full, fts_tr, lbls_tr, A_tr_edit, mask=retain_mask)
    f1_for, acc_for = eval_split_hgcn(model_full, fts_tr, lbls_tr, A_tr_edit, mask=del_mask)
    print(f"[Full@EditedHG] Retain ACC={acc_ret:.4f} | Forget ACC={acc_for:.4f}")

    results = []

    mia_o, mia_d = _maybe_mia_bank(
        "Full@EditedHG", model_full, A_tr_edit, hedges_tr_edit,
        X_tr, y_tr_np, args, device,
        retain_mask=retain_mask, deleted=deleted
    )
    results.append({
        "run": run_id,
        "method": "Full@EditedHG",
        "K": 0,
        "edit_time_sec": edit_time_sec,
        "time_sec": full_train_time,
        "total_time_sec": edit_time_sec + full_train_time,
        "test_acc": acc_test_full,
        "retain_acc": acc_ret,
        "forget_acc": acc_for,
        "mia_overall_auc": mia_o,
        "mia_deleted_auc": mia_d,
        "seed": seed
    })

    # ===== 11) FT-K =====
    print("\n== FT-K (warm-start) on EditedHG ==")
    for K in args.ft_steps:
        m = _clone_model(model_full)

        # 用 edited structure 训练
        m.structure = A_tr_edit
        for l in m.layers:
            l.reapproximate = False

        t1 = time.time()
        m = finetune_steps_hgcn(
            m,
            x_after=fts_tr_after,
            y=lbls_tr,
            retain_mask=retain_mask,
            steps=K,
            lr=args.ft_lr,
            weight_decay=args.ft_wd,
            print_freq=args.ft_print_freq
        )
        ft_time = time.time() - t1

        f1_test_k, acc_test_k = eval_split_hgcn(m, fts_te, lbls_te, A_te, mask=None)
        f1_ret_k, acc_ret_k = eval_split_hgcn(m, fts_tr, lbls_tr, A_tr_edit, mask=retain_mask)
        f1_for_k, acc_for_k = eval_split_hgcn(m, fts_tr, lbls_tr, A_tr_edit, mask=del_mask)

        print(f"[FT-K] K={K:4d} | Test ACC={acc_test_k:.4f} | Retain ACC={acc_ret_k:.4f} | Forget ACC={acc_for_k:.4f} "
              f"| update={ft_time:.4f}s | total={edit_time_sec + ft_time:.4f}s")

        mia_o, mia_d = _maybe_mia_bank(
            f"FT-K@EditedHG(K={K})", m, A_tr_edit, hedges_tr_edit,
            X_tr, y_tr_np, args, device,
            retain_mask=retain_mask, deleted=deleted
        )

        results.append({
            "run": run_id,
            "method": "FT-K@EditedHG",
            "K": K,
            "edit_time_sec": edit_time_sec,
            "time_sec": ft_time,
            "total_time_sec": edit_time_sec + ft_time,
            "test_acc": acc_test_k,
            "retain_acc": acc_ret_k,
            "forget_acc": acc_for_k,
            "mia_overall_auc": mia_o,
            "mia_deleted_auc": mia_d,
            "seed": seed
        })

    # ===== 12) FT-head =====
    print("\n== FT-head (only last layer) on EditedHG ==")
    for K in args.ft_steps:
        m = _clone_model(model_full)
        freeze_all_but_head_hgcn(m)

        m.structure = A_tr_edit
        for l in m.layers:
            l.reapproximate = False

        t1 = time.time()
        m = finetune_steps_hgcn(
            m,
            x_after=fts_tr_after,
            y=lbls_tr,
            retain_mask=retain_mask,
            steps=K,
            lr=args.ft_lr,
            weight_decay=args.ft_wd,
            print_freq=args.ft_print_freq
        )
        ft_time = time.time() - t1

        f1_test_k, acc_test_k = eval_split_hgcn(m, fts_te, lbls_te, A_te, mask=None)
        f1_ret_k, acc_ret_k = eval_split_hgcn(m, fts_tr, lbls_tr, A_tr_edit, mask=retain_mask)
        f1_for_k, acc_for_k = eval_split_hgcn(m, fts_tr, lbls_tr, A_tr_edit, mask=del_mask)

        print(f"[FT-head] K={K:4d} | Test ACC={acc_test_k:.4f} | Retain ACC={acc_ret_k:.4f} | Forget ACC={acc_for_k:.4f} "
              f"| update={ft_time:.4f}s | total={edit_time_sec + ft_time:.4f}s")

        mia_o, mia_d = _maybe_mia_bank(
            f"FT-head@EditedHG(K={K})", m, A_tr_edit, hedges_tr_edit,
            X_tr, y_tr_np, args, device,
            retain_mask=retain_mask, deleted=deleted
        )

        results.append({
            "run": run_id,
            "method": "FT-head@EditedHG",
            "K": K,
            "edit_time_sec": edit_time_sec,
            "time_sec": ft_time,
            "total_time_sec": edit_time_sec + ft_time,
            "test_acc": acc_test_k,
            "retain_acc": acc_ret_k,
            "forget_acc": acc_for_k,
            "mia_overall_auc": mia_o,
            "mia_deleted_auc": mia_d,
            "seed": seed
        })

    print("\nDone.\n")
    return results


# ============================================================
# Summary
# ============================================================

def summarize_and_save(all_rows, out_csv):
    import pandas as pd

    df = pd.DataFrame(all_rows)
    df.to_csv(out_csv, index=False)

    agg_cols = [
        "edit_time_sec", "time_sec", "total_time_sec",
        "test_acc", "retain_acc", "forget_acc",
        "mia_overall_auc", "mia_deleted_auc"
    ]
    g = df.groupby(["method", "K"], dropna=False)[agg_cols].agg(["mean", "std"]).reset_index()

    print("== Summary (mean±std) ==")

    def _fmt(r, col):
        m = r[(col, "mean")]
        s = r[(col, "std")]
        if pd.isna(m):
            return "NA"
        if pd.isna(s):
            return f"{m:.4f}"
        return f"{m:.4f}±{s:.4f}"

    for _, r in g.iterrows():
        method = r[("method", "")]
        K = int(r[("K", "")])

        print(
            f"{method:16s} K={K:4d} | "
            f"edit={_fmt(r,'edit_time_sec')} | "
            f"update={_fmt(r,'time_sec')} | "
            f"total={_fmt(r,'total_time_sec')} | "
            f"test_acc={_fmt(r,'test_acc')} | "
            f"retain_acc={_fmt(r,'retain_acc')} | "
            f"forget_acc={_fmt(r,'forget_acc')} | "
            f"mia_overall={_fmt(r,'mia_overall_auc')} | "
            f"mia_deleted={_fmt(r,'mia_deleted_auc')}"
        )

    print(f"\n[Saved] {out_csv}")


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