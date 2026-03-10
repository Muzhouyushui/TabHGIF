# run_ft_hgcn_bank_col_zero.py
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

# ===== Bank data preprocessing & column deletion (according to your existing Bank-HGCN col pipeline) =====
from bank.HGCN.data_preprocessing_col_bank import (
    preprocess_node_features,
    generate_hyperedge_dict,
    delete_feature_columns_hgcn,
)

# ===== HGCN model & Laplacian =====
from bank.HGCN.HGCN import HyperGCN, laplacian

# ===== Training / Evaluation / MIA =====
from bank.HGCN.GIF_HGCN_COL_bank import train_model
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


def _member_mask_all_train(N: int):
    # overall MIA: all training samples are treated as member=1 (comparison with non-training samples is generated/constructed inside MIA)
    return np.ones(N, dtype=np.int64)


def _new_model_hgcn_bank_col(X_np, hyperedges_list, args, device):
    """
    Construct the model according to the Bank-HGCN initialization style, and fix the structure
    """
    # Inject d/c required by HGCN directly into args (consistent with your col script)
    args.d = X_np.shape[1]
    args.c = 2  # bank binary classification

    model = HyperGCN(
        X_np.shape[0],
        hyperedges_list,
        X_np,
        args
    ).to(device)

    A = laplacian(hyperedges_list, X_np, args.mediators).to(device)
    model.structure = A
    for layer in model.layers:
        layer.reapproximate = False
    return model, A


def eval_hgcn(model, x_tensor, y_tensor, A_tensor):
    """
    Adapt to the interface of bank.HGCN.HGCN_utils.evaluate_hgcn_*
    """
    model.structure = A_tensor
    for l in model.layers:
        l.reapproximate = False

    data = {"lap": x_tensor, "y": y_tensor}
    f1 = evaluate_hgcn_f1(model, data)
    acc = evaluate_hgcn_acc(model, data)
    return f1, acc


def freeze_all_but_head_hgcn(model):
    """
    FT-head: only train the parameters of the last layer
    """
    for p in model.parameters():
        p.requires_grad = False

    if hasattr(model, "layers") and len(model.layers) > 0:
        for p in model.layers[-1].parameters():
            p.requires_grad = True
        return

    # fallback
    for n, p in model.named_parameters():
        if any(k in n.lower() for k in ["out", "classifier", "final", "last"]):
            p.requires_grad = True


def finetune_steps_hgcn(
    model,
    x_after: torch.Tensor,
    y: torch.Tensor,
    steps: int,
    lr: float,
    weight_decay: float = 0.0,
    print_freq: int = 0,
):
    """
    Perform K-step finetuning on EditedHG (for column deletion, supervision is usually applied on all samples)
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
        loss = criterion(out, y)

        loss.backward()
        opt.step()

        if print_freq > 0 and (step == 1 or step % print_freq == 0 or step == steps):
            with torch.no_grad():
                pred = out.argmax(dim=1)
                acc = (pred == y).float().mean().item()
            print(f"[FT] step {step:4d}/{steps} | loss={loss.item():.4f} | train_acc={acc:.4f}")

    return model


def _maybe_mia_bank_col(tag, model, hedges_used, X_train_np, y_train_np, args, device):
    """
    For the column deletion setting, only overall MIA is provided first; deleted-specific in column deletion usually needs separate definition (e.g., column-related subset), so None is returned for now
    """
    mia_overall_auc = None
    if not args.run_mia:
        return mia_overall_auc

    try:
        mm = _member_mask_all_train(len(y_train_np))
        print(f"— MIA on {tag} (overall) —")
        _, _, tgt = membership_inference_hgcn(
            X_train=X_train_np,
            y_train=y_train_np,
            hyperedges=hedges_used,
            target_model=model,
            args=args,
            device=device,
            member_mask=mm
        )
        if tgt is not None:
            mia_overall_auc = float(tgt[0])
    except Exception as e:
        print(f"[WARN] MIA failed on {tag}: {e}")
        mia_overall_auc = None

    return mia_overall_auc


# ============================================================
# Core
# ============================================================

def get_args():
    p = argparse.ArgumentParser("HGCN FT baselines on Bank (column deletion, feature-zero)")

    # ===== Data =====
    p.add_argument("--data_csv", type=str,
                   default="/root/autodl-tmp/TabHGIF/data_banking/bank/bank-full.csv")
    p.add_argument("--split_ratio", type=float, default=0.2)

    # Column unlearning targets (Bank)
    p.add_argument("--columns_to_unlearn", type=str, nargs="+",
                   default=["education"],
                   help="Original column names to unlearn, multiple columns are allowed")

    # ===== Feature / Hypergraph columns =====
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
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--fast", action="store_true")
    p.add_argument("--cuda", action="store_true", default=True)

    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight_decay", type=float, default=0.001)
    p.add_argument("--milestones", type=int, nargs="+", default=[100, 150])
    p.add_argument("--gamma", type=float, default=0.1)
    p.add_argument("--log_every", type=int, default=10)

    # ===== FT baselines =====
    p.add_argument("--ft_steps", type=int, nargs="+", default=[50, 100, 200])
    p.add_argument("--ft_lr", type=float, default=1e-3)
    p.add_argument("--ft_wd", type=float, default=0.0)
    p.add_argument("--ft_print_freq", type=int, default=0)

    # ===== MIA =====
    p.add_argument("--run_mia", action="store_true", default=False)

    # ===== Multi-run + save =====
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--out_csv", type=str, default="ft_hgcn_bank_col_zero_results.csv")

    return p.parse_args()


def run_one(args, run_id: int):
    device = _device(args)
    seed = args.seed + run_id
    set_seed(seed)
    print(f"[Device] {device} | seed={seed}")

    # ===== 1) Read Bank data and split =====
    df_full = pd.read_csv(args.data_csv, sep=';', header=0)
    assert 'y' in df_full.columns, "Column 'y' was not found in the CSV"

    df_tr, df_te = train_test_split(
        df_full,
        test_size=args.split_ratio,
        random_state=42 + run_id,
        stratify=df_full['y']
    )
    df_tr = df_tr.reset_index(drop=True)
    df_te = df_te.reset_index(drop=True)

    print(f"TRAIN={len(df_tr)} samples, TEST={len(df_te)} samples")

    # ===== 2) Preprocessing (train fit + test transform) =====
    X_tr, y_tr_np, df_tr_proc, transformer = preprocess_node_features(df_tr, is_test=False)
    X_te, y_te_np, df_te_proc, _ = preprocess_node_features(df_te, is_test=True, transformer=transformer)

    # ===== 3) Original hypergraph =====
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
    hedges_tr = list(hyper_tr.values())
    hedges_te = list(hyper_te.values())

    # ===== 4) Tensors =====
    fts_tr = torch.from_numpy(X_tr).float().to(device)
    lbls_tr = torch.tensor(y_tr_np, dtype=torch.long, device=device)
    fts_te = torch.from_numpy(X_te).float().to(device)
    lbls_te = torch.tensor(y_te_np, dtype=torch.long, device=device)

    # ===== 5) Full model + structures =====
    model_full, A_tr = _new_model_hgcn_bank_col(X_tr, hedges_tr, args, device)
    A_te = laplacian(hedges_te, X_te, args.mediators).to(device)

    # ===== 6) Full train =====
    print("== Train Full Model ==")
    t0 = time.time()

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model_full.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.milestones, gamma=args.gamma
    )

    model_full = train_model(
        model_full, criterion, optimizer, scheduler,
        fts_tr, lbls_tr,
        num_epochs=args.epochs,
        print_freq=args.log_every
    )
    full_train_time = time.time() - t0

    f1_test_full, acc_test_full = eval_hgcn(model_full, fts_te, lbls_te, A_te)
    print(f"[Full] Test ACC={acc_test_full:.4f} | F1={f1_test_full:.4f} | train_time={full_train_time:.4f}s")

    # ===== 7) Column unlearning: feature-zero + edited hypergraph + edited Laplacian =====
    print(f"[Column Unlearning] columns_to_unlearn={args.columns_to_unlearn}")

    t_edit0 = time.time()
    X_zero_tr, hyper_zero_tr, A_zero_tr = delete_feature_columns_hgcn(
        X_tensor=fts_tr,
        transformer=transformer,
        column_names=args.columns_to_unlearn,
        hyperedges=hyper_tr,
        mediators=args.mediators,
        use_cuda=(device.type == 'cuda')
    )
    edit_time_sec = time.time() - t_edit0

    # The same column deletion evaluation must also be applied on the test side
    X_zero_te, hyper_zero_te, A_zero_te = delete_feature_columns_hgcn(
        X_tensor=fts_te,
        transformer=transformer,
        column_names=args.columns_to_unlearn,
        hyperedges=hyper_te,
        mediators=args.mediators,
        use_cuda=(device.type == 'cuda')
    )

    # Convert uniformly to tensor / list
    if torch.is_tensor(X_zero_tr):
        fts_tr_zero = X_zero_tr.float().to(device)
        X_zero_tr_np = X_zero_tr.detach().cpu().numpy()
    else:
        X_zero_tr_np = X_zero_tr
        fts_tr_zero = torch.from_numpy(X_zero_tr).float().to(device)

    if torch.is_tensor(X_zero_te):
        fts_te_zero = X_zero_te.float().to(device)
    else:
        fts_te_zero = torch.from_numpy(X_zero_te).float().to(device)

    A_zero_tr = A_zero_tr.to(device)
    A_zero_te = A_zero_te.to(device)

    hedges_tr_edit = list(hyper_zero_tr.values()) if isinstance(hyper_zero_tr, dict) else list(hyper_zero_tr)
    hedges_te_edit = list(hyper_zero_te.values()) if isinstance(hyper_zero_te, dict) else list(hyper_zero_te)

    print(f"[EditedHG] train #hyperedges(orig)={len(hedges_tr)} -> #hyperedges(edit)={len(hedges_tr_edit)}")
    print(f"[EditedHG] test  #hyperedges(orig)={len(hedges_te)} -> #hyperedges(edit)={len(hedges_te_edit)}")
    print(f"[Edit] time={edit_time_sec:.4f}s")

    # ===== 8) Full@EditedHG (no model update, only switch structure + zeroed features) =====
    f1_train_e, acc_train_e = eval_hgcn(model_full, fts_tr_zero, lbls_tr, A_zero_tr)
    f1_test_e, acc_test_e = eval_hgcn(model_full, fts_te_zero, lbls_te, A_zero_te)
    print(f"[Full@EditedHG] Train ACC={acc_train_e:.4f} | Test ACC={acc_test_e:.4f}")

    results = []

    mia_o = _maybe_mia_bank_col(
        "Full@EditedHG", model_full, hedges_tr_edit, X_zero_tr_np, y_tr_np, args, device
    )
    results.append({
        "run": run_id,
        "method": "Full@EditedHG",
        "K": 0,
        "edit_time_sec": edit_time_sec,
        "time_sec": 0.0,                      # edit only, no parameter update
        "total_time_sec": edit_time_sec,
        "test_acc": acc_test_e,
        "train_acc": acc_train_e,
        "mia_overall_auc": mia_o,
        "seed": seed
    })

    # ===== 9) FT-K =====
    print("\n== FT-K (warm-start on EditedHG) ==")
    for K in args.ft_steps:
        m = _clone_model(model_full)
        m.structure = A_zero_tr
        for l in m.layers:
            l.reapproximate = False

        t1 = time.time()
        m = finetune_steps_hgcn(
            m,
            x_after=fts_tr_zero,
            y=lbls_tr,
            steps=K,
            lr=args.ft_lr,
            weight_decay=args.ft_wd,
            print_freq=args.ft_print_freq
        )
        ft_time = time.time() - t1

        f1_tr_k, acc_tr_k = eval_hgcn(m, fts_tr_zero, lbls_tr, A_zero_tr)
        f1_te_k, acc_te_k = eval_hgcn(m, fts_te_zero, lbls_te, A_zero_te)

        print(f"[FT-K] K={K:4d} | Train ACC={acc_tr_k:.4f} | Test ACC={acc_te_k:.4f} "
              f"| edit={edit_time_sec:.4f}s | update={ft_time:.4f}s | total={edit_time_sec + ft_time:.4f}s")

        mia_o = _maybe_mia_bank_col(
            f"FT-K@EditedHG(K={K})", m, hedges_tr_edit, X_zero_tr_np, y_tr_np, args, device
        )

        results.append({
            "run": run_id,
            "method": "FT-K@EditedHG",
            "K": K,
            "edit_time_sec": edit_time_sec,
            "time_sec": ft_time,
            "total_time_sec": edit_time_sec + ft_time,
            "test_acc": acc_te_k,
            "train_acc": acc_tr_k,
            "mia_overall_auc": mia_o,
            "seed": seed
        })

    # ===== 10) FT-head =====
    print("\n== FT-head (only last layer) on EditedHG ==")
    for K in args.ft_steps:
        m = _clone_model(model_full)
        freeze_all_but_head_hgcn(m)

        m.structure = A_zero_tr
        for l in m.layers:
            l.reapproximate = False

        t1 = time.time()
        m = finetune_steps_hgcn(
            m,
            x_after=fts_tr_zero,
            y=lbls_tr,
            steps=K,
            lr=args.ft_lr,
            weight_decay=args.ft_wd,
            print_freq=args.ft_print_freq
        )
        ft_time = time.time() - t1

        f1_tr_k, acc_tr_k = eval_hgcn(m, fts_tr_zero, lbls_tr, A_zero_tr)
        f1_te_k, acc_te_k = eval_hgcn(m, fts_te_zero, lbls_te, A_zero_te)

        print(f"[FT-head] K={K:4d} | Train ACC={acc_tr_k:.4f} | Test ACC={acc_te_k:.4f} "
              f"| edit={edit_time_sec:.4f}s | update={ft_time:.4f}s | total={edit_time_sec + ft_time:.4f}s")

        mia_o = _maybe_mia_bank_col(
            f"FT-head@EditedHG(K={K})", m, hedges_tr_edit, X_zero_tr_np, y_tr_np, args, device
        )

        results.append({
            "run": run_id,
            "method": "FT-head@EditedHG",
            "K": K,
            "edit_time_sec": edit_time_sec,
            "time_sec": ft_time,
            "total_time_sec": edit_time_sec + ft_time,
            "test_acc": acc_te_k,
            "train_acc": acc_tr_k,
            "mia_overall_auc": mia_o,
            "seed": seed
        })

    print("\nDone.\n")
    return results


# ============================================================
# Summary
# ============================================================

def summarize_and_save(all_rows, out_csv):
    df = pd.DataFrame(all_rows)
    df.to_csv(out_csv, index=False)

    agg_cols = [
        "edit_time_sec", "time_sec", "total_time_sec",
        "test_acc", "train_acc", "mia_overall_auc"
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
            f"train_acc={_fmt(r,'train_acc')} | "
            f"mia_overall={_fmt(r,'mia_overall_auc')}"
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