#!/usr/bin/env python
# coding: utf-8
"""
HGNN_PartialRetrain_row.py

Partial retraining baselines (reviewer-meaningful):
- warm-start from full model (NOT scratch)
- train only on retained data (retain_mask)
- update only part of parameters (head / last-k blocks / all)
- rebuild edited hypergraph structure after deletion (count as edit_time)

Outputs:
- edit_time_sec, update_time_sec, total_time_sec
- test_acc, retain_acc, forget_acc
- (optional) MIA overall / deleted

This script follows the same dependency/layout style as HGNN_Unlearning_row_nei.py.  :contentReference[oaicite:1]{index=1}
"""

import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter

# ===== project imports (same as your existing script) =====
from config import get_args
from database.data_preprocessing.data_preprocessing_K import preprocess_node_features, generate_hyperedge_dict
from HGNNs_Model.HGNN.HGNN_2 import HGNN_implicit, build_incidence_matrix, compute_degree_vectors
from GIF.GIF_HGNN_ROW_NEI import rebuild_structure_after_node_deletion, train_model

from MIA.MIA_utils import membership_inference

# -------------------------
# Utils: metrics
# -------------------------
def acc_f1_binary(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor):
    """Binary acc + F1 on masked indices. y in {0,1}."""
    if mask is None:
        mask = torch.ones_like(y, dtype=torch.bool)

    idx = mask.nonzero(as_tuple=False).view(-1)
    if idx.numel() == 0:
        return 0.0, 0.0

    pred = logits.argmax(dim=1)
    pred_m = pred[idx]
    y_m = y[idx]

    acc = (pred_m == y_m).float().mean().item()

    # F1 binary
    tp = ((pred_m == 1) & (y_m == 1)).sum().item()
    fp = ((pred_m == 1) & (y_m == 0)).sum().item()
    fn = ((pred_m == 0) & (y_m == 1)).sum().item()
    denom = (2 * tp + fp + fn)
    f1 = (2 * tp / denom) if denom > 0 else 0.0
    return acc, f1

@torch.no_grad()
def eval_on_mask(model, x, y, H, dv_inv, de_inv, mask: torch.Tensor):
    model.eval()
    logits = model(x, H, dv_inv, de_inv)
    acc, f1 = acc_f1_binary(logits, y, mask)
    return acc, f1

# -------------------------
# Partial retrain freezing
# -------------------------
def set_trainable_params(model: nn.Module, mode: str, last_k: int = 2):
    """
    mode:
      - 'all': train all params
      - 'head': train only head-like params (fc/classifier/out)
      - 'lastk': train last_k parameter blocks (heuristic) + head
    """
    # freeze all
    for p in model.parameters():
        p.requires_grad = False

    if mode == "all":
        for p in model.parameters():
            p.requires_grad = True
        return

    name_params = list(model.named_parameters())

    def is_head(n: str):
        n = n.lower()
        return ("classifier" in n) or (n.endswith("fc.weight") or n.endswith("fc.bias")) or ("fc" in n) or ("out" in n) or ("pred" in n) or ("lin" in n)

    if mode == "head":
        for n, p in name_params:
            if is_head(n):
                p.requires_grad = True
        return

    if mode == "lastk":
        # Heuristic block selection: use named_children() order, take last_k modules that have params
        modules = [(n, m) for n, m in model.named_children()]
        train_mod_names = []
        for n, m in reversed(modules):
            has_params = any(True for _ in m.parameters(recurse=True))
            if has_params:
                train_mod_names.append(n)
            if len(train_mod_names) >= last_k:
                break
        train_mod_names = set(train_mod_names)

        # Always include head-like params
        for n, p in name_params:
            if is_head(n):
                p.requires_grad = True

        # Unfreeze selected modules
        for mod_name, mod in model.named_children():
            if mod_name in train_mod_names:
                for p in mod.parameters(recurse=True):
                    p.requires_grad = True
        return

    raise ValueError(f"Unknown pr mode: {mode}")

# -------------------------
# Training loop (masked loss)
# -------------------------
def partial_retrain_masked(
    model,
    x, y,
    H, dv_inv, de_inv,
    retain_mask: torch.Tensor,
    epochs: int,
    lr: float,
    weight_decay: float,
    print_freq: int = 10
):
    """
    Warm-start training on retained nodes only.
    Loss computed only over retain_mask indices.
    """
    model.train()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=lr, weight_decay=weight_decay)

    # simple CE (or use your focal loss if preferred)
    criterion = nn.CrossEntropyLoss(reduction="none")

    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        optimizer.zero_grad()
        logits = model(x, H, dv_inv, de_inv)  # [N, C]

        loss_vec = criterion(logits, y)  # [N]
        loss = loss_vec[retain_mask].mean()

        loss.backward()
        optimizer.step()

        if (ep % print_freq == 0) or (ep == 1) or (ep == epochs):
            with torch.no_grad():
                acc, f1 = acc_f1_binary(logits, y, retain_mask)
            print(f"[PR] epoch {ep:4d}/{epochs} | loss={loss.item():.4f} | retain_acc={acc:.4f} retain_f1={f1:.4f}")

    t_train = time.perf_counter() - t0
    return model, t_train

# -------------------------
# Hypergraph building helpers
# -------------------------
def build_full_train_struct(df_train, X_train, args, device):
    cat_cols = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ]
    hyperedges = generate_hyperedge_dict(
        df_train, cat_cols,
        max_nodes_per_hyperedge=getattr(args, "max_nodes_per_hyperedge_train", 50),
        device=device
    )
    H_sp = build_incidence_matrix(hyperedges, X_train.shape[0])
    dv_np, de_np = compute_degree_vectors(H_sp)

    H_coo = H_sp.tocoo()
    idx = torch.LongTensor(np.vstack((H_coo.row, H_coo.col))).to(device)
    val = torch.FloatTensor(H_coo.data).to(device)
    H = torch.sparse_coo_tensor(idx, val, size=H_coo.shape).coalesce()
    dv = torch.FloatTensor(dv_np).to(device)
    de = torch.FloatTensor(de_np).to(device)
    return hyperedges, H, dv, de

def build_test_struct(df_test, X_test, args, device):
    cat_cols = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ]
    hyperedges_test = generate_hyperedge_dict(
        df_test, cat_cols,
        max_nodes_per_hyperedge=getattr(args, "max_nodes_per_hyperedge", 50),
        device=device
    )
    H_sp = build_incidence_matrix(hyperedges_test, X_test.shape[0])
    dv_np, de_np = compute_degree_vectors(H_sp)

    H_coo = H_sp.tocoo()
    idx = torch.LongTensor(np.vstack((H_coo.row, H_coo.col))).to(device)
    val = torch.FloatTensor(H_coo.data).to(device)
    H = torch.sparse_coo_tensor(idx, val, size=H_coo.shape).coalesce()
    dv = torch.FloatTensor(dv_np).to(device)
    de = torch.FloatTensor(de_np).to(device)
    return hyperedges_test, H, dv, de

# -------------------------
# MIA (optional)
# -------------------------
def mia_keep_del_auc(X, y, df, deleted_idx, args, device, target_model):
    """
    Same keep/del membership construction as your existing script. :contentReference[oaicite:2]{index=2}
    Returns AUC (target).
    """
    cat_cols = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ]
    all_idx = np.arange(X.shape[0])
    keep_idx = np.setdiff1d(all_idx, deleted_idx)

    X_keep, y_keep = X[keep_idx], y[keep_idx]
    X_del,  y_del  = X[deleted_idx], y[deleted_idx]

    df_keep = df.drop(index=deleted_idx).reset_index(drop=True)
    df_del  = df.iloc[deleted_idx].reset_index(drop=True)

    he_keep = generate_hyperedge_dict(
        df_keep, cat_cols,
        max_nodes_per_hyperedge=getattr(args, 'max_nodes_per_hyperedge_train', 50),
        device=device
    )
    he_del  = generate_hyperedge_dict(
        df_del, cat_cols,
        max_nodes_per_hyperedge=getattr(args, 'max_nodes_per_hyperedge_train', 50),
        device=device
    )

    he_attack = {}
    idx_he = 0
    for nodes in he_keep.values():
        he_attack[idx_he] = nodes
        idx_he += 1
    offset = len(X_keep)
    for nodes in he_del.values():
        he_attack[idx_he] = [n + offset for n in nodes]
        idx_he += 1

    X_attack = np.vstack([X_keep, X_del])
    y_attack = np.hstack([y_keep, y_del])
    member_mask = np.concatenate([
        np.ones(len(X_keep), dtype=bool),
        np.zeros(len(X_del),  dtype=bool)
    ])

    _, _, (auc_target, f1_target) = membership_inference(
        X_train=X_attack,
        y_train=y_attack,
        hyperedges=he_attack,
        target_model=target_model,
        args=args,
        device=device,
        member_mask=member_mask
    )
    return float(auc_target), float(f1_target)

# -------------------------
# One run
# -------------------------
def run_one(seed, X_train, y_train, df_train, transformer, X_test, y_test, df_test, args, device):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    N = X_train.shape[0]
    del_ratio = getattr(args, "remove_ratio", 0.30)
    del_count = int(del_ratio * N)
    deleted_idx = np.random.choice(N, size=del_count, replace=False)
    deleted_idx = np.asarray(deleted_idx, dtype=np.int64)

    retain_mask_np = np.ones(N, dtype=bool)
    retain_mask_np[deleted_idx] = False
    retain_mask = torch.from_numpy(retain_mask_np).to(device)

    print(f"[Delete] ratio={del_ratio} | deleted={len(deleted_idx)} / {N}")

    # ----- build full train structure -----
    hyperedges_full, H_full, dv_full, de_full = build_full_train_struct(df_train, X_train, args, device)

    # ----- init & train full model -----
    fts = torch.FloatTensor(X_train).to(device)
    lbls = torch.LongTensor(y_train).to(device)

    model_full = HGNN_implicit(
        in_ch=fts.shape[1],
        n_class=int(y_train.max()) + 1,
        n_hid=getattr(args, 'hidden_dim', 128),
        dropout=getattr(args, 'dropout', 0.1)
    ).to(device)

    optimizer = optim.Adam(model_full.parameters(), lr=getattr(args, 'lr', 0.01), weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=getattr(args, 'milestones', [100, 150]),
        gamma=getattr(args, 'gamma', 0.1)
    )
    criterion = nn.CrossEntropyLoss()

    t_full0 = time.perf_counter()
    model_full = train_model(
        model_full, criterion, optimizer, scheduler,
        fts, lbls, H_full, dv_full, de_full,
        num_epochs=getattr(args, 'epochs', 200), print_freq=10
    )
    full_train_time = time.perf_counter() - t_full0

    # ----- build test structure -----
    _, H_test, dv_test, de_test = build_test_struct(df_test, X_test, args, device)
    fts_test = torch.FloatTensor(X_test).to(device)
    lbls_test = torch.LongTensor(y_test).to(device)

    # ----- before PR eval on edited structure (to match your FT/HGIF style) -----
    # Edit structure after deletion (common cost)
    t_edit0 = time.perf_counter()
    H_edit, dv_edit, de_edit, _ = rebuild_structure_after_node_deletion(
        hyperedges_full, deleted_idx, N, device
    )
    edit_time = time.perf_counter() - t_edit0
    print(f"[EditedHG] edit_time={edit_time:.4f}s")

    # Evaluate full model under edited structure
    acc_te_before, f1_te_before = eval_on_mask(model_full, fts_test, lbls_test, H_test, dv_test, de_test, None)
    acc_ret_before, _ = eval_on_mask(model_full, fts, lbls, H_edit, dv_edit, de_edit, retain_mask)
    acc_for_before, _ = eval_on_mask(model_full, fts, lbls, H_edit, dv_edit, de_edit, ~retain_mask)
    print(f"[Full@EditedHG] test_acc={acc_te_before:.4f} | retain_acc={acc_ret_before:.4f} | forget_acc={acc_for_before:.4f}")

    # ----- partial retrain (warm-start) -----
    pr_model = copy.deepcopy(model_full)
    pr_mode = getattr(args, "pr_mode", "lastk")
    pr_last_k = getattr(args, "pr_last_k", 3)

    set_trainable_params(pr_model, pr_mode, last_k=pr_last_k)

    # Train on retained nodes only (masked loss), using edited structure
    t_up0 = time.perf_counter()
    pr_model, pr_train_time = partial_retrain_masked(
        pr_model,
        fts, lbls,
        H_edit, dv_edit, de_edit,
        retain_mask=retain_mask,
        epochs=getattr(args, "pr_epochs", 20),
        lr=getattr(args, "pr_lr", 1e-3),
        weight_decay=getattr(args, "pr_wd", 0.0),
        print_freq=10
    )
    update_time = time.perf_counter() - t_up0

    # ----- after PR eval -----
    acc_te, f1_te = eval_on_mask(pr_model, fts_test, lbls_test, H_test, dv_test, de_test, None)
    acc_ret, _ = eval_on_mask(pr_model, fts, lbls, H_edit, dv_edit, de_edit, retain_mask)
    acc_for, _ = eval_on_mask(pr_model, fts, lbls, H_edit, dv_edit, de_edit, ~retain_mask)

    # optional MIA
    mia_auc = None
    mia_f1 = None
    if getattr(args, "run_mia", False):
        mia_auc, mia_f1 = mia_keep_del_auc(X_train, y_train, df_train, deleted_idx, args, device, pr_model)

    out = {
        "seed": seed,
        "deleted": int(len(deleted_idx)),
        "full_train_time_sec": float(full_train_time),

        "edit_time_sec": float(edit_time),
        "update_time_sec": float(update_time),
        "total_time_sec": float(edit_time + update_time),

        "test_acc": float(acc_te),
        "retain_acc": float(acc_ret),
        "forget_acc": float(acc_for),

        "mia_auc": None if mia_auc is None else float(mia_auc),
        "mia_f1": None if mia_f1 is None else float(mia_f1),

        "pr_mode": pr_mode,
        "pr_last_k": int(pr_last_k),
        "pr_epochs": int(getattr(args, "pr_epochs", 20)),
        "pr_lr": float(getattr(args, "pr_lr", 1e-3)),
    }
    print(f"[PR@Retained] mode={pr_mode} last_k={pr_last_k} | edit={edit_time:.4f}s update={update_time:.4f}s total={edit_time+update_time:.4f}s "
          f"| test={acc_te:.4f} retain={acc_ret:.4f} forget={acc_for:.4f} | mia={mia_auc}")
    return out

def main():
    args = get_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # ---- load train/test once ----
    train_csv = getattr(args, "train_csv", ACI_TRAIN)
    test_csv  = getattr(args, "test_csv",  ACI_TEST)

    X_train, y_train, df_train, transformer = preprocess_node_features(train_csv, is_test=False)
    y_train = np.asarray(y_train)

    X_test, y_test, df_test, _ = preprocess_node_features(test_csv, is_test=True, transformer=transformer)
    y_test = np.asarray(y_test)

    # print label stats (optional)
    ctr = Counter(y_train.tolist())
    print(f"Train label dist: {ctr}")

    runs = int(getattr(args, "runs", 3))
    base_seed = int(getattr(args, "seed", 1))

    rows = []
    for r in range(runs):
        seed = base_seed + r
        print(f"\n================= RUN {r+1}/{runs} (seed={seed}) =================")
        row = run_one(seed, X_train, y_train, df_train, transformer, X_test, y_test, df_test, args, device)
        rows.append(row)

    # ---- summarize ----
    def mean_std(vals):
        v = np.array(vals, dtype=float)
        return float(v.mean()), float(v.std(ddof=0)) if v.size > 1 else 0.0

    edit_m, edit_s = mean_std([x["edit_time_sec"] for x in rows])
    up_m, up_s = mean_std([x["update_time_sec"] for x in rows])
    tot_m, tot_s = mean_std([x["total_time_sec"] for x in rows])
    te_m, te_s = mean_std([x["test_acc"] for x in rows])
    ret_m, ret_s = mean_std([x["retain_acc"] for x in rows])
    for_m, for_s = mean_std([x["forget_acc"] for x in rows])

    # print("\n== Summary (mean±std) ==")
    # print(f"PR@Retained mode={rows[0]['pr_mode']} last_k={rows[0]['pr_last_k']} | "
    #       f"edit={edit_m:.4f}±{edit_s:.4f} | update={up_m:.4f}±{up_s:.4f} | total={tot_m:.4f}±{tot_s:.4f} | "
    #       f"test_acc={te_m:.4f}±{te_s:.4f} | retain_acc={ret_m:.4f}±{ret_s:.4f} | forget_acc={for_m:.4f}±{for_s:.4f}")
    #
    # if getattr(args, "run_mia", False) and rows[0]["mia_auc"] is not None:
    #     mia_m, mia_s = mean_std([x["mia_auc"] for x in rows])
    #     print(f"mia_auc={mia_m:.4f}±{mia_s:.4f}")
    #
    # # save CSV (optional)
    # out_csv = getattr(args, "out_csv_pr", "partial_retrain_results.csv")
    # try:
    #     import pandas as pd
    #     from paths import ACI_TEST, ACI_TRAIN
    #         pd.DataFrame(rows).to_csv(out_csv, index=False)
    #         print(f"[Saved] {out_csv}")
    # except Exception as e:
    #     print(f"[WARN] Failed to save csv: {e}")
    print("\n== Summary (mean±std) ==")
    print(f"PR@Retained mode={rows[0]['pr_mode']} last_k={rows[0]['pr_last_k']} | "
          f"edit={edit_m:.4f}±{edit_s:.4f} | update={up_m:.4f}±{up_s:.4f} | total={tot_m:.4f}±{tot_s:.4f} | "
          f"test_acc={te_m:.4f}±{te_s:.4f} | retain_acc={ret_m:.4f}±{ret_s:.4f} | forget_acc={for_m:.4f}±{for_s:.4f}")

    if getattr(args, "run_mia", False) and rows[0]["mia_auc"] is not None:
        mia_m, mia_s = mean_std([x["mia_auc"] for x in rows])
        print(f"mia_auc={mia_m:.4f}±{mia_s:.4f}")

    # save CSV (optional)
    out_csv = getattr(args, "out_csv_pr", "partial_retrain_results.csv")
    try:
        import pandas as pd
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"[Saved] {out_csv}")
    except Exception as e:
        print(f"[WARN] Failed to save csv: {e}")

if __name__ == "__main__":
    main()
