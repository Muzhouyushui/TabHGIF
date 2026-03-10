#!/usr/bin/env python
# coding: utf-8
import time
import copy
import numpy as np
import pandas as pd
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ---- project imports (same family as your HGNNP_Unlearning_row_nei.py) ---- :contentReference[oaicite:2]{index=2}
from config import get_args
from utils.common_utils import evaluate_test_acc, evaluate_test_f1
from database.data_preprocessing.data_preprocessing_K import preprocess_node_features, generate_hyperedge_dict
from HGNNs_Model.HGNNP import HGNNP_implicit, build_incidence_matrix, compute_degree_vectors
from GIF.GIF_HGNNP_ROW_NEI import rebuild_structure_after_node_deletion, train_model

# Optional MIA (your repo already uses these)  :contentReference[oaicite:3]{index=3}
from MIA.MIA_utils import membership_inference
from paths import ACI_TEST, ACI_TRAIN

# -------------------------
# Loss (same as your file)
# -------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

# -------------------------
# Metrics helpers
# -------------------------
@torch.no_grad()
def eval_model(model, test_obj):
    model.eval()
    acc = evaluate_test_acc(model, test_obj)
    f1 = evaluate_test_f1(model, test_obj)
    return float(acc), float(f1)

def make_sparse_from_incidence(H_sp, device):
    H_coo = H_sp.tocoo()
    idx = torch.LongTensor(np.vstack((H_coo.row, H_coo.col))).to(device)
    val = torch.FloatTensor(H_coo.data).to(device)
    H = torch.sparse_coo_tensor(idx, val, size=H_coo.shape).coalesce()
    return H

def build_obj_from_edges(X_np, y_np, hyperedges, device):
    """Build (H, dv_inv, de_inv, fts, lbls) for HGNNP."""
    N = X_np.shape[0]
    H_sp = build_incidence_matrix(hyperedges, N)
    dv_np, de_np = compute_degree_vectors(H_sp)

    H = make_sparse_from_incidence(H_sp, device)
    dv = torch.FloatTensor(dv_np).to(device)
    de = torch.FloatTensor(de_np).to(device)
    fts = torch.FloatTensor(X_np).to(device)
    lbls = torch.LongTensor(y_np).to(device)
    return H, dv, de, fts, lbls

# -------------------------
# Partial retrain freezing
# -------------------------
def set_trainable_params(model: nn.Module, pr_mode: str, pr_last_k: int = 2):
    """
    pr_mode:
      - all  : train all params
      - head : train head-like params (classifier/fc/out/lin). If not found, unfreeze last child module.
      - lastk: train last-k child modules that have parameters + head-like params
    """
    for p in model.parameters():
        p.requires_grad = False

    if pr_mode == "all":
        for p in model.parameters():
            p.requires_grad = True
        return

    named = list(model.named_parameters())

    def is_head(name: str):
        n = name.lower()
        return ("classifier" in n) or ("fc" in n) or ("out" in n) or ("pred" in n) or ("linear" in n) or ("lin" in n)

    head_found = False
    for n, p in named:
        if is_head(n):
            p.requires_grad = True
            head_found = True

    if pr_mode == "head":
        if (not head_found):
            # fallback: unfreeze last module that has params
            children = [(n, m) for n, m in model.named_children()]
            for _, m in reversed(children):
                if any(True for _ in m.parameters(recurse=True)):
                    for p in m.parameters(recurse=True):
                        p.requires_grad = True
                    break
        return

    if pr_mode == "lastk":
        # pick last-k named_children that have params
        children = [(n, m) for n, m in model.named_children()]
        picked = []
        for n, m in reversed(children):
            if any(True for _ in m.parameters(recurse=True)):
                picked.append(n)
            if len(picked) >= max(1, int(pr_last_k)):
                break
        picked = set(picked)

        for n, m in model.named_children():
            if n in picked:
                for p in m.parameters(recurse=True):
                    p.requires_grad = True
        return

    raise ValueError(f"Unknown pr_mode: {pr_mode}")

# -------------------------
# Masked training (retained only)
# -------------------------
def partial_retrain_masked(
    model: nn.Module,
    fts: torch.Tensor,
    lbls: torch.Tensor,
    H: torch.Tensor,
    dv: torch.Tensor,
    de: torch.Tensor,
    retain_mask: torch.Tensor,
    epochs: int,
    lr: float,
    weight_decay: float,
    print_freq: int = 10
):
    """
    Warm-start training on retained nodes only:
    loss = mean(loss_i for i in retain_mask)
    """
    model.train()
    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    criterion = FocalLoss(gamma=2.0, reduction='none')

    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        opt.zero_grad()
        logits = model(fts, H, dv, de)  # HGNNP forward signature in your codebase :contentReference[oaicite:4]{index=4}
        loss_vec = criterion(logits, lbls)  # [N]
        loss = loss_vec[retain_mask].mean()
        loss.backward()
        opt.step()

        if (ep == 1) or (ep == epochs) or (ep % print_freq == 0):
            with torch.no_grad():
                pred = logits.argmax(dim=1)
                acc = (pred[retain_mask] == lbls[retain_mask]).float().mean().item()
            print(f"[PR] ep {ep:4d}/{epochs} | loss={loss.item():.4f} | retain_acc={acc:.4f}")

    t_train = time.perf_counter() - t0
    return model, t_train

# -------------------------
# Optional MIA (keep/del)
# -------------------------
def mia_keep_del_auc(X_train, y_train, df_train, deleted_idx, args, device, target_model):
    """
    Same keep/del attack construction pattern used in your codebase. :contentReference[oaicite:5]{index=5}
    Returns target AUC (and F1) from membership_inference.
    """
    cat_cols = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ]

    all_idx = np.arange(X_train.shape[0])
    keep_idx = np.setdiff1d(all_idx, deleted_idx)

    X_keep, y_keep = X_train[keep_idx], y_train[keep_idx]
    X_del, y_del = X_train[deleted_idx], y_train[deleted_idx]

    df_keep = df_train.drop(index=deleted_idx).reset_index(drop=True)
    df_del = df_train.iloc[deleted_idx].reset_index(drop=True)

    he_keep = generate_hyperedge_dict(
        df_keep, cat_cols,
        max_nodes_per_hyperedge=getattr(args, 'max_nodes_per_hyperedge_train', 50),
        device=device
    )
    he_del = generate_hyperedge_dict(
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
        np.zeros(len(X_del), dtype=bool)
    ])

    _, (_, _), (auc_target, f1_target) = membership_inference(
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
def run_one(run_id: int, X_train, y_train, df_train, transformer, args, device):
    seed = int(getattr(args, "seed", 1)) + run_id
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    N = X_train.shape[0]
    del_ratio = float(getattr(args, "remove_ratio", 0.30))
    del_count = int(del_ratio * N)
    deleted_idx = np.random.choice(np.arange(N), size=del_count, replace=False).astype(np.int64)

    retain_mask_np = np.ones(N, dtype=bool)
    retain_mask_np[deleted_idx] = False
    retain_mask = torch.from_numpy(retain_mask_np).to(device)
    del_mask = ~retain_mask

    print(f"\n=== Run {run_id+1} | seed={seed} ===")
    print(f"[Delete] {len(deleted_idx)}/{N} ({del_ratio*100:.1f}%)")

    # ---- Build full hypergraph ----
    cat_cols = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ]
    hyperedges = generate_hyperedge_dict(
        df_train, cat_cols,
        max_nodes_per_hyperedge=getattr(args, 'max_nodes_per_hyperedge_train', 50),
        device=device
    )
    H, dv, de, fts, lbls = build_obj_from_edges(X_train, y_train, hyperedges, device)

    # ---- Train full model (baseline checkpoint) ----
    model_full = HGNNP_implicit(
        in_ch=fts.shape[1],
        n_class=int(np.max(y_train)) + 1,
        n_hid=getattr(args, 'hidden_dim', 128),
        dropout=getattr(args, 'dropout', 0.1)
    ).to(device)

    optimizer = optim.Adam(model_full.parameters(), lr=getattr(args, 'lr', 0.01), weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=getattr(args, 'milestones', [100, 150]),
        gamma=getattr(args, 'gamma', 0.1)
    )
    criterion = FocalLoss(gamma=2.0, reduction='mean')

    t_full0 = time.perf_counter()
    model_full = train_model(
        model_full, criterion, optimizer, scheduler,
        fts, lbls, H, dv, de,
        num_epochs=getattr(args, 'epochs', 200), print_freq=10
    )
    full_train_time = time.perf_counter() - t_full0

    # ---- Prepare test object ----
    test_csv = getattr(args, "test_csv", ACI_TEST)
    X_test, y_test, df_test, _ = preprocess_node_features(test_csv, is_test=True, transformer=transformer)

    hyperedges_test = generate_hyperedge_dict(
        df_test, cat_cols,
        max_nodes_per_hyperedge=getattr(args, 'max_nodes_per_hyperedge', 50),
        device=device
    )
    H_te, dv_te, de_te, fts_te, lbls_te = build_obj_from_edges(X_test, y_test, hyperedges_test, device)
    test_obj = {"x": fts_te, "y": lbls_te, "H": H_te, "dv_inv": dv_te, "de_inv": de_te}

    acc_before, f1_before = eval_model(model_full, test_obj)
    print(f"[Full] train_time={full_train_time:.4f}s | test_acc={acc_before:.4f} f1={f1_before:.4f}")

    # =====================================================
    # Edited hypergraph build (common cost) + timing
    # =====================================================
    t_edit0 = time.perf_counter()
    H_new, dv_new, de_new, _ = rebuild_structure_after_node_deletion(
        hyperedges, deleted_idx, N, device
    )
    edit_time = time.perf_counter() - t_edit0
    print(f"[EditedHG] edit_time={edit_time:.4f}s")

    # Evaluate full model under EditedHG (optional)
    with torch.no_grad():
        logits_full_edited = model_full(fts, H_new, dv_new, de_new)
        pred = logits_full_edited.argmax(dim=1)
        retain_acc_full = (pred[retain_mask] == lbls[retain_mask]).float().mean().item()
        forget_acc_full = (pred[del_mask] == lbls[del_mask]).float().mean().item()
    print(f"[Full@EditedHG] retain_acc={retain_acc_full:.4f} | forget_acc={forget_acc_full:.4f}")

    # =====================================================
    # Partial Retraining (warm-start) on retained only
    # =====================================================
    pr_model = copy.deepcopy(model_full)

    pr_mode = getattr(args, "pr_mode", "lastk")  # head / lastk / all
    pr_last_k = int(getattr(args, "pr_last_k", 2))
    pr_epochs = int(getattr(args, "pr_epochs", 20))
    pr_lr = float(getattr(args, "pr_lr", 1e-3))
    pr_wd = float(getattr(args, "pr_wd", 0.0))

    set_trainable_params(pr_model, pr_mode, pr_last_k)
    print(f"[PR] mode={pr_mode} last_k={pr_last_k} epochs={pr_epochs} lr={pr_lr}")

    t_up0 = time.perf_counter()
    pr_model, pr_train_time = partial_retrain_masked(
        pr_model,
        fts=fts, lbls=lbls,
        H=H_new, dv=dv_new, de=de_new,
        retain_mask=retain_mask,
        epochs=pr_epochs,
        lr=pr_lr,
        weight_decay=pr_wd,
        print_freq=10
    )
    update_time = time.perf_counter() - t_up0
    total_time = edit_time + update_time

    # ---- Evaluate after PR ----
    acc_after, f1_after = eval_model(pr_model, test_obj)

    with torch.no_grad():
        logits_pr_edited = pr_model(fts, H_new, dv_new, de_new)
        pred2 = logits_pr_edited.argmax(dim=1)
        retain_acc = (pred2[retain_mask] == lbls[retain_mask]).float().mean().item()
        forget_acc = (pred2[del_mask] == lbls[del_mask]).float().mean().item()

    mia_overall = None
    mia_deleted = None
    if bool(getattr(args, "run_mia", False)):
        mia_overall, _ = mia_keep_del_auc(X_train, y_train, df_train, deleted_idx, args, device, pr_model)
        # (可选) 你如果想要 deleted-only 的 MIA，可以自己按你工程的方式构造 member_mask，这里先保持简洁

    print(f"[PR@EditedHG] edit={edit_time:.4f}s update={update_time:.4f}s total={total_time:.4f}s | "
          f"test_acc={acc_after:.4f} retain_acc={retain_acc:.4f} forget_acc={forget_acc:.4f} | mia_overall={mia_overall}")

    return {
        "run": run_id,
        "seed": seed,
        "method": f"PR-{pr_mode}@EditedHG",
        "pr_mode": pr_mode,
        "pr_last_k": pr_last_k,
        "pr_epochs": pr_epochs,
        "deleted": int(len(deleted_idx)),
        "edit_time_sec": float(edit_time),
        "update_time_sec": float(update_time),
        "total_time_sec": float(total_time),
        "test_acc": float(acc_after),
        "test_f1": float(f1_after),
        "retain_acc": float(retain_acc),
        "forget_acc": float(forget_acc),
        "mia_overall_auc": mia_overall,
        "full_train_time_sec": float(full_train_time),
    }

def main():
    args = get_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # defaults (in case not defined in config)
    if not hasattr(args, "runs"): args.runs = 3
    if not hasattr(args, "remove_ratio"): args.remove_ratio = 0.30
    if not hasattr(args, "train_csv"): args.train_csv = ACI_TRAIN
    if not hasattr(args, "test_csv"): args.test_csv = ACI_TEST

    if not hasattr(args, "pr_mode"): args.pr_mode = "lastk"      # head / lastk / all
    if not hasattr(args, "pr_last_k"): args.pr_last_k = 2
    if not hasattr(args, "pr_epochs"): args.pr_epochs = 20
    if not hasattr(args, "pr_lr"): args.pr_lr = 1e-3
    if not hasattr(args, "pr_wd"): args.pr_wd = 0.0
    if not hasattr(args, "run_mia"): args.run_mia = False

    # ---- load train once ----
    X_train, y_train, df_train, transformer = preprocess_node_features(args.train_csv, is_test=False)
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train, dtype=int)

    # sanity label distribution
    ctr = Counter(y_train.tolist())
    print(f"Train label dist: {ctr}")

    rows = []
    for r in range(int(args.runs)):
        rows.append(run_one(r, X_train, y_train, df_train, transformer, args, device))

    df = pd.DataFrame(rows)

    def ms(col):
        v = df[col].astype(float).values
        return float(v.mean()), float(v.std(ddof=0)) if len(v) > 1 else 0.0

    edit_m, edit_s = ms("edit_time_sec")
    up_m, up_s = ms("update_time_sec")
    tot_m, tot_s = ms("total_time_sec")
    te_m, te_s = ms("test_acc")
    ret_m, ret_s = ms("retain_acc")
    for_m, for_s = ms("forget_acc")

    print("\n== Summary (mean±std) ==")
    print(f"{df['method'].iloc[0]} | "
          f"edit={edit_m:.4f}±{edit_s:.4f} | update={up_m:.4f}±{up_s:.4f} | total={tot_m:.4f}±{tot_s:.4f} | "
          f"test_acc={te_m:.4f}±{te_s:.4f} | retain_acc={ret_m:.4f}±{ret_s:.4f} | forget_acc={for_m:.4f}±{for_s:.4f}")

    out_csv = getattr(args, "out_csv_pr", "hgnnp_partial_retrain_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n[Saved] {out_csv}")

if __name__ == "__main__":
    main()
