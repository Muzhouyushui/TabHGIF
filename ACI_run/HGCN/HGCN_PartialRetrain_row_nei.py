#!/usr/bin/env python
# coding: utf-8
"""
HGCN_PartialRetrain_row_nei.py (FIXED)

Fixes:
1) Evaluate on EditedHG must use fts_after (feature-zero after deletion), not fts_tr.
2) Retain/forget accuracy computed with explicit masks (manual), avoiding any mask-ignoring in utils.
3) Add safety asserts for masks & devices.

Based on your uploaded script.  :contentReference[oaicite:1]{index=1}
"""

import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from GIF.GIF_HGCN_ROW_NEI import train_model, apply_node_deletion_unlearning
from utils.common_utils import evaluate_hgcn_f1, evaluate_hgcn_acc
from database.data_preprocessing.data_preprocessing_K import preprocess_node_features, generate_hyperedge_dict
from HGCN.HyperGCN import HyperGCN, laplacian
from config_HGCN import get_args
from MIA.MIA_HGCN import membership_inference_hgcn


# =========================================================
# Helpers
# =========================================================
def _device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _ensure_attr(args, name, default):
    if not hasattr(args, name):
        setattr(args, name, default)


def _build_cfg_for_hgcn(args, X_np, y_tensor):
    cfg = lambda: None
    cfg.d = X_np.shape[1]
    cfg.depth = args.depth
    cfg.c = int(y_tensor.max().item()) + 1
    cfg.dropout = args.dropout
    cfg.fast = args.fast
    cfg.mediators = args.mediators
    cfg.cuda = True
    cfg.dataset = getattr(args, "dataset", "adult")
    return cfg


def _new_model(X_np, y_tensor, hyperedges, args, device):
    cfg = _build_cfg_for_hgcn(args, X_np, y_tensor)
    model = HyperGCN(
        num_nodes=X_np.shape[0],
        edge_list=hyperedges,
        X_init=X_np,
        args=cfg
    ).to(device)

    A = laplacian(hyperedges, X_np, args.mediators).to(device)
    model.structure = A
    for l in model.layers:
        l.reapproximate = False
    return model, A


def _set_trainable_params_hgcn(model: nn.Module, pr_mode: str, pr_last_k: int = 2):
    for p in model.parameters():
        p.requires_grad = False

    if pr_mode == "all":
        for p in model.parameters():
            p.requires_grad = True
        return

    named = list(model.named_parameters())

    def is_head(name: str):
        n = name.lower()
        return ("classifier" in n) or ("linear" in n) or ("out" in n) or ("pred" in n) or ("fc" in n)

    head_found = False
    if pr_mode in ("head", "lastk"):
        for n, p in named:
            if is_head(n):
                p.requires_grad = True
                head_found = True

    if pr_mode == "head":
        if (not head_found) and hasattr(model, "layers") and len(model.layers) > 0:
            for p in model.layers[-1].parameters():
                p.requires_grad = True
        return

    if pr_mode == "lastk":
        if hasattr(model, "layers") and len(model.layers) > 0:
            k = max(1, int(pr_last_k))
            for layer in model.layers[-k:]:
                for p in layer.parameters():
                    p.requires_grad = True
        return

    raise ValueError(f"Unknown pr_mode: {pr_mode}")


def _masked_nll_loss(logp: torch.Tensor, y: torch.Tensor, mask: torch.Tensor):
    loss_vec = nn.NLLLoss(reduction="none")(logp, y)  # [N]
    idx = mask.nonzero(as_tuple=False).view(-1)
    if idx.numel() == 0:
        return loss_vec.mean() * 0.0
    return loss_vec[idx].mean()


@torch.no_grad()
def _masked_acc_from_logits(logp: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:
    pred = logp.argmax(dim=1)
    idx = mask.nonzero(as_tuple=False).view(-1)
    if idx.numel() == 0:
        return 0.0
    return float((pred[idx] == y[idx]).float().mean().item())


@torch.no_grad()
def _eval_edited_ret_for(model, fts_after, y, retain_mask, del_mask) -> (float, float):
    model.eval()
    out = model(fts_after)
    acc_ret = _masked_acc_from_logits(out, y, retain_mask)
    acc_del = _masked_acc_from_logits(out, y, del_mask)
    return acc_ret, acc_del


def partial_retrain_warmstart(
    model: nn.Module,
    fts_after: torch.Tensor,
    lbls: torch.Tensor,
    retain_mask: torch.Tensor,
    del_mask: torch.Tensor,
    args,
):
    model.train()

    lr = getattr(args, "pr_lr", 1e-3)
    wd = getattr(args, "pr_wd", 0.0)
    epochs = getattr(args, "pr_epochs", 20)
    milestones = getattr(args, "pr_milestones", [])
    gamma = getattr(args, "pr_gamma", 0.1)
    log_every = getattr(args, "pr_log_every", 10)

    # Safety checks: these catch silent mask bugs early
    assert retain_mask.dtype == torch.bool and del_mask.dtype == torch.bool
    assert retain_mask.device == fts_after.device == lbls.device
    assert retain_mask.shape[0] == fts_after.shape[0] == lbls.shape[0]
    assert 0 < int(retain_mask.sum().item()) < retain_mask.numel()

    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=wd)
    sch = None
    if milestones and isinstance(milestones, (list, tuple)) and len(milestones) > 0:
        sch = optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=gamma)

    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        opt.zero_grad()
        out = model(fts_after)
        loss = _masked_nll_loss(out, lbls, retain_mask)
        loss.backward()
        opt.step()
        if sch is not None:
            sch.step()

        if (ep == 1) or (ep == epochs) or (ep % log_every == 0):
            with torch.no_grad():
                acc_r = _masked_acc_from_logits(out, lbls, retain_mask)
                acc_d = _masked_acc_from_logits(out, lbls, del_mask)
            print(f"[PR] ep {ep:4d}/{epochs} | loss={loss.item():.4f} | retain_acc={acc_r:.4f} | del_acc={acc_d:.4f}")

    t_train = time.perf_counter() - t0
    return model, t_train


def _build_masks(N: int, deleted: torch.LongTensor, device):
    retain_mask = torch.ones(N, dtype=torch.bool, device=device)
    retain_mask[deleted] = False
    del_mask = ~retain_mask
    return retain_mask, del_mask


# =========================================================
# One run
# =========================================================
def run_one(run_id: int, args, device):
    seed = int(getattr(args, "seed", 1)) + run_id
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ---- load data ----
    X_tr, y_tr, df_tr, transformer = preprocess_node_features(args.train_csv, is_test=False)
    X_te, y_te_raw, df_te, _ = preprocess_node_features(args.test_csv, is_test=True, transformer=transformer)

    # adult label fix
    y_te = [1 if v.strip().rstrip('.') == ">50K" else 0 for v in df_te["income"].values]

    # ---- hyperedges ----
    hedges_tr = list(generate_hyperedge_dict(
        df_tr, args.categate_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        device=device
    ).values())
    hedges_te = list(generate_hyperedge_dict(
        df_te, args.categate_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        device=device
    ).values())

    # ---- tensors ----
    fts_tr = torch.from_numpy(X_tr).float().to(device)
    lbls_tr = torch.tensor(np.array(y_tr, dtype=int), dtype=torch.long, device=device)
    fts_te = torch.from_numpy(X_te).float().to(device)
    lbls_te = torch.tensor(np.array(y_te, dtype=int), dtype=torch.long, device=device)

    N = X_tr.shape[0]
    n_del = int(N * float(args.remove_ratio))
    deleted_idx = np.random.choice(N, n_del, replace=False)
    deleted = torch.tensor(deleted_idx, dtype=torch.long, device=device)
    retain_mask, del_mask = _build_masks(N, deleted, device)

    print(f"\n[Run {run_id+1}] seed={seed} | deleted={len(deleted)}/{N} ({args.remove_ratio})")

    # ---- train full model on original graph ----
    model_full, A_tr = _new_model(X_tr, lbls_tr, hedges_tr, args, device)
    A_te = laplacian(hedges_te, X_te, args.mediators).to(device)

    opt = optim.Adam(model_full.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sch = optim.lr_scheduler.MultiStepLR(opt, milestones=args.milestones, gamma=args.gamma)
    crit = nn.NLLLoss()

    print("== Train Full Model ==")
    t_full0 = time.perf_counter()
    model_full = train_model(
        model_full, crit, opt, sch,
        fts_tr, lbls_tr,
        num_epochs=args.epochs,
        print_freq=args.log_every
    )
    full_train_time = time.perf_counter() - t_full0

    # eval full on test
    model_full.structure = A_te
    for l in model_full.layers:
        l.reapproximate = False
    test_acc_full = evaluate_hgcn_acc(model_full, {"x": fts_te, "y": lbls_te})
    test_f1_full = evaluate_hgcn_f1(model_full, {"x": fts_te, "y": lbls_te})
    print(f"[Full] test_acc={test_acc_full:.4f} test_f1={test_f1_full:.4f} | train_time={full_train_time:.4f}s")

    # ============================================================
    # Edited hypergraph (structure + feature-zero) + timing
    # ============================================================
    print("\n== Build Edited Hypergraph (apply_node_deletion_unlearning) ==")
    t_edit0 = time.perf_counter()
    fts_after, hedges_edit, A_edit = apply_node_deletion_unlearning(
        fts_tr, edge_list=hedges_tr,
        deleted_nodes=deleted,
        mediators=args.mediators, device=device
    )
    edit_time = time.perf_counter() - t_edit0
    print(f"[EditedHG] edit_time={edit_time:.4f}s | #edges {len(hedges_tr)} -> {len(hedges_edit)}")

    # Full model evaluated under edited structure (must use fts_after)
    model_full_edit = copy.deepcopy(model_full)
    model_full_edit.structure = A_edit
    for l in model_full_edit.layers:
        l.reapproximate = False

    acc_ret_full, acc_for_full = _eval_edited_ret_for(
        model_full_edit, fts_after, lbls_tr, retain_mask, del_mask
    )
    print(f"[Full@EditedHG] retain_acc={acc_ret_full:.4f} | forget_acc={acc_for_full:.4f}")

    # ============================================================
    # Partial Retraining (warm-start) on retained nodes only
    # ============================================================
    pr_mode = getattr(args, "pr_mode", "lastk")
    pr_last_k = int(getattr(args, "pr_last_k", 2))

    pr_model = copy.deepcopy(model_full)
    pr_model.structure = A_edit
    for l in pr_model.layers:
        l.reapproximate = False

    _set_trainable_params_hgcn(pr_model, pr_mode, pr_last_k)

    print(f"\n== Partial Retraining (warm-start) == mode={pr_mode}, last_k={pr_last_k}")
    t_up0 = time.perf_counter()
    pr_model, _ = partial_retrain_warmstart(
        pr_model,
        fts_after=fts_after,
        lbls=lbls_tr,
        retain_mask=retain_mask,
        del_mask=del_mask,
        args=args
    )
    update_time = time.perf_counter() - t_up0

    # ---- eval after PR ----
    # test (use A_te, fts_te)
    pr_model.structure = A_te
    for l in pr_model.layers:
        l.reapproximate = False
    test_acc = evaluate_hgcn_acc(pr_model, {"x": fts_te, "y": lbls_te})
    test_f1 = evaluate_hgcn_f1(pr_model, {"x": fts_te, "y": lbls_te})

    # retain/forget on edited graph (use A_edit, fts_after)
    pr_model.structure = A_edit
    for l in pr_model.layers:
        l.reapproximate = False
    retain_acc, forget_acc = _eval_edited_ret_for(pr_model, fts_after, lbls_tr, retain_mask, del_mask)

    mia_overall = None
    mia_deleted = None
    if bool(getattr(args, "run_mia", False)):
        mm_overall = np.ones(N, dtype=bool)
        mm_overall[deleted.detach().cpu().numpy()] = False
        print("— MIA on PR (overall) —")
        _, _, tgt = membership_inference_hgcn(
            X_train=X_tr,
            y_train=np.asarray(y_tr, dtype=int),
            hyperedges=hedges_edit,
            target_model=pr_model,
            args=args,
            device=device,
            member_mask=mm_overall
        )
        if tgt is not None:
            mia_overall = float(tgt[0])

        mm_del = np.zeros(N, dtype=bool)
        mm_del[deleted.detach().cpu().numpy()] = True
        print("— MIA on PR (deleted-only) —")
        _, _, tgt2 = membership_inference_hgcn(
            X_train=X_tr,
            y_train=np.asarray(y_tr, dtype=int),
            hyperedges=hedges_edit,
            target_model=pr_model,
            args=args,
            device=device,
            member_mask=mm_del
        )
        if tgt2 is not None:
            mia_deleted = float(tgt2[0])

    total_time = edit_time + update_time
    print(f"[PR@EditedHG] edit={edit_time:.4f}s update={update_time:.4f}s total={total_time:.4f}s | "
          f"test_acc={test_acc:.4f} retain_acc={retain_acc:.4f} forget_acc={forget_acc:.4f} | "
          f"mia_overall={mia_overall} mia_deleted={mia_deleted}")

    return {
        "run": run_id,
        "seed": seed,
        "method": f"PR-{pr_mode}@EditedHG",
        "pr_mode": pr_mode,
        "pr_last_k": pr_last_k,
        "deleted": int(len(deleted)),
        "edit_time_sec": float(edit_time),
        "update_time_sec": float(update_time),
        "total_time_sec": float(total_time),
        "test_acc": float(test_acc),
        "test_f1": float(test_f1),
        "retain_acc": float(retain_acc),
        "forget_acc": float(forget_acc),
        "mia_overall_auc": mia_overall,
        "mia_deleted_auc": mia_deleted,
        "full_train_time_sec": float(full_train_time),
    }


def main():
    args = get_args()
    device = _device()
    print(f"[Device] {device}")

    # ---------- add PR args if not in config_HGCN ----------
    _ensure_attr(args, "runs", 3)
    _ensure_attr(args, "pr_mode", "lastk")       # head / lastk / all
    _ensure_attr(args, "pr_last_k", 2)
    _ensure_attr(args, "pr_epochs", 100)
    _ensure_attr(args, "pr_lr", 1e-3)
    _ensure_attr(args, "pr_wd", 0.0)
    _ensure_attr(args, "pr_log_every", 10)
    _ensure_attr(args, "run_mia", False)

    rows = []
    for r in range(int(args.runs)):
        rows.append(run_one(r, args, device))

    df = pd.DataFrame(rows)

    def _ms(col):
        v = df[col].astype(float).values
        return float(v.mean()), float(v.std(ddof=0)) if len(v) > 1 else 0.0

    edit_m, edit_s = _ms("edit_time_sec")
    up_m, up_s = _ms("update_time_sec")
    tot_m, tot_s = _ms("total_time_sec")
    ta_m, ta_s = _ms("test_acc")
    ra_m, ra_s = _ms("retain_acc")
    fa_m, fa_s = _ms("forget_acc")

    print("\n== Summary (mean±std) ==")
    print(f"{df['method'].iloc[0]} | "
          f"edit={edit_m:.4f}±{edit_s:.4f} | update={up_m:.4f}±{up_s:.4f} | total={tot_m:.4f}±{tot_s:.4f} | "
          f"test_acc={ta_m:.4f}±{ta_s:.4f} | retain_acc={ra_m:.4f}±{ra_s:.4f} | forget_acc={fa_m:.4f}±{fa_s:.4f}")

    out_csv = getattr(args, "out_csv_pr", "hgcn_partial_retrain_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n[Saved] {out_csv}")


if __name__ == "__main__":
    main()
