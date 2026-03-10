#!/usr/bin/env python3
# coding: utf-8

import argparse
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from collections import Counter
from bank.HGCN.GIF_HGCN_ROW_bank import approx_gif,train_model,apply_node_deletion_unlearning,find_hyperneighbors

# Bank preprocessing & hypergraph construction
from bank.HGNNP.data_preprocessing_bank import (
    preprocess_node_features_bank,
    generate_hyperedge_dict_bank,
)
# HyperGCN model & Laplacian
from bank.HGCN.HGCN import HyperGCN, laplacian
# GIF-based Unlearning
from bank.HGCN.GIF_HGCN_ROW_bank import approx_gif, train_model
# Evaluation
from bank.HGCN.HGCN_utils import evaluate_hgcn_acc, evaluate_hgcn_f1
# MIA
from bank.HGCN.MIA_HGCN import membership_inference_hgcn
from paths import BANK_DATA

def retrain_after_prune(
    X_tr, y_tr, df_tr, hyperedges_tr,
    deleted: torch.LongTensor,
    transformer, args, device,
    df_te  # Bank test-set DataFrame
):
    """
    Train a new HyperGCN from scratch on the "remaining-node subset",
    and also perform MIA on the retrained model, returning all metrics.
    """
    # 1) Compute node indices to keep — ensure deleted is on the same device
    if isinstance(deleted, torch.Tensor):
        deleted = deleted.to(device)
    all_idx  = torch.arange(X_tr.shape[0], device=device)
    keep_idx = all_idx[~torch.isin(all_idx, deleted)]

    # 2) Prune the original data — use indexing on CPU/NumPy
    keep_idx_cpu = keep_idx.cpu().numpy()        # convert back to CPU NumPy indices first
    X_keep = X_tr[keep_idx_cpu]                 # X_tr is a NumPy array
    print(f"Size of retained-node subset: {X_keep.shape[0]}")

    y_arr   = np.array(y_tr, dtype=int)
    y_keep  = y_arr[keep_idx_cpu]
    df_keep = df_tr.loc[keep_idx_cpu].reset_index(drop=True)
    # 3) Rebuild hypergraph & Laplacian (move directly to GPU)
    cat_cols = args.cat_cols
    cont_cols= args.cont_cols
    hyper_keep_dict = generate_hyperedge_dict_bank(
        df_keep, cat_cols, cont_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        device=device
    )
    hyper_keep_list = list(hyper_keep_dict.values())
    A_keep = laplacian(hyper_keep_list, X_keep, args.mediators).to(device)

    # 4) Prepare tensors and initialize the model
    fts_keep  = torch.from_numpy(X_keep).float().to(device)
    lbls_keep = torch.tensor(y_keep, dtype=torch.long, device=device)
    mask_keep = torch.ones(len(y_keep), dtype=torch.bool, device=device)

    cfg = lambda: None
    cfg.d         = X_keep.shape[1]
    cfg.c         = int(lbls_keep.max().item()) + 1
    cfg.depth     = args.depth
    cfg.hidden    = args.hidden_dim
    cfg.dropout   = args.dropout
    cfg.fast      = args.fast
    cfg.mediators = args.mediators
    cfg.cuda      = True
    cfg.dataset = getattr(args, 'dataset', '')

    model_re = HyperGCN(
        X_keep.shape[0],
        hyper_keep_list,
        X_keep,
        cfg
    ).to(device)
    # Disable layer-wise reapproximation
    model_re.structure = A_keep
    for layer in model_re.layers:
        layer.reapproximate = False

    # 5) Training
    opt  = optim.Adam(model_re.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sch  = optim.lr_scheduler.MultiStepLR(opt, args.milestones, gamma=args.gamma)
    crit = nn.NLLLoss()
    model_re = train_model(
        model_re, crit, opt, sch,
        fts_keep, lbls_keep,
        num_epochs=args.epochs,
        print_freq=args.log_every
    )

    # 6) Subset evaluation
    # Pass feature x instead of lap to match the evaluate_hgcn_* interface
    rem = {"lap": fts_keep, "y": lbls_keep}
    f1_rem = evaluate_hgcn_f1(model_re, rem)
    acc_rem = evaluate_hgcn_acc(model_re, rem)
    print(f"[Subset Evaluation] F1={f1_rem:.4f}, Acc={acc_rem:.4f}")

    # 7) Test-set evaluation
    X_te, y_te, df_te_proc, _ = preprocess_node_features_bank(
        df_te, is_test=True, transformer=transformer
    )
    fts_te  = torch.from_numpy(X_te).float().to(device)
    lbls_te = torch.tensor(y_te, dtype=torch.long, device=device)

    hyper_te_dict = generate_hyperedge_dict_bank(
        df_te_proc, cat_cols, cont_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        device=device
    )
    hyper_te_list = list(hyper_te_dict.values())
    A_te = laplacian(hyper_te_list, X_te, args.mediators).to(device)

    model_re.structure = A_te
    for l in model_re.layers:
        l.reapproximate = False

    # Directly pass feature fts_te to match the evaluate_hgcn_* interface
    test = {"lap": fts_te, "y": lbls_te}
    f1_test = evaluate_hgcn_f1(model_re, test)
    acc_test = evaluate_hgcn_acc(model_re, test)
    print(f"[Test-set Evaluation] F1={f1_test:.4f}, Acc={acc_test:.4f}")

    # ——— 8) Construct the full attack set & perform MIA on the retrained model ———
    # 8.1) Prepare retained/deleted subsets' features/labels/hyperedges
    deleted_idx_cpu = deleted.cpu().numpy()
    X_del = X_tr[deleted_idx_cpu]
    y_del = np.array(y_tr, dtype=int)[deleted_idx_cpu]
    df_del = df_tr.loc[deleted_idx_cpu].reset_index(drop=True)

    hyper_del_dict = generate_hyperedge_dict_bank(
        df_del, cat_cols, cont_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        device=device
    )
    hyper_del_list = list(hyper_del_dict.values())

    # 8.2) Merge retained & deleted hyperedges (deleted node indices need offsetting)
    hyper_attack_list = []
    for he in hyper_keep_list:
        hyper_attack_list.append(he)
    offset = len(X_keep)
    for he in hyper_del_list:
        hyper_attack_list.append([n + offset for n in he])

    # 8.3) Merge features & labels
    X_attack = np.vstack([X_keep, X_del])
    y_attack = np.hstack([y_keep, y_del])

    # 8.4) Rebuild the "attack-use" Laplacian and assign it to model_re
    A_attack = laplacian(hyper_attack_list, X_attack, args.mediators).to(device)
    model_re.structure = A_attack
    for layer in model_re.layers:
        layer.reapproximate = False

    # 8.5) Construct member_mask: retained=True, deleted=False
    member_mask = np.concatenate([
        np.ones(len(X_keep), dtype=bool),
        np.zeros(len(X_del), dtype=bool),
    ])

    # 8.6) Call MIA
    print("— Membership Inference on Retrained Model —")
    _, (_, _), (auc_re, f1_re) = membership_inference_hgcn(
        X_train=X_attack,
        y_train=y_attack,
        hyperedges=hyper_attack_list,
        target_model=model_re,
        args=args,
        device=device,
        member_mask=member_mask
    )
    print(f"[Retrained MIA] AUC={auc_re:.4f}, F1={f1_re:.4f}")

    return {
        'f1_rem':  f1_rem,
        'acc_rem': acc_rem,
        'f1_test': f1_test,
        'acc_test': acc_test,
        'mia_auc': auc_re,
        'mia_f1':  f1_re
    }
def main():
    # parser = argparse.ArgumentParser(description="HyperGCN on the Bank dataset (node-level unlearning)")
    # parser.add_argument(
    #     "--data-csv",
    #     type=str,
    #
    #     help="Path to the Bank Marketing CSV data file (; separated)"
    # )
    parser = argparse.ArgumentParser(description="HyperGCN on the Bank dataset (node-level unlearning)")
    parser.add_argument(
        "--data-csv",
        type=str,
        default=BANK_DATA,
        help="Path to the Bank Marketing CSV data file (; separated)"
    )
    parser.add_argument(
        "--cat-cols",
        nargs="+",
        type=str,
        default=['job','marital','education','default','housing','loan','contact','month','poutcome'],
        help="List of categorical feature column names"
    )
    parser.add_argument(
        "--cont-cols",
        nargs="+",
        type=str,
        default=['age','balance','day','duration','campaign','pdays','previous'],
        help="List of continuous feature column names"
    )
    parser.add_argument("--split-ratio",    type=float, default=0.2, help="Train/test split ratio")
    parser.add_argument("--max-nodes-per-hyperedge", type=int, default=50, help="Maximum number of nodes per hyperedge")
    parser.add_argument("--depth",        type=int,   default=3,    help="HyperGCN depth")
    parser.add_argument("--hidden-dim",   type=int,   default=128,  help="Hidden dimension")
    parser.add_argument("--dropout",      type=float, default=0.01, help="Dropout")
    parser.add_argument("--fast",         action="store_true",     help="Fast mode")
    parser.add_argument("--mediators",    action="store_true",     help="Use Mediators")
    parser.add_argument("--lr",           type=float, default=0.01, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.001,help="Weight decay")
    parser.add_argument("--milestones",   nargs="+", type=int, default=[100,150], help="Learning rate milestones")
    parser.add_argument("--gamma",        type=float, default=0.1,  help="LR decay factor")
    parser.add_argument("--epochs",       type=int,   default=120,  help="Number of training epochs")
    parser.add_argument("--log-every",    type=int,   default=10,   help="Logging frequency")
    parser.add_argument("--gif-iters",    type=int,   default=80,   help="GIF iterations")
    parser.add_argument("--gif-damp",     type=float, default=0.01, help="GIF damping")
    parser.add_argument("--gif-scale",    type=float, default=1e7,  help="GIF scale")
    parser.add_argument("--remove-ratio", type=float, default=0.3,  help="Ratio of randomly deleted nodes")
    parser.add_argument("--neighbor-k",   type=int,   default=12,   help="Hyperedge-sharing threshold K")
    parser.add_argument("--test-csv",     type=str,   default=None, help="Optional independent test-set CSV")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # 1) Read CSV & split
    df_full = pd.read_csv(args.data_csv, sep=';', skipinitialspace=True)
    df_tr, df_te = train_test_split(
        df_full, test_size=args.split_ratio,
        stratify=df_full["y"], random_state=42
    )
    df_tr, df_te = df_tr.reset_index(drop=True), df_te.reset_index(drop=True)
    print(f"Train: {len(df_tr)}, Test: {len(df_te)}")

    # 2) Preprocessing
    X_tr, y_tr, df_tr_proc, transformer = preprocess_node_features_bank(df_tr, is_test=False)
    X_te, y_te, df_te_proc, _          = preprocess_node_features_bank(df_te, is_test=True, transformer=transformer)

    # 3) Build original hyperedge list
    hyperedges_tr = list(generate_hyperedge_dict_bank(
        df_tr_proc, args.cat_cols, args.cont_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        device=device
    ).values())

    # 4) Full-model initialization & training
    # 4.1 Convert to tensors
    fts  = torch.from_numpy(X_tr).float().to(device)
    lbls = torch.tensor(y_tr, dtype=torch.long, device=device)
    # 4.2 Build cfg_full
    cfg_full = lambda: None
    cfg_full.d         = X_tr.shape[1]
    cfg_full.c         = int(max(y_tr)) + 1
    cfg_full.depth     = args.depth
    cfg_full.hidden    = args.hidden_dim
    cfg_full.dropout   = args.dropout
    cfg_full.fast      = args.fast
    cfg_full.mediators = args.mediators
    cfg_full.cuda      = True
    cfg_full.dataset   = getattr(args, 'dataset', '')
    # 4.3 Initialize HyperGCN and fix the Laplacian
    model = HyperGCN(
        X_tr.shape[0],
        hyperedges_tr,
        X_tr,
        cfg_full
    ).to(device)
    A_before = laplacian(hyperedges_tr, X_tr, args.mediators).to(device)
    model.structure = A_before
    for layer in model.layers:
        layer.reapproximate = False
    # 4.4 Train the full model
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    criterion = nn.NLLLoss()
    model = train_model(
        model, criterion, optimizer, scheduler,
        fts, lbls,
        num_epochs=args.epochs,
        print_freq=args.log_every
    )
    # 4.5 Prepare data_train for GIF
    data_train = {"lap": fts, "y": lbls}

    # 5) Delete nodes by value: delete samples whose job belongs to ['blue-collar','technician']
    # mask = df_tr_proc['job'].isin(['blue-collar', 'technician'])
    mask = (df_tr['marital'] == 'married') & (df_tr['housing'] == 'yes')
    deleted_idx = np.where(mask.values)[0]
    deleted = torch.tensor(deleted_idx, dtype=torch.long, device=device)
    pct = 100.0 * len(deleted_idx) / len(y_tr)
    print(f"Delete by value: {len(deleted_idx)}/{len(y_tr)} nodes in total (≈{pct:.1f}%), first 5 indices: {deleted_idx[:5]}")

    # 6) Retraining + MIA baseline
    baseline = retrain_after_prune(
        X_tr, y_tr, df_tr_proc, hyperedges_tr,
        deleted, transformer, args, device,
        df_te_proc
    )
    print("[Baseline Retraining]")
    print(f" Remaining subset  F1/Acc = {baseline['f1_rem']:.4f}/{baseline['acc_rem']:.4f}")
    print(f" Test set          F1/Acc = {baseline['f1_test']:.4f}/{baseline['acc_test']:.4f}")

    # 7.1.1) Compute neighbors of the deleted nodes
    deleted_list = deleted.cpu().tolist()
    K = getattr(args, "neighbor_k", 12)
    deleted_neighbors = find_hyperneighbors(
        hyperedges_tr,
        deleted_list,
        K
    )
    print(f"Found {len(deleted_neighbors)} neighbors sharing ≥{K} hyperedges")

    # 7.2) Restore the original training structure & evaluate before deletion
    model.structure = A_before
    for layer in model.layers:
        layer.reapproximate = False
    mask_del = torch.zeros_like(lbls, dtype=torch.bool, device=device)
    mask_del[deleted] = True
    data_del_before = {"lap": fts, "y": lbls, "train_mask": mask_del}
    print(f" Deleted-nodes F1 before: {evaluate_hgcn_f1(model, data_del_before):.4f}, "
          f"Acc: {evaluate_hgcn_acc(model, data_del_before):.4f}")

    # 7.3) Delete nodes → obtain zeroed features & new Laplacian
    fts_new, edge_list_new, A_after = apply_node_deletion_unlearning(
        fts,
        edge_list=hyperedges_tr,
        deleted_nodes=deleted,
        mediators=args.mediators,
        device=device
    )
    print("Check whether the features of deleted nodes are all zero:", (fts_new[deleted] != 0).sum().item())
    print(f" Deleted {len(deleted)} nodes; {len(edge_list_new)} hyperedges remain")

    # 7.4) GIF update (compensating both deleted nodes and their neighbors)
    gif_time = approx_gif(
        model, data_train,
        A_before, A_after,
        deleted_nodes=deleted,
        deleted_neighbors=deleted_neighbors,
        x_before=fts,
        x_after=fts_new,
        iters=args.gif_iters,
        damp=args.gif_damp,
        scale=args.gif_scale
    )
    print(f" GIF time: {gif_time:.4f}s")
    data_del_after = {"lap": fts_new, "y": lbls, "train_mask": mask_del}
    print(f" Deleted-nodes F1 after: {evaluate_hgcn_f1(model, data_del_after):.4f}, "
          f"Acc: {evaluate_hgcn_acc(model, data_del_after):.4f}")

    # 8) Restore the test structure & final evaluation
    model.structure = laplacian(
        list(generate_hyperedge_dict_bank(
            df_te_proc, args.cat_cols, args.cont_cols,
            max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
            device=device
        ).values()),
        X_te,
        args.mediators
    ).to(device)
    for layer in model.layers:
        layer.reapproximate = False
    data_test = {"lap": torch.from_numpy(X_te).float().to(device), "y": torch.tensor(y_te, dtype=torch.long, device=device)}
    print("— After Unlearning —")
    print(f" Test F1: {evaluate_hgcn_f1(model, data_test):.4f}, "
          f"Acc: {evaluate_hgcn_acc(model, data_test):.4f}")

    # 9) MIA on Unlearned Model
    model.structure = A_before
    for layer in model.layers:
        layer.reapproximate = False
    member_mask = np.ones(len(y_tr), dtype=bool)
    member_mask[deleted.cpu().numpy()] = False
    print("— Membership Inference Attack on Unlearned Model —")
    attack_model, _, (auc_un, f1_un) = membership_inference_hgcn(
        X_train=X_tr,
        y_train=y_tr,
        hyperedges=hyperedges_tr,
        target_model=model,
        args=args,
        device=device,
        member_mask=member_mask
    )
    print(f" Unlearned Model MIA  AUC = {auc_un:.4f},  F1 = {f1_un:.4f}")

if __name__ == "__main__":
    # main()
    for run in range(1,21):
        print("=== Run",run,"===")
        main()