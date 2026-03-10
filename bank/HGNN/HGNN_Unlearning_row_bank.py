#!/usr/bin/env python
# coding: utf-8
"""
train_unlearning_hgnn_refactored.py

Refactored version: separates retraining on a pruned dataset and GIF-based unlearning
into two clear functions, using the same deleted node indices for both parts.
"""
import pandas as pd

from collections import Counter
import argparse                      # MOD: add argparse to support command-line arguments

from sklearn.model_selection import train_test_split

from torch_geometric.data import Data

import torch.nn as nn
import torch.optim as optim
import time
import torch.nn.functional as F

from bank.HGNN.HGNN_utils import evaluate_test_acc, evaluate_test_f1
from bank.HGNN.data_preprocessing_bank import (
    preprocess_node_features_bank,
    generate_hyperedge_dict_bank,
)
from bank.HGNN.HGNN import HGNN_implicit, build_incidence_matrix, compute_degree_vectors
from bank.HGNN.GIF_HGNN_ROW import (
    rebuild_structure_after_node_deletion,
    approx_gif,
    train_model,
)

import numpy as np
import torch
from MIA_HGNN        import train_shadow_model, membership_inference
from paths import BANK_DATA

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
def mia_with_hgnn(X_train, y_train, hyperedges, args, device):
    """
    1) Construct the data structure Data_full for the full model
    2) Train the full model
    3) Call membership_inference(...), passing in the already trained full_model
    4) Return (AUC, F1) on the full model
    """
    N = X_train.shape[0]

    # 1) Construct hypergraph and degree inverses
    H_sp        = build_incidence_matrix(hyperedges, N)
    dv_inv, de_inv = compute_degree_vectors(H_sp)
    Hc = H_sp.tocoo()
    idx = torch.LongTensor([Hc.row, Hc.col])
    val = torch.FloatTensor(Hc.data)
    H_tensor = torch.sparse_coo_tensor(idx, val, Hc.shape).to(device)

    # 2) Prepare Data_full, with all nodes used for training
    full_train_mask = torch.ones(N, dtype=torch.bool, device=device)
    data_full = Data(
        x = torch.from_numpy(X_train).float().to(device),
        y = torch.from_numpy(y_train).long().to(device),
        H         = H_tensor,
        dv_inv    = torch.from_numpy(dv_inv).to(device),
        de_inv    = torch.from_numpy(de_inv).to(device),
        train_mask    = full_train_mask,
        train_indices = list(range(N)),
        test_indices  = []               # test_indices are not needed for Full model training
    )

    # 3) Initialize and train the Full model
    full_model = HGNN_implicit(
        in_ch   = X_train.shape[1],
        n_class = int(y_train.max()) + 1,
        n_hid   = args.full_hidden,
        dropout = args.full_dropout
    ).to(device)

    full_model = train_shadow_model(
        full_model, data_full,
        lr     = args.full_lr,
        epochs = args.full_epochs
    )

    # 4) Use membership_inference to compute AUC/F1 on the Full model
    #    Return format: (atk_model, (auc_shadow,f1_shadow), (auc_target,f1_target))
    _, (_, _), (auc_target, f1_target) = membership_inference(
        X_train = X_train,
        y_train = y_train,
        hyperedges   = hyperedges,
        target_model = full_model,
        args         = args,
        device       = device
    )
    return auc_target, f1_target

    # # ——— Column name definitions ———
    # col_names = [
    #     "age", "job", "marital", "education", "default",
    #     "balance", "housing", "loan", "contact", "day",
    #     "month", "duration", "campaign", "pdays",
    #     "previous", "poutcome", "y"
    # ]

def retrain_on_pruned(X, y, df, transformer, deleted_idx, args, device):
    """
    Prunes training data by removing nodes at deleted_idx,
    retrains a HGNN model from scratch, and evaluates on the test set.
    Now additionally outputs:
      - Total number of training nodes after pruning
      - Hypergraph reconstruction time & total number of hyperedges
      - Training time & total elapsed time
      - Total number of hyperedges in the test set
      - Test-set F1 & Acc
      1) Delete the nodes specified by deleted_idx and retrain HGNN;
    2) Perform one evaluation on the full test set (F1/Acc);
    3) On the deleted nodes, use orig_model and retrained_model to make predictions respectively,
       then use the F1/acc difference to measure “forgetting”.
    """
    # —— Start of original code replacement ——
    num_train = X.shape[0]
    keep_mask = np.ones(num_train, dtype=bool)
    keep_mask[deleted_idx] = False

    # —— Modified: consistently convert y to NumPy —— #
    y = np.asarray(y)

    # 1) Data pruning
    X_pruned  = X[keep_mask]
    y_pruned  = y[keep_mask]
    # Use loc + boolean mask to preserve row alignment
    df_pruned = df.loc[keep_mask].reset_index(drop=True)

    print(f"[Retraining] Number of training nodes after pruning: {X_pruned.shape[0]} / original {num_train}")

    # Start total time measurement
    t_start = time.time()

    # 2) Hypergraph reconstruction
    cat_cols = [
                "job", "marital", "education", "default",
                "housing", "loan", "contact", "month", "poutcome"
    ]
    cont_cols = ['age','balance','day','duration','campaign','pdays','previous']

    t0 = time.time()
    hyperedges_pruned = generate_hyperedge_dict_bank(
        df_pruned,
        cat_cols,cont_cols,
        max_nodes_per_hyperedge=getattr(args, 'max_nodes_per_hyperedge_train', 100),
        device=device,
    )
    t_hg = time.time() - t0
    print(f"[Retraining] Hypergraph reconstruction time: {t_hg:.2f}s")
    print(f"[Retraining] Total number of hyperedges after pruning: {len(hyperedges_pruned)}")

    # Build incidence matrix and degree vectors
    H_sp = build_incidence_matrix(hyperedges_pruned, X_pruned.shape[0])
    dv_np, de_np = compute_degree_vectors(H_sp)

    # Convert to torch sparse format
    H_coo = H_sp.tocoo()
    idx = torch.LongTensor(np.vstack((H_coo.row, H_coo.col))).to(device)
    val = torch.FloatTensor(H_coo.data).to(device)
    H_pruned = torch.sparse_coo_tensor(idx, val, size=H_coo.shape).coalesce()
    dv = torch.FloatTensor(dv_np).to(device)
    de = torch.FloatTensor(de_np).to(device)
    fts = torch.FloatTensor(X_pruned).to(device)
    lbls = torch.LongTensor(y_pruned).to(device)

    # 3) Model training
    t_init_start = time.time()
    model = HGNN_implicit(
        in_ch=fts.shape[1],
        n_class=len(np.unique(y)),
        n_hid=getattr(args, 'hidden_dim', 128),
        dropout=getattr(args, 'dropout', 0.1)
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=getattr(args, 'lr', 0.01), weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=getattr(args, 'milestones', [100,150]),
        gamma=getattr(args, 'gamma', 0.1)
    )
    criterion = FocalLoss(gamma=2.0, reduction='mean')
    t_init_end = time.time()
    t_train_start = time.time()
    model = train_model(
        model, criterion, optimizer, scheduler,
        fts, lbls, H_pruned, dv, de,
        num_epochs=getattr(args, 'epochs', 200), print_freq=10
    )
    t_train_end = time.time()
    t_total = time.time() - t_start
     # —— Output all stage timings in one line ——
    print(
        f"[Retraining] Initialization: {t_init_end - t_init_start:.2f}s | "
        f"Training: {t_train_end - t_train_start:.2f}s | "
        f"Total: {t_total:.2f}s"
    )
    # 4) Test-set evaluation
    # 4) Test-set evaluation (using the same data_csv path passed from main)
    test_csv = args.data_csv
    X_test, y_test, df_test, _ = preprocess_node_features_bank(
        test_csv, is_test=True, transformer=transformer
    )
    t2 = time.time()
    hyperedges_test = generate_hyperedge_dict_bank(
        df_test, cat_cols,cont_cols,
        max_nodes_per_hyperedge=getattr(args, 'max_nodes_per_hyperedge', 50),
        device=device
    )
    t_hg_te = time.time() - t2
    print(f"[Retraining] Test-set hypergraph reconstruction time: {t_hg_te:.2f}s")
    print(f"[Retraining] Total number of hyperedges in the test set: {len(hyperedges_test)}")

    H_sp_te = build_incidence_matrix(hyperedges_test, X_test.shape[0])
    dv_te, de_te = compute_degree_vectors(H_sp_te)
    H_coo_te = H_sp_te.tocoo()
    idx_te = torch.LongTensor(np.vstack((H_coo_te.row, H_coo_te.col))).to(device)
    val_te = torch.FloatTensor(H_coo_te.data).to(device)
    H_test = torch.sparse_coo_tensor(idx_te, val_te, size=H_coo_te.shape).coalesce()
    dv_test = torch.FloatTensor(dv_te).to(device)
    de_test = torch.FloatTensor(de_te).to(device)
    fts_test = torch.FloatTensor(X_test).to(device)
    lbls_test = torch.LongTensor(y_test).to(device)

    test_obj = {"x": fts_test, "y": lbls_test, "H": H_test, "dv_inv": dv_test, "de_inv": de_test}
    # Evaluate and handle the case where a tuple is returned
    f1_res = evaluate_test_f1(model, test_obj)
    acc    = evaluate_test_acc(model, test_obj)
    # If f1_res is a tuple, take the first element; otherwise use it directly
    f1_val = f1_res[0] if isinstance(f1_res, tuple) else f1_res
    print(f"[Retraining] Test-set F1: {f1_val:.4f}, Acc: {acc:.4f}")

    # (3) Model training & test-set evaluation and printing omitted…

    # —— New: evaluation of the retrained model on the deleted nodes —— #
    # X, y, df, deleted_idx are all available in the function scope
    # 1) Prepare the subset of deleted nodes
    X_del = X[deleted_idx]
    y_del = y[deleted_idx]
    df_del = df.iloc[deleted_idx].reset_index(drop=True)

    # 2) Rebuild hypergraph
    hyperedges_del = generate_hyperedge_dict_bank(
        df_del, cat_cols,cont_cols,
        max_nodes_per_hyperedge=getattr(args, 'max_nodes_per_hyperedge_train', 10),
        device=device
    )
    H_sp_del = build_incidence_matrix(hyperedges_del, len(deleted_idx))
    dv_np_del, de_np_del = compute_degree_vectors(H_sp_del)
    H_coo_del = H_sp_del.tocoo()
    idx_del = torch.LongTensor(np.vstack((H_coo_del.row, H_coo_del.col))).to(device)
    val_del = torch.FloatTensor(H_coo_del.data).to(device)
    H_del = torch.sparse_coo_tensor(idx_del, val_del, size=H_coo_del.shape).coalesce()

    # 3) Convert to model input
    fts_del = torch.FloatTensor(X_del).to(device)
    lbls_del = torch.LongTensor(y_del).to(device)
    dv_del = torch.FloatTensor(dv_np_del).to(device)
    de_del = torch.FloatTensor(de_np_del).to(device)
    test_obj_del = {
        "x": fts_del,
        "y": lbls_del,
        "H": H_del,
        "dv_inv": dv_del,
        "de_inv": de_del
    }
    # 1) Determine whether the number of hyperedges in the deleted set is greater than 0
    num_hyperedges = len(hyperedges_del)

    # 2) Determine whether the node features in the deleted set are not all zeros
    #    Here fts_del is assumed to be a torch.Tensor
    has_nonzero_features = (fts_del.abs().sum().item() > 0)

    # 3) Output the result
    print(f"Number of hyperedges in the deleted set {num_hyperedges}, "
          f"deleted-set node features were not zeroed out? {has_nonzero_features}")

    # 4) Use the retrained model to make predictions
    model.eval()
    with torch.no_grad():
        # Directly call your evaluation functions
        f1_del = evaluate_test_f1(model, test_obj_del)
        acc_del = evaluate_test_acc(model, test_obj_del)

    print(f"[Deleted-node Evaluation] Scores of the retrained model on the deleted nodes:: F1={f1_del:.4f}, Acc={acc_del:.4f}")

    # ——— New: perform MIA evaluation on the retrained model ———
    # Make sure membership_inference and generate_hyperedge_dict have already been imported
    # 1) Construct keep / del attack set
    all_idx = np.arange(num_train)
    keep_idx = np.setdiff1d(all_idx, deleted_idx)
    X_keep, y_keep = X[keep_idx], y[keep_idx]
    X_del, y_del = X[deleted_idx], y[deleted_idx]

    # 2) Rebuild keep hyperedges (already using hyperedges_pruned) and del hyperedges
    he_keep = hyperedges_pruned
    df_del = df.iloc[deleted_idx].reset_index(drop=True)
    he_del = generate_hyperedge_dict_bank(
        df_del, cat_cols,cont_cols,
        max_nodes_per_hyperedge=getattr(args, 'max_nodes_per_hyperedge_train', 10),
        device=device
    )

    # 3) Merge hyperedges (del needs an offset added)
    he_attack = {}
    cnt = 0
    for nodes in he_keep.values():
        he_attack[cnt] = nodes
        cnt += 1
    offset = len(X_keep)
    for nodes in he_del.values():
        he_attack[cnt] = [n + offset for n in nodes]
        cnt += 1

    # 4) Construct attack set X_attack/y_attack and member_mask
    X_attack = np.vstack([X_keep, X_del])
    y_attack = np.hstack([y_keep, y_del])
    member_mask = np.concatenate([
        np.ones(len(X_keep), dtype=bool),  # remaining nodes as positive examples
        np.zeros(len(X_del), dtype=bool)  # deleted nodes as negative examples
    ])

    # 5) Call membership_inference
    _, _, (auc_rt, f1_rt) = membership_inference(
        X_train=X_attack,
        y_train=y_attack,
        hyperedges=he_attack,
        target_model=model,
        args=args,
        device=device,
        member_mask=member_mask
    )
    print(f"[MIA-Retrain Keep/Del] AUC={auc_rt:.4f}, F1={f1_rt:.4f}")
    # ———————————————————————————————

    return model

def full_train_and_unlearn(X, y, df, transformer, deleted_idx, args, device):
    """
    Trains on the full dataset, evaluates before unlearning,
    performs GIF-based unlearning on deleted_idx, then evaluates again.
    """

    # Build full hypergraph
    cat_cols = [
                "job", "marital", "education", "default",
                "housing", "loan", "contact", "month", "poutcome"
    ]
    cont_cols = ['age','balance','day','duration','campaign','pdays','previous']
    hyperedges = generate_hyperedge_dict_bank(
        df, cat_cols,cont_cols,
        max_nodes_per_hyperedge=getattr(args, 'max_nodes_per_hyperedge_train', 50),
        device=device
    )
    H_sp = build_incidence_matrix(hyperedges, X.shape[0])
    dv_np, de_np = compute_degree_vectors(H_sp)

    # Convert to torch sparse + tensors
    H_coo = H_sp.tocoo()
    idx = torch.LongTensor(np.vstack((H_coo.row, H_coo.col))).to(device)
    val = torch.FloatTensor(H_coo.data).to(device)
    H = torch.sparse_coo_tensor(idx, val, size=H_coo.shape).coalesce()
    dv = torch.FloatTensor(dv_np).to(device)
    de = torch.FloatTensor(de_np).to(device)
    fts = torch.FloatTensor(X).to(device)
    lbls = torch.LongTensor(y).to(device)

    # Prepare train_mask for GIF
    train_mask = torch.ones(X.shape[0], dtype=torch.bool, device=device)

    # Train full model
    model = HGNN_implicit(
        in_ch=fts.shape[1],
        n_class=len(np.unique(y)),
        n_hid=getattr(args, 'hidden_dim', 128),
        dropout=getattr(args, 'dropout', 0.1)
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=getattr(args, 'lr', 0.01), weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=getattr(args, 'milestones', [100,150]),
        gamma=getattr(args, 'gamma', 0.1)
    )
    # ← Start inserting here
    # —— MOD: attach the reason_once methods required by GIF to the model ——
    def reason_once(data):
        return model(
            data["x"],
            data["H"],
            data["dv_inv"],
            data["de_inv"]
        )

    def reason_once_unlearn(data):
        # If you need to zero out data["x"] or rebuild the structure, modify it here
        return model(
            data["x"],
            data["H"],
            data["dv_inv"],
            data["de_inv"]
        )

    model.reason_once         = reason_once        # MOD
    model.reason_once_unlearn = reason_once_unlearn  # MOD
    # ← Insertion ends
    # criterion = nn.CrossEntropyLoss()
    criterion = FocalLoss(gamma=2.0
                          # , weight=weights   # Uncomment if you want to add class weights
                          , reduction='mean')

    model = train_model(
        model, criterion, optimizer, scheduler,
        fts, lbls, H, dv, de,
        num_epochs=getattr(args, 'epochs', 200), print_freq=10
    )

    # Prepare test data
    # 4) Test-set evaluation (using the same data_csv path passed from main)
    test_csv = args.data_csv
    X_test, y_test, df_test, _ = preprocess_node_features_bank(
        test_csv, is_test=True, transformer=transformer
    )
    hyperedges_test = generate_hyperedge_dict_bank(
        df_test, cat_cols,cont_cols,
        max_nodes_per_hyperedge=getattr(args, 'max_nodes_per_hyperedge', 50),
        device=device
    )

    # Count the numbers of positive and negative samples in the test set
    test_counter = Counter(y_test)
    num_neg_test = test_counter[0]
    num_pos_test = test_counter[1]
    print(f"Test set → <=50K: {num_neg_test}, >50K: {num_pos_test}")

    H_sp_te = build_incidence_matrix(hyperedges_test, X_test.shape[0])
    dv_te, de_te = compute_degree_vectors(H_sp_te)
    H_coo_te = H_sp_te.tocoo()
    idx_te = torch.LongTensor(np.vstack((H_coo_te.row, H_coo_te.col))).to(device)
    val_te = torch.FloatTensor(H_coo_te.data).to(device)
    H_test = torch.sparse_coo_tensor(idx_te, val_te, size=H_coo_te.shape).coalesce()
    dv_test = torch.FloatTensor(dv_te).to(device)
    de_test = torch.FloatTensor(de_te).to(device)
    fts_test = torch.FloatTensor(X_test).to(device)
    lbls_test = torch.LongTensor(y_test).to(device)
    test_obj = {"x": fts_test, "y": lbls_test, "H": H_test, "dv_inv": dv_test, "de_inv": de_test}

    # Evaluate before unlearning
    f1_before = evaluate_test_f1(model, test_obj)
    acc_before = evaluate_test_acc(model, test_obj)
    print(f"Before GIF unlearning: F1={f1_before:.4f}, Acc={acc_before:.4f}")

    # Build data_obj for GIF
    data_obj = {"x": fts, "y": lbls, "H": H, "dv_inv": dv, "de_inv": de, "train_mask": train_mask}

    # —— New: calculate the number of misclassifications on the test set —— #
    with torch.no_grad():
        # test_obj already contains fts_test, lbls_test, H_test, dv_test, de_test
        logits_after = model(
            test_obj["x"],
            test_obj["H"],
            test_obj["dv_inv"],
            test_obj["de_inv"]
        )
        preds_after = logits_after.argmax(dim=1)  # predicted class
        mismatches = (preds_after != test_obj["y"])  # boolean mask
        num_errors = mismatches.sum().item()  # number of errors
        total = test_obj["y"].shape[0]  # test set size

    print(f"[After Unlearning] Test errors: {num_errors}/{total}  (err_rate={num_errors / total:.4f})")

    # Attach deletion logic
    def reason_once(data):
        return model(data["x"], data["H"], data["dv_inv"], data["de_inv"])

    def reason_once_unlearn(data):
        x_mod = data["x"].clone()
        x_mod[deleted_idx] = 0.0
        H_new, dv_new, de_new, _ = rebuild_structure_after_node_deletion(
            hyperedges, deleted_idx, X.shape[0], device
        )
        return model(x_mod, H_new, dv_new, de_new)

    model.reason_once = reason_once
    model.reason_once_unlearn = reason_once_unlearn

    # Perform GIF-based unlearning
    unlearning_time, _ = approx_gif(
        model,
        data_obj,
        (deleted_idx, hyperedges, 13),   # <<< pass hyperedges instead of None
        iteration=getattr(args, 'gif_iters', 20),
        damp=getattr(args, 'gif_damp', 0.01),
        scale=getattr(args, 'gif_scale', 1e7)
    )
    print(f"[Unlearn] Took {unlearning_time:.4f}s")

    # Evaluate after unlearning
    f1_after = evaluate_test_f1(model, test_obj)
    acc_after = evaluate_test_acc(model, test_obj)
    print(f"After unlearning:  F1={f1_after:.4f}, Acc={acc_after:.4f}")

    # —— New: calculate the number of misclassifications on the test set —— #
    with torch.no_grad():
        # test_obj already contains fts_test, lbls_test, H_test, dv_test, de_test
        logits_after = model(
            test_obj["x"],
            test_obj["H"],
            test_obj["dv_inv"],
            test_obj["de_inv"]
        )
        preds_after = logits_after.argmax(dim=1)  # predicted class
        mismatches = (preds_after != test_obj["y"])  # boolean mask
        num_errors = mismatches.sum().item()  # number of errors
        total = test_obj["y"].shape[0]  # test set size

    y = np.asarray(y)
    # —— New: evaluation of the model on the deleted nodes after retraining —— #
    # X, y, df, deleted_idx are all available in the function scope
    # 1) Prepare the subset of deleted nodes
    X_del = X[deleted_idx]
    y_del = y[deleted_idx]
    df_del = df.iloc[deleted_idx].reset_index(drop=True)

    # 2) Rebuild hypergraph
    hyperedges_del = generate_hyperedge_dict_bank(
        df_del, cat_cols,cont_cols,
        max_nodes_per_hyperedge=getattr(args, 'max_nodes_per_hyperedge_train', 50),
        device=device
    )
    H_sp_del = build_incidence_matrix(hyperedges_del, len(deleted_idx))
    dv_np_del, de_np_del = compute_degree_vectors(H_sp_del)
    H_coo_del = H_sp_del.tocoo()
    idx_del = torch.LongTensor(np.vstack((H_coo_del.row, H_coo_del.col))).to(device)
    val_del = torch.FloatTensor(H_coo_del.data).to(device)
    H_del = torch.sparse_coo_tensor(idx_del, val_del, size=H_coo_del.shape).coalesce()

    # 3) Convert to model input
    fts_del = torch.FloatTensor(X_del).to(device)
    lbls_del = torch.LongTensor(y_del).to(device)
    dv_del = torch.FloatTensor(dv_np_del).to(device)
    de_del = torch.FloatTensor(de_np_del).to(device)
    test_obj_del = {
        "x": fts_del,
        "y": lbls_del,
        "H": H_del,
        "dv_inv": dv_del,
        "de_inv": de_del
    }

    # 1) Determine whether the number of hyperedges in the deleted set is greater than 0
    num_hyperedges = len(hyperedges_del)

    # 2) Determine whether the node features in the deleted set are not all zeros
    #    Here fts_del is assumed to be a torch.Tensor
    has_nonzero_features = (fts_del.abs().sum().item() > 0)

    # 3) Output the result
    print(f"Number of hyperedges in the deleted set  {num_hyperedges}, "
          f"deleted-set node features were not zeroed out? {has_nonzero_features}")
    # 4) Use the model after retraining to make predictions
    model.eval()
    with torch.no_grad():
        # Directly call your evaluation functions
        f1_del = evaluate_test_f1(model, test_obj_del)
        acc_del = evaluate_test_acc(model, test_obj_del)

    print(f"[Deleted-node Evaluation] HGIF scores on the deleted nodes: F1={f1_del:.4f}, Acc={acc_del:.4f}")

    # ——— MIA-On-Unlearned ———
    # 1) Split keep / del

    all_idx  = np.arange(X.shape[0])
    keep_idx = np.setdiff1d(all_idx, deleted_idx)
    X_keep, y_keep = X[keep_idx], y[keep_idx]
    X_del,  y_del  = X[deleted_idx], y[deleted_idx]

    # 2) Rebuild keep / del hyperedges
    df_keep = df.drop(index=deleted_idx).reset_index(drop=True)
    df_del  = df.iloc[deleted_idx].reset_index(drop=True)
    he_keep = generate_hyperedge_dict_bank(
        df_keep, cat_cols,cont_cols,
        max_nodes_per_hyperedge=getattr(args, 'max_nodes_per_hyperedge_train', 50),
        device=device
    )
    he_del  = generate_hyperedge_dict_bank(
        df_del, cat_cols,cont_cols,
        max_nodes_per_hyperedge=getattr(args, 'max_nodes_per_hyperedge_train', 50),
        device=device
    )

    # 3) Merge hyperedges (node indices in the del part need to be offset)
    he_attack = {}
    idx_he = 0
    for nodes in he_keep.values():
        he_attack[idx_he] = nodes
        idx_he += 1
    offset = len(X_keep)
    for nodes in he_del.values():
        he_attack[idx_he] = [n + offset for n in nodes]
        idx_he += 1

    # 4) Construct attack set & member_mask
    X_attack = np.vstack([X_keep, X_del])
    y_attack = np.hstack([y_keep, y_del])
    member_mask = np.concatenate([
        np.ones(len(X_keep), dtype=bool),   # keep nodes as positive examples
        np.zeros(len(X_del),  dtype=bool)   # del nodes as negative examples
    ])

    # 5) Call membership_inference
    _, _, (auc_un, f1_un) = membership_inference(
        X_train      = X_attack,
        y_train      = y_attack,
        hyperedges   = he_attack,
        target_model = model,
        args         = args,
        device       = device,
        member_mask  = member_mask
    )
    print(f"[MIA-Unlearned Keep/Del] AUC={auc_un:.4f}, F1={f1_un:.4f}")
    # ————————————————————

    return model

def main():
    parser = argparse.ArgumentParser(description="HGNN on the Bank dataset")
    # parser.add_argument("--data-csv", type=str,
    #                     help="Path to the full Bank dataset CSV file")
    parser.add_argument(
        "--data-csv",
        type=str,
        default=BANK_DATA,

        help="Path to the Bank Marketing CSV file (; separated)"
    )
    parser.add_argument("--split-ratio", type=float, default=0.2,
                        help="Test set ratio")
    parser.add_argument("--max-nodes-per-hyperedge", type=int, default=50,
                        help="Maximum number of nodes per hyperedge")
    parser.add_argument("--hidden-dim", type=int, default=128, help="HGNN hidden dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout ratio")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--milestones", nargs='+', type=int, default=[100,150], help="Learning rate milestones")
    parser.add_argument("--gamma", type=float, default=0.1, help="Learning rate decay factor")
    parser.add_argument("--gif-iters", type=int, default=20, help="Number of GIF iterations")
    parser.add_argument("--gif-damp", type=float, default=0.01, help="GIF damping")
    parser.add_argument("--gif-scale", type=float, default=1e7, help="GIF scale")
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {device}")

    # 1) Read and split the original DataFrame
    df_full = pd.read_csv(args.data_csv, sep=';', skipinitialspace=True)
    df_tr, df_te = train_test_split(df_full,
                                     test_size=args.split_ratio,
                                     random_state=42,
                                     stratify=df_full['y'])
    df_tr.reset_index(drop=True, inplace=True)
    df_te.reset_index(drop=True, inplace=True)
    print(f"Training set: {len(df_tr)} samples, Test set: {len(df_te)} samples")

    # 2) Perform feature preprocessing on the training set to obtain X_tr, y_tr, and transformer
    X_tr, y_tr, df_tr_proc, transformer = preprocess_node_features_bank(df_tr, is_test=False)

    # 3) Randomly select the node indices to delete
    num_to_delete = int(0.1 * X_tr.shape[0])
    deleted_idx = np.random.choice(X_tr.shape[0], num_to_delete, replace=False)

    # 4) Retraining on pruned dataset
    print("=== Retraining on pruned dataset ===")
    retrained_model = retrain_on_pruned(
        X            = X_tr,
        y            = y_tr,
        df           = df_tr,         # pass the original df_tr to preserve categorical columns
        transformer  = transformer,
        deleted_idx  = deleted_idx,
        args         = args,
        device       = device
    )

    # 5) Full train + GIF Unlearning
    print("\n=== Full training + GIF Unlearning ===")
    _ = full_train_and_unlearn(
        X            = X_tr,
        y            = y_tr,
        df           = df_tr,         # likewise pass the original df_tr
        transformer  = transformer,
        deleted_idx  = deleted_idx,
        args         = args,
        device       = device
    )

if __name__ == "__main__":
    # main()
    for run in range(1,4):
        print("=== Run",run,"===")
        main()