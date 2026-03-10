#!/usr/bin/env python
# coding: utf-8
"""
train_unlearning_hgnn_refactored.py

Refactored version: separates retraining on a pruned dataset and GIF-based unlearning
into two clear functions, using the same deleted node indices for both parts.
"""
from collections import Counter
import pandas as pd
from torch_geometric.data import Data
import torch.nn as nn
import torch.optim as optim
import time
import torch.nn.functional as F
from utils.common_utils import evaluate_test_acc, evaluate_test_f1
from database.data_preprocessing.data_preprocessing_K import (
    preprocess_node_features,
    generate_hyperedge_dict,
)
from HGNNs_Model.HGNNP import HGNNP_implicit, build_incidence_matrix, compute_degree_vectors
from GIF.GIF_HGNNP_ROW_NEI import (
    rebuild_structure_after_node_deletion,
    approx_gif,
    train_model,
)
from config import get_args
import numpy as np
import torch
from MIA.MIA_utils import train_shadow_model, membership_inference
from paths import ACI_TEST, ACI_TRAIN
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None, reduction: str = 'mean'):
        """
        gamma: focusing parameter, usually set to 2.0
        weight: class weight tensor([w0, w1]), optional
        reduction: 'mean' or 'sum'
        """
        super().__init__()
        self.gamma    = gamma
        self.weight   = weight
        self.reduction= reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 1) Compute standard CE first (without reduction)
        ce_loss = F.cross_entropy(logits, targets,
                                  weight=self.weight,
                                  reduction='none')
        # 2) Compute pt = exp(-CE)
        pt = torch.exp(-ce_loss)
        # 3) Apply focal weighting
        loss = ((1 - pt) ** self.gamma) * ce_loss
        # 4) reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
def mia_with_hgnn(X_train, y_train, hyperedges, args, device):
    """
    1) Construct the data structure Data_full for the full model
    2) Train the full model (HGNN⁺)
    3) Call membership_inference(...), passing in the trained full_model
    4) Return (AUC, F1) on the full model
    """
    N = X_train.shape[0]

    # 1) Construct the hypergraph and degree inverses
    H_sp        = build_incidence_matrix(hyperedges, N)
    dv_inv, de_inv = compute_degree_vectors(H_sp)
    Hc = H_sp.tocoo()
    idx = torch.LongTensor([Hc.row, Hc.col])
    val = torch.FloatTensor(Hc.data)
    H_tensor = torch.sparse_coo_tensor(idx, val, Hc.shape).to(device)

    # 2) Prepare Data_full, using all nodes for training
    full_train_mask = torch.ones(N, dtype=torch.bool, device=device)
    data_full = Data(
        x = torch.from_numpy(X_train).float().to(device),
        y = torch.from_numpy(y_train).long().to(device),
        H         = H_tensor,
        dv_inv    = torch.from_numpy(dv_inv).to(device),
        de_inv    = torch.from_numpy(de_inv).to(device),
        train_mask    = full_train_mask,
        train_indices = list(range(N)),
        test_indices  = []               # test_indices are not needed for full model training
    )

    # 3) Initialize and train the full model (HGNN⁺)
    full_model = HGNNP_implicit(
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

    # 4) Use membership_inference to compute AUC/F1 on the full model
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

def retrain_on_pruned(X, y, df, transformer, deleted_idx, args, device):
    """
    Prunes training data by removing nodes at deleted_idx,
    retrains a HGNN⁺ model from scratch, and evaluates on the test set.
    Additional outputs now include:
      - total number of training nodes after pruning
      - hypergraph reconstruction time & total number of hyperedges
      - training time & total runtime
      - total number of hyperedges in the test set
      - test-set F1 & Acc
      - evaluation on deleted nodes & MIA evaluation
    """
    num_train = X.shape[0]
    keep_mask = np.ones(num_train, dtype=bool)
    keep_mask[deleted_idx] = False
    # Fix: ensure boolean array indexing works properly
    X = np.asarray(X)
    y = np.asarray(y)

    # 1) Prune the data
    X_pruned = X[keep_mask]
    y_pruned = y[keep_mask]
    df_pruned = df.drop(index=deleted_idx).reset_index(drop=True)
    print(f"[Retraining] Number of training nodes after pruning: {X_pruned.shape[0]} / original {num_train}")

    # Start total time measurement
    t_start = time.time()

    # 2) Rebuild the hypergraph
    cat_cols = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ]
    t0 = time.time()
    hyperedges_pruned = generate_hyperedge_dict(
        df_pruned,
        cat_cols,
        max_nodes_per_hyperedge=getattr(args, 'max_nodes_per_hyperedge_train', 10),
        device=device,
    )
    t_hg = time.time() - t0
    print(f"[Retraining] Hypergraph reconstruction time: {t_hg:.2f}s")
    print(f"[Retraining] Total number of hyperedges after pruning: {len(hyperedges_pruned)}")

    # Build the incidence matrix and degree vectors
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

    # 3) Model training — using HGNN⁺
    t_init_start = time.time()
    model = HGNNP_implicit(
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

    # Output runtime for all stages in one line
    print(
        f"[Retraining] Initialization: {t_init_end - t_init_start:.2f}s | "
        f"Training: {t_train_end - t_train_start:.2f}s | "
        f"Total: {t_total:.2f}s"
    )
    train_csv, test_csv = ACI_TRAIN, ACI_TEST

    # 4) Test-set evaluation
    X_test, y_test, df_test, _ = preprocess_node_features(
        test_csv, is_test=True, transformer=transformer
    )
    t2 = time.time()
    hyperedges_test = generate_hyperedge_dict(
        df_test, cat_cols,
        max_nodes_per_hyperedge=getattr(args, 'max_nodes_per_hyperedge', 50),
        device=device
    )
    t_hg_te = time.time() - t2
    print(f"[Retraining] Test-set hypergraph reconstruction time: {t_hg_te:.2f}s")
    print(f"[Retraining] Total number of hyperedges in test set: {len(hyperedges_test)}")

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
    f1 = evaluate_test_f1(model, test_obj)
    acc = evaluate_test_acc(model, test_obj)
    print(f"[Retraining] Test set F1: {f1:.4f}, Acc: {acc:.4f}")

    # Evaluation on deleted nodes & MIA
    X_del = X[deleted_idx]
    y_del = y[deleted_idx]
    df_del = df.iloc[deleted_idx].reset_index(drop=True)

    hyperedges_del = generate_hyperedge_dict(
        df_del, cat_cols,
        max_nodes_per_hyperedge=getattr(args, 'max_nodes_per_hyperedge_train', 10),
        device=device
    )
    H_sp_del = build_incidence_matrix(hyperedges_del, len(deleted_idx))
    dv_np_del, de_np_del = compute_degree_vectors(H_sp_del)
    H_coo_del = H_sp_del.tocoo()
    idx_del = torch.LongTensor(np.vstack((H_coo_del.row, H_coo_del.col))).to(device)
    val_del = torch.FloatTensor(H_coo_del.data).to(device)
    H_del = torch.sparse_coo_tensor(idx_del, val_del, size=H_coo_del.shape).coalesce()

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

    num_hyperedges = len(hyperedges_del)
    has_nonzero_features = (fts_del.abs().sum().item() > 0)
    print(f"Number of hyperedges in deleted set: {num_hyperedges}, deleted-set node features not zeroed out? {has_nonzero_features}")

    model.eval()
    with torch.no_grad():
        f1_del = evaluate_test_f1(model, test_obj_del)
        acc_del = evaluate_test_acc(model, test_obj_del)
    print(f"[Deleted-Node Evaluation] Retrained model scores on deleted nodes: F1={f1_del:.4f}, Acc={acc_del:.4f}")

    all_idx = np.arange(num_train)
    keep_idx = np.setdiff1d(all_idx, deleted_idx)
    X_keep, y_keep = X[keep_idx], y[keep_idx]
    he_keep = hyperedges_pruned
    he_attack = {}
    cnt = 0
    for nodes in he_keep.values():
        he_attack[cnt] = nodes
        cnt += 1
    offset = len(X_keep)
    for nodes in hyperedges_del.values():
        he_attack[cnt] = [n + offset for n in nodes]
        cnt += 1

    X_attack = np.vstack([X_keep, X_del])
    y_attack = np.hstack([y_keep, y_del])
    member_mask = np.concatenate([
        np.ones(len(X_keep), dtype=bool),
        np.zeros(len(X_del), dtype=bool)
    ])

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

    return t_total, acc, auc_rt

def full_train_and_unlearn(X, y, df, transformer, deleted_idx, args, device):
    """
    Trains on the full dataset, evaluates before unlearning,
    performs GIF-based unlearning on deleted_idx, then evaluates again.
    """
    # Fix: ensure integer indexing works properly
    X = np.asarray(X)
    y = np.asarray(y)

    # Build full hypergraph
    cat_cols = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ]
    hyperedges = generate_hyperedge_dict(
        df, cat_cols,
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

    # Train full model — using HGNN⁺
    model = HGNNP_implicit(
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
    # criterion = nn.CrossEntropyLoss()
    criterion = FocalLoss(
        gamma=2.0,
        # weight=weights  # Uncomment if class weights are needed
        reduction='mean'
    )

    model = train_model(
        model, criterion, optimizer, scheduler,
        fts, lbls, H, dv, de,
        num_epochs=getattr(args, 'epochs', 200), print_freq=10
    )

    # Prepare test data
    train_csv, test_csv = ACI_TRAIN, ACI_TEST
    X_test, y_test, df_test, _ = preprocess_node_features(
        test_csv, is_test=True, transformer=transformer
    )
    hyperedges_test = generate_hyperedge_dict(
        df_test, cat_cols,
        max_nodes_per_hyperedge=getattr(args, 'max_nodes_per_hyperedge', 50),
        device=device
    )

    # Count the number of positive and negative samples in the test set
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
    print(f"Before GIF Unlearning: F1={f1_before:.4f}, Acc={acc_before:.4f}")

    # Build data_obj for GIF
    data_obj = {"x": fts, "y": lbls, "H": H, "dv_inv": dv, "de_inv": de, "train_mask": train_mask}

    # Newly added: compute the number of misclassified samples on the test set
    with torch.no_grad():
        # test_obj already contains fts_test, lbls_test, H_test, dv_test, de_test
        logits_after = model(
            test_obj["x"],
            test_obj["H"],
            test_obj["dv_inv"],
            test_obj["de_inv"]
        )
        preds_after = logits_after.argmax(dim=1)  # predicted classes
        mismatches = (preds_after != test_obj["y"])  # boolean mask
        num_errors = mismatches.sum().item()  # number of errors
        total = test_obj["y"].shape[0]  # test-set size

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

    # Replace it with the following: first package deleted_idx, hyperedges, and K, then call:
    K = getattr(args, 'neighbor_k', 12)  # setting: nodes sharing at least K hyperedges are considered neighbors
    unlearn_info = (deleted_idx, hyperedges, K)
    unlearning_time, _ = approx_gif(
        model,
        data_obj,
        unlearn_info,
        iteration=args.if_iters,
        damp=args.if_damp,
        scale=args.if_scale
    )

    print(f"[Unlearn] Took {unlearning_time:.4f}s")

    # Evaluate after unlearning
    f1_after = evaluate_test_f1(model, test_obj)
    acc_after = evaluate_test_acc(model, test_obj)
    print(f"After Unlearning: F1={f1_after:.4f}, Acc={acc_after:.4f}")

    # Newly added: compute the number of misclassified samples on the test set
    with torch.no_grad():
        # test_obj already contains fts_test, lbls_test, H_test, dv_test, de_test
        logits_after = model(
            test_obj["x"],
            test_obj["H"],
            test_obj["dv_inv"],
            test_obj["de_inv"]
        )
        preds_after = logits_after.argmax(dim=1)  # predicted classes
        mismatches = (preds_after != test_obj["y"])  # boolean mask
        num_errors = mismatches.sum().item()  # number of errors
        total = test_obj["y"].shape[0]  # test-set size

    # Newly added: evaluate the model on deleted nodes after retraining
    # X, y, df, deleted_idx are all available in the function scope
    # 1) Prepare the subset of deleted nodes
    X_del = X[deleted_idx]
    y_del = y[deleted_idx]
    df_del = df.iloc[deleted_idx].reset_index(drop=True)

    # 2) Rebuild the hypergraph
    hyperedges_del = generate_hyperedge_dict(
        df_del, cat_cols,
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

    # 1) Check whether the number of hyperedges in the deleted set is greater than 0
    num_hyperedges = len(hyperedges_del)

    # 2) Check whether the node features in the deleted set are not all zeros
    #    Here we assume fts_del is a torch.Tensor
    has_nonzero_features = (fts_del.abs().sum().item() > 0)

    # 3) Output the result
    print(f"Number of hyperedges in deleted set: {num_hyperedges}, "
          f"Deleted-set node features not zeroed out? {has_nonzero_features}")
    # 4) Use the retrained model for prediction
    model.eval()
    with torch.no_grad():
        # Directly call your evaluation functions
        f1_del = evaluate_test_f1(model, test_obj_del)
        acc_del = evaluate_test_acc(model, test_obj_del)

    print(f"[Deleted-Node Evaluation] Retrained model scores on deleted nodes: F1={f1_del:.4f}, Acc={acc_del:.4f}")

    # MIA-On-Unlearned
    # 1) Split keep / del

    all_idx  = np.arange(X.shape[0])
    keep_idx = np.setdiff1d(all_idx, deleted_idx)
    X_keep, y_keep = X[keep_idx], y[keep_idx]
    X_del,  y_del  = X[deleted_idx], y[deleted_idx]

    # 2) Rebuild hyperedges for keep / del
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

    # 3) Merge hyperedges (node indices in the del part need an offset)
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
        np.ones(len(X_keep), dtype=bool),   # keep nodes as positive samples
        np.zeros(len(X_del),  dtype=bool)   # del nodes as negative samples
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
    # ----------------------------

    return unlearning_time, acc_after, auc_un
def batch_main(runs=20):
    args     = get_args()
    device   = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # 1. Preload the training set
    train_csv, test_csv = ACI_TRAIN, ACI_TEST
    X_train, y_train, df_train, transformer = preprocess_node_features(train_csv, is_test=False)
    num_train = X_train.shape[0]

    # Value-based deletion logic
    # Samples with sex=='Male' and relationship=='Husband' will be deleted
    mask = (df_train['sex'] == 'Male') & (df_train['relationship'] == 'Husband')
    deleted_idx = np.where(mask.values)[0]
    print(f"[Batch] Value-based deletion: {len(deleted_idx)} nodes in total, first 5 indices: {deleted_idx[:5]}")

    # 2. Prepare result storage
    records = {
        'retrain_time': [], 'retrain_acc': [], 'retrain_mia': [],
        'unlearn_time': [], 'unlearn_acc': [], 'unlearn_mia': []
    }

    for run in range(1, runs+1):
        print(f"\n=== Run {run}/{runs} ===")

        # 3. Retraining
        t_retrain, acc_retrain, auc_retrain = retrain_on_pruned(
            X_train, y_train, df_train, transformer,
            deleted_idx, args, device
        )
        records['retrain_time'].append(t_retrain)
        records['retrain_acc'].append(acc_retrain)
        records['retrain_mia'].append(auc_retrain)

        # 4. GIF Unlearning
        t_unlearn, acc_unlearn, auc_unlearn = full_train_and_unlearn(
            X_train, y_train, df_train, transformer,
            deleted_idx, args, device
        )
        records['unlearn_time'].append(t_unlearn)
        records['unlearn_acc'].append(acc_unlearn)
        records['unlearn_mia'].append(auc_unlearn)

    # 5. Summary
    df = pd.DataFrame(records)
    summary = df.describe().loc[['mean','std','min','max']]
    print("\n=== 20‐Run Summary ===")
    print(summary)

    # Optional: save to CSV
    # df.to_csv("unlearning_batch_results.csv", index=False)
    return df, summary

if __name__ == "__main__":
    batch_main(20)