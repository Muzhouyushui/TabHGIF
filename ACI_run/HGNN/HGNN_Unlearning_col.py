#!/usr/bin/env python3
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils.common_utils import evaluate_test_acc, evaluate_test_f1
from collections import Counter

# Column-level preprocessing & deletion interface
from database.data_preprocessing.data_preprocessing_column import (
    preprocess_node_features_HGNNcol,
    generate_hyperedge_dict,
    delete_feature_column,
)

# HGNN basic modules
from HGNNs_Model.HGNN.HGNN_2 import HGNN_implicit, build_incidence_matrix, compute_degree_vectors
from config import get_args
args = get_args()

# Column-level GIF interface
from GIF.GIF_HGNN_COL import approx_gif_col, train_model

def main():
    args = get_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(args)

    ###################################
    # 1) Load training data and train the model
    ###################################
    X_train, y_train, df_train, transformer = preprocess_node_features_HGNNcol(
        args.train_csv, is_test=False
    )
    cat_cols = args.cat_cols
    hyperedges = generate_hyperedge_dict(
        df_train, cat_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge_train,
        device=device
    )

    # Build sparse tensor for the training set
    H_sparse = build_incidence_matrix(hyperedges, X_train.shape[0])
    dv_train_np, de_train_np = compute_degree_vectors(H_sparse)
    H_coo = H_sparse.tocoo()
    idx_train = np.vstack((H_coo.row, H_coo.col)).astype(np.int64)
    H_tensor = torch.sparse_coo_tensor(
        torch.from_numpy(idx_train), torch.FloatTensor(H_coo.data), size=H_coo.shape
    ).to(device)

    dv_inv = torch.FloatTensor(dv_train_np).to(device)
    de_inv = torch.FloatTensor(de_train_np).to(device)
    fts = torch.FloatTensor(X_train).to(device)
    lbls = torch.LongTensor(y_train).to(device)

    model = HGNN_implicit(
        in_ch=fts.shape[1], n_class=args.n_class,
        n_hid=args.hidden_dim, dropout=args.dropout
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    criterion = nn.CrossEntropyLoss()

    # Train the model
    model = train_model(
        model, criterion, optimizer, scheduler,
        fts, lbls, H_tensor, dv_inv, de_inv,
        num_epochs=args.epochs, print_freq=args.log_every
    )

    ###################################
    # 2) Prepare test set + first evaluation (before deletion)
    ###################################
    X_test, y_test, df_test, _ = preprocess_node_features_HGNNcol(
        args.test_csv, is_test=True, transformer=transformer
    )
    hyperedges_test = generate_hyperedge_dict(
        df_test, cat_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge_test,
        device=device
    )
    H_sparse_test = build_incidence_matrix(hyperedges_test, X_test.shape[0])
    dv_test_np, de_test_np = compute_degree_vectors(H_sparse_test)
    H_coo_test = H_sparse_test.tocoo()
    idx_test = np.vstack((H_coo_test.row, H_coo_test.col)).astype(np.int64)
    H_test_tensor = torch.sparse_coo_tensor(
        torch.from_numpy(idx_test), torch.FloatTensor(H_coo_test.data), size=H_coo_test.shape
    ).to(device)

    test_fts = torch.FloatTensor(X_test).to(device)
    test_lbls = torch.LongTensor(y_test).to(device)
    test_data_obj = {
        "x": test_fts,
        "y": test_lbls,
        "H": H_test_tensor,
        "dv_inv": torch.FloatTensor(dv_test_np).to(device),
        "de_inv": torch.FloatTensor(de_test_np).to(device),
    }
    f1_before = evaluate_test_f1(model, test_data_obj)
    acc_before = evaluate_test_acc(model, test_data_obj)
    print(f"Before Unlearning — F1: {f1_before:.4f}, Acc: {acc_before:.4f}")

    ###################################
    # 2.5) Evaluate data deletion only (without parameter update)
    ###################################
    # Reuse fts_test_new and H_test_tensor_new obtained earlier from delete_feature_column
    col_to_remove = args.columns_to_unlearn[0]

    # (If not prepared yet, call it again as before)
    fts_test_new, H_test_tensor_new, hyperedges_test_new = delete_feature_column(
        test_fts.clone(), transformer, col_to_remove,
        H_test_tensor, hyperedges_test,
        continuous_cols=args.continuous_cols
    )
    # Recompute degree vectors
    H_sparse_test_new     = build_incidence_matrix(hyperedges_test_new, fts_test_new.shape[0])
    dv_test_np_new, de_test_np_new = compute_degree_vectors(H_sparse_test_new)
    # Construct evaluation object for data-only deletion
    test_data_obj_del = {
        "x":      fts_test_new,
        "y":      test_lbls,
        "H":      H_test_tensor_new,
        "dv_inv": torch.FloatTensor(dv_test_np_new).to(device),
        "de_inv": torch.FloatTensor(de_test_np_new).to(device),
    }
    model.eval()
    f1_deldata = evaluate_test_f1(model, test_data_obj_del)
    acc_deldata = evaluate_test_acc(model, test_data_obj_del)
    print(f"After Data-Only Deletion — F1: {f1_deldata:.4f}, Acc: {acc_deldata:.4f}")

    # Count the number of positive and negative samples in the training set
    train_counter = Counter(y_train)
    num_neg_train = train_counter[0]
    num_pos_train = train_counter[1]
    print(f"Training set → <=50K: {num_neg_train}, >50K: {num_pos_train}")

    # Check the specific values of model predictions and true labels to confirm correctness
    # spot_check_samples_HGNN(test_data_obj, model, k=150)

    ###################################
    # 3) Column-level Unlearning: delete the specified column and update model parameters
    ###################################
    col_to_remove = args.columns_to_unlearn[0]

    # ❶ Backup the state before deletion (must be done before delete_feature_column)
    fts_before = fts.clone()
    H_before = H_tensor.coalesce()
    dv_before = dv_inv.clone()
    de_before = de_inv.clone()



    # ❷ Perform deletion (fts itself is modified in place)
    fts_after, H_tensor_after, hyperedges_new = delete_feature_column(
        fts, transformer, col_to_remove,
        H_tensor, hyperedges,
        continuous_cols=args.continuous_cols
    )
    # Debug output: check whether the features were actually deleted
    # Count the number of positive and negative samples in the test set
    test_counter = Counter(y_test)
    num_neg_test = test_counter[0]
    num_pos_test = test_counter[1]
    print(f"Test set → <=50K: {num_neg_test}, >50K: {num_pos_test}")

    print("<<<< TRAIN DELETE DEBUG:",
          f"sum(fts_before)={fts_before.sum().item():.4e}",
          f"sum(fts_after)={fts_after.sum().item():.4e}")
    t1=time.time()
    # Recompute degree vectors and rebuild H_tensor_after
    H_sparse_new = build_incidence_matrix(hyperedges_new, fts_after.shape[0])
    dv_np_new, de_np_new = compute_degree_vectors(H_sparse_new)
    H_coo_new = H_sparse_new.tocoo()
    idx_new = np.vstack((H_coo_new.row, H_coo_new.col)).astype(np.int64)
    H_tensor_after = torch.sparse_coo_tensor(
        torch.from_numpy(idx_new), torch.FloatTensor(H_coo_new.data), size=H_coo_new.shape
    ).to(device)
    dv_after = torch.FloatTensor(dv_np_new).to(device)
    de_after = torch.FloatTensor(de_np_new).to(device)

    # ❸ Construct batches and call GIF
    batch_before = (fts_before, H_before, dv_before, de_before, lbls)
    batch_after = (fts_after, H_tensor_after, dv_after, de_after, lbls)
    v_updates = approx_gif_col(
        model, criterion,
        batch_before, batch_after,
        cg_iters=args.if_iters,
        damping=args.if_damp,
        scale=1e2
    )
    t2=time.time()
    print("GIF time=",t2-t1)
    with torch.no_grad():
        norm_v = torch.sqrt(sum((dv ** 2).sum() for dv in v_updates))
        print(f"<<<< GIF UPDATE DEBUG: ||v_updates||₂ = {norm_v.item():.4e}")
        # Then apply the updates
        for p, dv in zip(model.parameters(), v_updates):
            p.sub_(dv)
    ###################################
    # 4) Synchronize test set deletion + second evaluation (after deletion)
    ###################################
    fts_test_new, H_test_new, hyperedges_test_new = delete_feature_column(
        test_fts.clone(), transformer, col_to_remove,
        H_test_tensor, hyperedges_test,
        continuous_cols=args.continuous_cols
    )
    # Debug output: check test feature deletion
    print("<<<< TEST DELETE DEBUG:",
          f"sum(test_fts)={test_fts.sum().item():.4e}",
          f"sum(fts_test_new)={fts_test_new.sum().item():.4e}",
          f"len(hyperedges_test_new)={len(hyperedges_test_new)}")

    # Recompute degree vectors for the test set and rebuild H_test_tensor_new
    H_sparse_test_new = build_incidence_matrix(hyperedges_test_new, fts_test_new.shape[0])
    dv_test_np_new, de_test_np_new = compute_degree_vectors(H_sparse_test_new)
    H_coo_test_new = H_sparse_test_new.tocoo()
    idx_test_new = np.vstack((H_coo_test_new.row, H_coo_test_new.col)).astype(np.int64)
    H_test_tensor_new = torch.sparse_coo_tensor(
        torch.from_numpy(idx_test_new), torch.FloatTensor(H_coo_test_new.data), size=H_coo_test_new.shape
    ).to(device)

    test_data_obj_new = {
        "x": fts_test_new,
        "y": test_lbls,
        "H": H_test_tensor_new,
        "dv_inv": torch.FloatTensor(dv_test_np_new).to(device),
        "de_inv": torch.FloatTensor(de_test_np_new).to(device),
    }
    f1_after = evaluate_test_f1(model, test_data_obj_new)
    acc_after = evaluate_test_acc(model, test_data_obj_new)
    print(f"After Unlearning  — F1: {f1_after:.4f}, Acc: {acc_after:.4f}")




if __name__ == "__main__":
    # If you want to run multiple times for stability testing, you can add a loop:
    for run in range(1,5):
        print("=== Run",run,"===")
        main()
    # main()