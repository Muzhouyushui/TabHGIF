#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from collections import Counter

from utils.common_utils import evaluate_test_acc, evaluate_test_f1
from bank.HGNN.data_preprocessing_bank_col import (
    generate_hyperedge_dict_bank,
    preprocess_node_features_bank,
    delete_feature_column,
)
from bank.HGNN.HGNN import HGNN_implicit, build_incidence_matrix, compute_degree_vectors
from GIF.GIF_HGNN_COL import approx_gif_col, train_model
from paths import BANK_DATA

def main():
    parser = argparse.ArgumentParser(description="HGNN on Bank dataset - Column-level Unlearning version")
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
    parser.add_argument("--max-nodes-per-hyperedge-train", type=int, default=50,
                        help="Maximum number of nodes per hyperedge in the training set")
    parser.add_argument("--max-nodes-per-hyperedge-test", type=int, default=50,
                        help="Maximum number of nodes per hyperedge in the test set")
    parser.add_argument("--hidden-dim", type=int, default=128, help="HGNN hidden dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout ratio")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="Adam weight decay")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--milestones", nargs='+', type=int, default=[100,150], help="Learning rate milestones")
    parser.add_argument("--gamma", type=float, default=0.1, help="Learning rate decay factor")
    parser.add_argument("--gif-iters", type=int, default=20, help="Number of GIF iterations")
    parser.add_argument("--gif-damp", type=float, default=0.01, help="GIF damping")
    parser.add_argument("--gif-scale", type=float, default=1e7, help="GIF scale")
    parser.add_argument(
        "--columns-to-unlearn", nargs=1, default=["age"],
        help="List of column names for column-level unlearning (single column)"
    )
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {device}")

    # —— 1) Read the full CSV and split ——
    # The Bank dataset is semicolon-separated, and the first row is the header
    df = pd.read_csv(args.data_csv, sep=';', header=0)
    # Ensure the 'y' column exists
    assert 'y' in df.columns, "Column 'y' not found in the CSV, please check the file and separator"
    df_train, df_test = train_test_split(
        df,
        test_size=args.split_ratio,
        random_state=42,
        stratify=df['y']
    )
    print(f"Number of training samples: {len(df_train)}, number of test samples: {len(df_test)}")
    print("– TRAIN label dist:", Counter(df_train['y']))
    print("– TEST  label dist:", Counter(df_test['y']))
    # —— 1) Read the full CSV and split ——
    df = pd.read_csv(args.data_csv, sep=';', header=0)
    df_train, df_test = train_test_split(
        df,
        test_size=args.split_ratio,
        random_state=42,
        stratify=df['y']
    )
    # Reset indices and discard the original row numbers
    df_train = df_train.reset_index(drop=True)
    df_test  = df_test .reset_index(drop=True)
    # —— 2) Preprocess the training set ——
    X_train, y_train, df_train_proc, transformer = preprocess_node_features_bank(
        df_train, is_test=False
    )
    print(f"After preprocessing TRAIN shape: {X_train.shape}, labels: {Counter(y_train)}")

    # —— 3) Build the training hypergraph ——
    cat_cols  = ['job','marital','education','default','housing','loan','contact','month','poutcome']
    cont_cols = ['age','balance','day','duration','campaign','pdays','previous']
    hyperedges_train = generate_hyperedge_dict_bank(
        df_train_proc, cat_cols, cont_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge_train,
        device=device
    )
    H_sp_tr = build_incidence_matrix(hyperedges_train, X_train.shape[0])
    dv_tr, de_tr = compute_degree_vectors(H_sp_tr)
    Hc_tr = H_sp_tr.tocoo()
    idx_tr = np.vstack((Hc_tr.row, Hc_tr.col)).astype(np.int64)
    H_tensor_tr = torch.sparse_coo_tensor(
        torch.from_numpy(idx_tr),
        torch.from_numpy(Hc_tr.data).float(),
        size=Hc_tr.shape
    ).to(device)
    fts_tr  = torch.from_numpy(X_train).float().to(device)
    lbls_tr = torch.LongTensor(y_train).to(device)

    # —— 4) Initialize the model & train ——
    model = HGNN_implicit(
        in_ch=fts_tr.shape[1],
        n_class=int(max(y_train)) + 1,
        n_hid=args.hidden_dim,
        dropout=args.dropout
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    criterion = nn.CrossEntropyLoss()

    def reason_once(data):
        return model(data["x"], data["H"], data["dv_inv"], data["de_inv"])
    model.reason_once = reason_once
    model.reason_once_unlearn = reason_once

    print("—— Start training ——")
    model = train_model(
        model, criterion, optimizer, scheduler,
        fts_tr, lbls_tr, H_tensor_tr,
        torch.from_numpy(dv_tr).float().to(device),
        torch.from_numpy(de_tr).float().to(device),
        num_epochs=args.epochs, print_freq=10
    )

    # —— 5) Preprocess the test set & first evaluation ——
    X_test, y_test, df_test_proc, _ = preprocess_node_features_bank(
        df_test, is_test=True, transformer=transformer
    )
    print(f"After preprocessing TEST shape: {X_test.shape}, labels: {Counter(y_test)}")

    hyperedges_test = generate_hyperedge_dict_bank(
        df_test_proc, cat_cols, cont_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge_test,
        device=device
    )
    H_sp_te = build_incidence_matrix(hyperedges_test, X_test.shape[0])
    dv_te, de_te = compute_degree_vectors(H_sp_te)
    Hc_te = H_sp_te.tocoo()
    idx_te = np.vstack((Hc_te.row, Hc_te.col)).astype(np.int64)
    H_tensor_te = torch.sparse_coo_tensor(
        torch.from_numpy(idx_te),
        torch.from_numpy(Hc_te.data).float(),
        size=Hc_te.shape
    ).to(device)
    fts_te  = torch.from_numpy(X_test).float().to(device)
    lbls_te = torch.LongTensor(y_test).to(device)

    test_obj = {
        "x": fts_te, "y": lbls_te,
        "H": H_tensor_te,
        "dv_inv": torch.from_numpy(dv_te).float().to(device),
        "de_inv": torch.from_numpy(de_te).float().to(device)
    }
    f1_before, acc_before = evaluate_test_f1(model, test_obj), evaluate_test_acc(model, test_obj)
    print(f"Before Unlearning — F1: {f1_before:.4f}, Acc: {acc_before:.4f}")

    # —— 6) Column-level Unlearning ——
    col_to_remove = args.columns_to_unlearn[0]

    # 6a) Back up the training structure
    fts_b, H_b, dv_b, de_b = (
        fts_tr.clone(),
        H_tensor_tr.coalesce(),
        torch.from_numpy(dv_tr).float().to(device),
        torch.from_numpy(de_tr).float().to(device)
    )

    # 6b) Delete the training-set column and update with GIF
    fts_a, H_tensor_a, hyperedges_a = delete_feature_column(
        fts_tr, transformer, col_to_remove,
        H_tensor_tr, hyperedges_train,
        continuous_cols=cont_cols
    )
    H_sp_a = build_incidence_matrix(hyperedges_a, fts_a.shape[0])
    dv_a, de_a = compute_degree_vectors(H_sp_a)
    Hc_a = H_sp_a.tocoo()
    idx_a = np.vstack((Hc_a.row, Hc_a.col)).astype(np.int64)
    H_tensor_a = torch.sparse_coo_tensor(
        torch.from_numpy(idx_a),
        torch.from_numpy(Hc_a.data).float(),
        size=Hc_a.shape
    ).to(device)

    batch_before = (fts_b, H_b, dv_b, de_b, lbls_tr)
    batch_after  = (fts_a, H_tensor_a, torch.from_numpy(dv_a).float().to(device),
                    torch.from_numpy(de_a).float().to(device), lbls_tr)
    v_updates    = approx_gif_col(
        model, criterion, batch_before, batch_after,
        cg_iters=args.gif_iters, damping=args.gif_damp, scale=args.gif_scale
    )
    with torch.no_grad():
        for p, v in zip(model.parameters(), v_updates):
            p.sub_(v)

    # —— 7) Post-deletion test evaluation ——
    fts_te_new, H_tensor_te_new, hyperedges_te_new = delete_feature_column(
        fts_te.clone(), transformer, col_to_remove,
        H_tensor_te, hyperedges_test,
        continuous_cols=cont_cols
    )
    H_sp_te_new = build_incidence_matrix(hyperedges_te_new, fts_te_new.shape[0])
    dv_te_new, de_te_new = compute_degree_vectors(H_sp_te_new)
    Hc_te_new = H_sp_te_new.tocoo()
    idx_te_new = np.vstack((Hc_te_new.row, Hc_te_new.col)).astype(np.int64)
    H_tensor_te_new = torch.sparse_coo_tensor(
        torch.from_numpy(idx_te_new),
        torch.from_numpy(Hc_te_new.data).float(),
        size=Hc_te_new.shape
    ).to(device)

    test_obj_after = {
        "x": fts_te_new, "y": lbls_te,
        "H": H_tensor_te_new,
        "dv_inv": torch.from_numpy(dv_te_new).float().to(device),
        "de_inv": torch.from_numpy(de_te_new).float().to(device)
    }
    f1_after, acc_after = evaluate_test_f1(model, test_obj_after), evaluate_test_acc(model, test_obj_after)
    print(f"After Unlearning  — F1: {f1_after:.4f}, Acc: {acc_after:.4f}")

if __name__ == "__main__":
    # main()
    for run in range(1,4):
        print("=== Run",run,"===")
        main()