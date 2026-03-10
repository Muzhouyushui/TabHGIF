
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from collections import Counter

from utils.common_utils import evaluate_test_acc, evaluate_test_f1
from Credit.HGNN.data_proprocessing_Credit_col_retrain  import (
    generate_hyperedge_dict,
    preprocess_node_features,
)
from bank.HGNN.HGNN import HGNN_implicit, build_incidence_matrix, compute_degree_vectors
from GIF.GIF_HGNN_COL import approx_gif_col, train_model
from paths import CREDIT_DATA

def main_baseline():
    parser = argparse.ArgumentParser(description="HGNN Baseline on Credit Approval Dataset")
    # parser.add_argument(
    #     "--data-csv", type=str,
    #     help="Credit Approval data path"
    # )
    parser.add_argument("--data-csv", type=str,
                        default=CREDIT_DATA,
                        help="Credit Approval data CSV file path")
    parser.add_argument("--split-ratio", type=float, default=0.2, help="Test set ratio")
    parser.add_argument("--max-nodes-per-hyperedge", type=int, default=50,
                        help="Max nodes per hyperedge")
    parser.add_argument("--hidden-dim", type=int, default=128, help="HGNN hidden dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--milestones", nargs='+', type=int, default=[100,150], help="LR milestones")
    parser.add_argument("--gamma", type=float, default=0.1, help="LR decay factor")
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {device}")

    # 1) Load & split raw data
    df = pd.read_csv(args.data_csv, header=None, na_values='?', skipinitialspace=True)
    df.columns = [f"A{i}" for i in range(1,16)] + ["class"]
    df = df.dropna(subset=["class"]).reset_index(drop=True)
    df_train, df_test = train_test_split(
        df, test_size=args.split_ratio,
        stratify=df["class"], random_state=42
    )
    df_train, df_test = df_train.reset_index(drop=True), df_test.reset_index(drop=True)
    print(f"TRAIN samples: {len(df_train)}, TEST samples: {len(df_test)}")
    print("– TRAIN dist:", Counter(df_train["class"]))
    print("– TEST  dist:", Counter(df_test["class"]))

    # Columns to drop
    drop_cols = ["A5"]

    # 2) Preprocess features (drop specified columns)
    X_train, y_train, df_train_proc, transformer = preprocess_node_features(
        data=df_train,
        transformer=None,
        drop_cols=drop_cols
    )
    X_test, y_test, df_test_proc, _ = preprocess_node_features(
        data=df_test,
        transformer=transformer,
        drop_cols=drop_cols
    )
    print(f"➤ TRAIN: X={X_train.shape}, y={y_train.shape}")
    print(f"➤ TEST : X={X_test.shape}, y={y_test.shape}")

    # 3) Build hyperedges & incidence
    cont_cols = ["A2","A3","A8","A11","A14","A15"]
    cat_cols  = [f"A{i}" for i in range(1,16) if f"A{i}" not in cont_cols]
    feature_cols = [c for c in cat_cols + cont_cols if c not in drop_cols]

    hyper_train = generate_hyperedge_dict(
        df_train_proc,
        feature_cols=feature_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        device=device,
        drop_cols=drop_cols
    )
    hyper_test = generate_hyperedge_dict(
        df_test_proc,
        feature_cols=feature_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        device=device,
        drop_cols=drop_cols
    )
    print(f"Hyperedges: train={len(hyper_train)}, test={len(hyper_test)}")

    # 4) Convert to incidence tensors and degree vectors
    H_train_sp = build_incidence_matrix(hyper_train, X_train.shape[0])
    dv_train, de_train = compute_degree_vectors(H_train_sp)
    Hc_tr = H_train_sp.tocoo()
    idx_tr = np.vstack((Hc_tr.row, Hc_tr.col)).astype(np.int64)
    H_train = torch.sparse_coo_tensor(
        torch.from_numpy(idx_tr),
        torch.from_numpy(Hc_tr.data).float(),
        size=Hc_tr.shape
    ).to(device)
    dv_train, de_train = torch.from_numpy(dv_train).float().to(device), torch.from_numpy(de_train).float().to(device)

    H_test_sp = build_incidence_matrix(hyper_test, X_test.shape[0])
    dv_test, de_test = compute_degree_vectors(H_test_sp)
    Hc_te = H_test_sp.tocoo()
    idx_te = np.vstack((Hc_te.row, Hc_te.col)).astype(np.int64)
    H_test = torch.sparse_coo_tensor(
        torch.from_numpy(idx_te),
        torch.from_numpy(Hc_te.data).float(),
        size=Hc_te.shape
    ).to(device)
    dv_test, de_test = torch.from_numpy(dv_test).float().to(device), torch.from_numpy(de_test).float().to(device)

    # 5) Prepare tensors
    X_tr_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_train, dtype=torch.long,    device=device)
    X_te_t = torch.tensor(X_test,  dtype=torch.float32, device=device)
    y_te_t = torch.tensor(y_test,  dtype=torch.long,    device=device)

    # 6) Initialize & train HGNN
    model = HGNN_implicit(
        in_ch=X_tr_t.size(1),
        n_class=int(y_tr_t.max().item()) + 1,
        n_hid=args.hidden_dim,
        dropout=args.dropout
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    criterion = nn.CrossEntropyLoss()

    print("—— Training baseline HGNN ——")
    model = train_model(
        model, criterion, optimizer, scheduler,
        X_tr_t, y_tr_t, H_train, dv_train, de_train,
        num_epochs=args.epochs, print_freq=10
    )

    # 7) Test evaluation
    test_obj = {
        "x": X_te_t,
        "y": y_te_t,
        "H": H_test,
        "dv_inv": dv_test,
        "de_inv": de_test
    }
    acc = evaluate_test_acc(model, test_obj)
    f1  = evaluate_test_f1(model, test_obj)
    print(f"Baseline HGNN Test — Acc: {acc:.4f}, F1: {f1:.4f}")

if __name__ == "__main__":
    main_baseline()
    for run in range(1,5):
        print("=== Run",run,"===")
        main_baseline()
