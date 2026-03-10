#!/usr/bin/env python3
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from collections import Counter

from bank.HGCN.config import get_args
from bank.HGCN.data_preprocessing_col_retrain import (
    preprocess_node_features,
    generate_hyperedge_dict,
)
from bank.HGCN.HGCN_utils import evaluate_hgcn_f1, evaluate_hgcn_acc
from bank.HGCN.GIF_HGCN_COL_bank import approx_gif, train_model
from bank.HGCN.HGCN import HyperGCN, laplacian

def main():
    # 0) Arguments and device
    args = get_args()
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-ratio", type=float, default=args.split_ratio,
                        help="Test set ratio")
    cli_args = parser.parse_args()
    split_ratio = cli_args.split_ratio

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {device}")

    # 1) Read CSV and split
    df = pd.read_csv(args.train_csv, sep=';', header=0)
    assert 'y' in df.columns, "Column 'y' not found in the CSV"
    df_tr, df_te = train_test_split(
        df,
        test_size=split_ratio,
        random_state=42,
        stratify=df['y']
    )
    df_tr = df_tr.reset_index(drop=True)
    df_te = df_te.reset_index(drop=True)
    print(f"TRAIN={len(df_tr)} samples, TEST={len(df_te)} samples")
    print("– TRAIN dist:", Counter(df_tr['y']))
    print("– TEST  dist:", Counter(df_te['y']))

    # 2) Preprocessing: node features and labels
    X_tr, y_tr, df_tr_proc, transformer = preprocess_node_features(
        df_tr, is_test=False
    )
    X_te, y_te, df_te_proc, _ = preprocess_node_features(
        df_te, is_test=True, transformer=transformer
    )
    args.d = X_tr.shape[1]
    args.c = int(max(y_tr)) + 1

    # 3) Construct hypergraph & Laplacian
    hyper_tr = generate_hyperedge_dict(
        df_tr_proc, args.cat_cols, args.cont_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        device=device
    )
    A_tr = laplacian(list(hyper_tr.values()), X_tr, args.mediators).to(device)

    # 4) Train model
    fts_tr = torch.from_numpy(X_tr).float().to(device)
    lbls_tr = torch.tensor(y_tr, dtype=torch.long, device=device)
    model = HyperGCN(X_tr.shape[0], list(hyper_tr.values()), X_tr, args).to(device)
    model.structure = A_tr
    for layer in model.layers:
        layer.reapproximate = False

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=args.milestones,
                                               gamma=args.gamma)
    criterion = nn.NLLLoss()

    print("—— Training model ——")
    model = train_model(
        model, criterion, optimizer, scheduler,
        fts_tr, lbls_tr,
        num_epochs=args.epochs,
        print_freq=args.log_every
    )

    # 5) Test evaluation
    hyper_te = generate_hyperedge_dict(
        df_te_proc, args.cat_cols, args.cont_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        device=device
    )
    A_te = laplacian(list(hyper_te.values()), X_te, args.mediators).to(device)
    fts_te = torch.from_numpy(X_te).float().to(device)
    model.structure = A_te
    print("—— Evaluation on test set ——")
    test_obj = {"lap": fts_te, "y": torch.tensor(y_te, device=device)}
    print(f"Test Acc: {evaluate_hgcn_acc(model, test_obj):.4f}, ")
    print(f"Test F1 : {evaluate_hgcn_f1(model, test_obj):.4f}")

if __name__ == "__main__":
    for run in range(1,5):
        print("=== Run",run,"===")
        main()