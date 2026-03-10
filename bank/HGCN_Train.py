#!/usr/bin/env python
# coding: utf-8
"""
train_unlearning_HGCN_ALLDELETE.py

1) Load the training data, construct the hypergraph, and train HyperGCN;
2) After training, randomly select a% of nodes from the training data as "nodes to delete" and perform Unlearning using the GIF method;
3) Evaluate the model performance difference before and after deletion on an independent test dataset (micro-F1).
"""
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
from bank.GIF_HGCN_ROW import *
from GIF.GIF_HGCN_ROW import train_model
import torch.nn as nn
import torch.optim as optim

from bank.HGCN_utils import evaluate_hgcn_f1, evaluate_hgcn_acc
from bank.HGNN.data_preprocessing_bank import preprocess_node_features_bank, generate_hyperedge_dict_bank
from bank.HGCN_bank import HyperGCN, laplacian  # your custom HyperGCN model
import argparse
from paths import BANK_DATA

def main():
    # —— 0) Command-line arguments & device —— #
    parser = argparse.ArgumentParser(
        description="HyperGCN on the Bank dataset (with column-level/node-level Unlearning)"
    )
    default_csv = BANK_DATA
    parser.add_argument(
        "--data-csv", type=str, default=default_csv,
        help=f"Full Bank dataset CSV file (default: {default_csv})"
    )
    parser.add_argument(
        "--split-ratio", type=float, default=0.2,
        help="Test set ratio (between 0 and 1, default 0.2)"
    )
    parser.add_argument(
        "--categate-cols", nargs="+",
        default=["job", "marital", "education", "default",
                 "housing", "loan", "contact", "month", "poutcome"],
        help="List of categorical columns used to generate discrete hyperedges"
    )
    parser.add_argument(
        "--continuous-cols", nargs="+",
        default=["age", "balance", "day",
                 "duration", "campaign", "pdays", "previous"],
        help="List of continuous columns used to generate continuous hyperedges"
    )
    parser.add_argument(
        "--max-nodes-per-hyperedge", type=int, default=50,
        help="Maximum number of nodes allowed in a single hyperedge"
    )
    parser.add_argument("--depth",        type=int,   default=3,     help="HyperGCN depth (number of layers)")
    parser.add_argument("--dropout",      type=float, default=0.01,  help="dropout ratio")
    parser.add_argument("--fast",         action="store_true",        help="Enable Fast mode")
    parser.add_argument("--mediators",    action="store_true",        help="Use Mediators in Laplacian")
    parser.add_argument("--dataset",      type=str,   default="bank", help="Dataset name, used for logging")
    parser.add_argument("--lr",           type=float, default=0.01,  help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.001, help="Weight decay")
    parser.add_argument(
        "--milestones", nargs="+", type=int, default=[100, 150],
        help="Milestones for learning-rate scheduling (list of epochs)"
    )
    parser.add_argument("--gamma",        type=float, default=0.1,   help="Decay factor for learning-rate scheduling")
    parser.add_argument("--epochs",       type=int,   default=120,   help="Number of training epochs")
    parser.add_argument("--log-every",    type=int,   default=10,    help="Print logs every N steps")
    # GIF update hyperparameters
    parser.add_argument("--gif-iters", type=int,   default=80,    help="Number of GIF update iterations (default 80)")
    parser.add_argument("--gif-damp",  type=float, default=0.01,  help="GIF damping coefficient (default 0.01)")
    parser.add_argument("--gif-scale", type=float, default=1e7,   help="GIF scale (default 1e7)")
    args = parser.parse_args()

    # —— 1) Read and split the training/test set —— #
    if not os.path.isfile(args.data_csv):
        raise FileNotFoundError(f"Data file not found: {args.data_csv}")

    df_full = pd.read_csv(
        args.data_csv,
        sep=';', header=0,
        skipinitialspace=True
    )
    print(f"Total dataset size: {len(df_full)}, distribution → {df_full['y'].value_counts().to_dict()}")

    df_tr, df_te = train_test_split(
        df_full,
        test_size=args.split_ratio,
        random_state=42,
        stratify=df_full["y"]
    )
    df_tr.reset_index(drop=True, inplace=True)
    df_te.reset_index(drop=True, inplace=True)

    print(f"Training set size: {len(df_tr)}, distribution → {df_tr['y'].value_counts().to_dict()}")
    print(f"Test set size: {len(df_te)}, distribution → {df_te['y'].value_counts().to_dict()}")

    # —— 2) Device setup —— #
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # —— 3) Training set preprocessing & hyperedge construction —— #
    print("—— Training set preprocessing ——")
    X_tr, y_tr, df_tr_proc, transformer = preprocess_node_features_bank(
        df_tr, is_test=False
    )
    hyperedges_tr = list(generate_hyperedge_dict_bank(
        df_tr_proc,
        args.categate_cols,
        args.continuous_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        device=device
    ).values())

    # —— 4) Laplacian construction & model initialization —— #
    print("—— Constructing initial Laplacian and initializing model ——")
    A_before = laplacian(hyperedges_tr, X_tr, args.mediators).to(device)
    A_tr_before = A_before  # save for later restoration

    cfg = lambda: None
    cfg.d         = X_tr.shape[1]
    cfg.depth     = args.depth
    cfg.c         = int(max(y_tr) + 1)
    cfg.dropout   = args.dropout
    cfg.fast      = args.fast
    cfg.mediators = args.mediators
    cfg.cuda      = False
    cfg.dataset   = args.dataset

    model = HyperGCN(
        num_nodes=X_tr.shape[0],
        edge_list=hyperedges_tr,
        X_init=X_tr,
        args=cfg
    ).to(device)
    model.structure = A_before
    for layer in model.layers:
        layer.reapproximate = False

    # —— 5) Training —— #
    print("—— Start training ——")
    fts = torch.from_numpy(X_tr).float().to(device)
    lbls = torch.tensor(y_tr, dtype=torch.long).to(device)
    train_mask = torch.ones(len(y_tr), dtype=torch.bool, device=device)
    data_train = {"x": fts, "y": lbls, "train_mask": train_mask}

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.milestones, gamma=args.gamma
    )
    criterion = nn.NLLLoss()
    model = train_model(
        model, criterion, optimizer, scheduler,
        fts, lbls,
        num_epochs=args.epochs,
        print_freq=args.log_every
    )
    print(f"Best Train Acc: {evaluate_hgcn_acc(model, data_train):.4f}")

    # —— 6) Test set preprocessing & pre-deletion evaluation —— #
    print("—— Test set preprocessing & pre-deletion evaluation ——")
    X_te, y_te, df_te_proc, _ = preprocess_node_features_bank(
        df_te, is_test=True, transformer=transformer
    )
    hyperedges_te = list(generate_hyperedge_dict_bank(
        df_te_proc,
        args.categate_cols,
        args.continuous_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        device=device
    ).values())
    A_te = laplacian(hyperedges_te, X_te, args.mediators).to(device)
    model.structure = A_te
    for layer in model.layers:
        layer.reapproximate = False

    fts_te = torch.from_numpy(X_te).float().to(device)
    lbls_te = torch.tensor(y_te, dtype=torch.long).to(device)
    data_test = {"x": fts_te, "y": lbls_te}

    print("— Before Unlearning —")
    print(f" Test F1: {evaluate_hgcn_f1(model, data_test):.4f}, "
          f"Acc: {evaluate_hgcn_acc(model, data_test):.4f}")

    # —— 7) Randomly select nodes to delete & pre-deletion subset evaluation —— #
    deleted_neighbors = []  # can use get_neighbors if needed
    deleted = torch.tensor(
        np.random.choice(X_tr.shape[0],
                         int(X_tr.shape[0] * 0.3),
                         replace=False),
        dtype=torch.long, device=device
    )

    model.structure = A_tr_before
    for layer in model.layers:
        layer.reapproximate = False

    mask_del = torch.zeros_like(lbls, dtype=torch.bool, device=device)
    mask_del[deleted] = True
    data_del_before = {"x": fts, "y": lbls, "train_mask": mask_del}
    print(f" Deleted-nodes F1 before: {evaluate_hgcn_f1(model, data_del_before):.4f}, "
          f"Acc: {evaluate_hgcn_acc(model, data_del_before):.4f}")

    # —— 7.2 Delete nodes → obtain zeroed features & new Laplacian —— #
    fts_new, edge_list_new, A_after = apply_node_deletion_unlearning(
        fts,
        edge_list=hyperedges_tr,
        deleted_nodes=deleted,
        mediators=args.mediators,
        device=device
    )
    print("Check whether the features of deleted nodes are all zero:", (fts_new[deleted] != 0).sum().item())
    print(f" Deleted {len(deleted)} nodes; {len(edge_list_new)} hyperedges remain")

    # —— 7.3 GIF update —— #
    gif_time = approx_gif(
        model,
        data_train,
        A_before,
        A_after,
        deleted_neighbors,
        x_before=fts,
        x_after=fts_new,
        deleted_nodes=deleted,
        iters=args.gif_iters,
        damp=args.gif_damp,
        scale=args.gif_scale
    )
    print(f" GIF time: {gif_time:.4f}s")

    # —— 7.4 Post-deletion subset evaluation —— #
    data_del_after = {"x": fts_new, "y": lbls, "train_mask": mask_del}
    print(f" Deleted-nodes F1 after: {evaluate_hgcn_f1(model, data_del_after):.4f}, "
          f"Acc: {evaluate_hgcn_acc(model, data_del_after):.4f}")

    # —— 8) Restore test structure & final evaluation —— #
    model.structure = A_te
    for layer in model.layers:
        layer.reapproximate = False

    print("— After Unlearning —")
    print(f" Test F1: {evaluate_hgcn_f1(model, data_test):.4f}, "
          f"Acc: {evaluate_hgcn_acc(model, data_test):.4f}")

if __name__ == "__main__":
    main()