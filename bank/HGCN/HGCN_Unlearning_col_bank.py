#!/usr/bin/env python3
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from collections import Counter

from bank.HGCN.config import get_args
from bank.HGCN.data_preprocessing_col_bank import (
    preprocess_node_features,
    generate_hyperedge_dict,
    delete_feature_columns_hgcn
)
from bank.HGCN.HGCN_utils import evaluate_hgcn_f1, evaluate_hgcn_acc
from bank.HGCN.GIF_HGCN_COL_bank import approx_gif, train_model
from bank.HGCN.HGCN import HyperGCN, laplacian

def main():
    # 0) Arguments
    args = get_args()
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-ratio", type=float, default=args.split_ratio,
                        help="Test set ratio")
    parser.add_argument("--columns-to-unlearn", nargs='+',
                        default=args.columns_to_unlearn,
                        help="List of column names to unlearn")
    cli_args = parser.parse_args()
    cols = cli_args.columns_to_unlearn
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

    # 2) Preprocessing: training set first
    X_tr, y_tr, df_tr_proc, transformer = preprocess_node_features(
        df_tr, is_test=False
    )
    # Then test set (transform only)
    X_te, y_te, df_te_proc, _ = preprocess_node_features(
        df_te, is_test=True, transformer=transformer
    )
    # Inject the two attributes required by HyperGCN into args
    args.d = X_tr.shape[1]          # Number of input channels
    args.c = int(max(y_tr)) + 1     # Number of classes

    # 3) Generate hyperedges & Laplacian
    hyper_tr = generate_hyperedge_dict(
        df_tr_proc, args.cat_cols, args.cont_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        device=device
    )
    A_before = laplacian(list(hyper_tr.values()), X_tr, args.mediators).to(device)

    # 4) Train the original model
    fts = torch.from_numpy(X_tr).float().to(device)
    lbls = torch.tensor(y_tr, dtype=torch.long, device=device)
    model = HyperGCN(X_tr.shape[0], list(hyper_tr.values()), X_tr, args).to(device)
    model.structure = A_before
    for layer in model.layers:
        layer.reapproximate = False

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=args.milestones,
                                               gamma=args.gamma)
    criterion = nn.NLLLoss()

    print("—— Training base model ——")
    model = train_model(model, criterion, optimizer, scheduler,
                        fts, lbls,
                        num_epochs=args.epochs,
                        print_freq=args.log_every)

    # 5) Column-level unlearning: delete features & build new A
    X_zero, hyper_zero, A_zero = delete_feature_columns_hgcn(
        X_tensor=fts,
        transformer=transformer,
        column_names=cols,
        hyperedges=hyper_tr,
        mediators=args.mediators,
        use_cuda=(device.type=='cuda')
    )
    A_zero = A_zero.to(device)

    # 6) Retrain the unlearned model
    fts_zero = torch.from_numpy(X_zero.cpu().numpy() if torch.is_tensor(X_zero) else X_zero).float().to(device)
    model_un = HyperGCN(X_zero.shape[0], list(hyper_zero.values()), X_zero, args).to(device)
    model_un.structure = A_zero
    for layer in model_un.layers:
        layer.reapproximate = False

    opt_un = optim.Adam(model_un.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sch_un = optim.lr_scheduler.MultiStepLR(opt_un,
                                            milestones=args.milestones,
                                            gamma=args.gamma)
    print("—— Retraining unlearned model ——")
    model_un = train_model(model_un, criterion, opt_un, sch_un,
                           fts_zero, lbls,
                           num_epochs=args.epochs,
                           print_freq=args.log_every)

    # 7) GIF update
    feat_names = transformer.get_feature_names_out()
    deleted_idx = [i for i, name in enumerate(feat_names)
                   if any(c in name for c in cols)]
    gif_time = approx_gif(
        model, {"lap": fts, "y": lbls},
        A_before, A_zero,
        deleted_column=deleted_idx,
        x_before=fts, x_after=fts_zero,
        iters=args.gif_iters, damp=args.gif_damp, scale=args.gif_scale
    )
    print(f"[GIF] time={gif_time:.2f}s")

    # 8) Test set evaluation: delete columns & re-evaluate
    hyper_te = generate_hyperedge_dict(
        df_te_proc, args.cat_cols, args.cont_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        device=device
    )
    X_te_zero, hyper_te_zero, A_te_zero = delete_feature_columns_hgcn(
        X_tensor=torch.from_numpy(X_te).float().to(device),
        transformer=transformer,
        column_names=cols,
        hyperedges=hyper_te,
        mediators=args.mediators,
        use_cuda=(device.type=='cuda')
    )
    fts_te_zero = torch.from_numpy(
        X_te_zero.cpu().numpy() if torch.is_tensor(X_te_zero) else X_te_zero
    ).float().to(device)
    model_un.structure = A_te_zero.to(device)


    # —— Evaluate the GIF-updated model here ——
    # (You have already constructed fts_te_zero and A_te_zero below)
    model.structure = A_te_zero.to(device)
    test_obj = {"lap": fts_te_zero, "y": torch.tensor(y_te, device=device)}
    acc_gif = evaluate_hgcn_acc(model, test_obj)
    f1_gif  = evaluate_hgcn_f1(model, test_obj)
    print(f"[GIF Unlearned] Test Acc: {acc_gif:.4f}, F1: {f1_gif:.4f}")


    print("— After Column-Unlearn on Test —")
    test_obj = {"lap": fts_te_zero, "y": torch.tensor(y_te, device=device)}
    print(f"F1={evaluate_hgcn_f1(model_un, test_obj):.4f}, "
          f"Acc={evaluate_hgcn_acc(model_un, test_obj):.4f}")

if __name__ == "__main__":
    # main()
    for run in range(1,5):
        print("=== Run",run,"===")
        main()