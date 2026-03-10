import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split

from Credit.HGCN.config import get_args
from Credit.HGCN.data_proprocessing_Credit_col_retrain import (
    preprocess_node_features, generate_hyperedge_dict
)
from Credit.HGCN.HGCN_utils import evaluate_hgcn_f1, evaluate_hgcn_acc
from Credit.HGCN.GIF_HGCN_COL_Credit import train_model
from Credit.HGCN.HGCN import HyperGCN, laplacian


def main():
    args = get_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {device}")

    # 1) Load & split raw data
    df = pd.read_csv(args.train_csv, header=None, na_values='?', skipinitialspace=True)
    df.columns = [f"A{i}" for i in range(1,16)] + ["class"]
    df = df.dropna(subset=["class"]).reset_index(drop=True)
    df_tr, df_te = train_test_split(df, test_size=args.split_ratio, stratify=df['class'], random_state=21)
    df_tr, df_te = df_tr.reset_index(drop=True), df_te.reset_index(drop=True)
    print(f"TRAIN samples: {len(df_tr)}, TEST samples: {len(df_te)}")
    print("– TRAIN dist:", Counter(df_tr['class']))
    print("– TEST  dist:", Counter(df_te['class']))

    # columns to drop for unlearning
    drop_cols = args.columns_to_unlearn

    # 2) Preprocess with column drop
    X_tr, y_tr, df_tr_proc, transformer = preprocess_node_features(
        data=df_tr, transformer=None, drop_cols=drop_cols
    )
    X_te, y_te, df_te_proc, _ = preprocess_node_features(
        data=df_te, transformer=transformer, drop_cols=drop_cols
    )
    print(f"➤ TRAIN: X_tr={X_tr.shape}, y_tr={y_tr.shape}")
    print(f"➤ TEST : X_te={X_te.shape}, y_te={y_te.shape}")

    # 3) Generate hyperedges ignoring dropped columns
    feature_cols = [c for c in args.cat_cols + args.cont_cols if c not in drop_cols]
    hyper_tr = generate_hyperedge_dict(df_tr_proc, feature_cols=feature_cols,
                                       max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
                                       device=device, drop_cols=drop_cols)
    hyper_te = generate_hyperedge_dict(df_te_proc, feature_cols=feature_cols,
                                       max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
                                       device=device, drop_cols=drop_cols)
    print(f"Hyperedges: train={len(hyper_tr)}, test={len(hyper_te)}")

    # 4) Train baseline HyperGCN
    fts_tr = torch.from_numpy(X_tr).float().to(device)
    lbls_tr = torch.tensor(y_tr, dtype=torch.long, device=device)

    cfg = lambda: None
    cfg.d, cfg.c = X_tr.shape[1], int(y_tr.max()) + 1
    cfg.depth, cfg.hidden = args.depth, args.hidden_dim
    cfg.dropout, cfg.fast = args.dropout, args.fast
    cfg.mediators, cfg.cuda = args.mediators, args.cuda
    cfg.dataset = getattr(args, 'dataset', '')
    model = HyperGCN(X_tr.shape[0], list(hyper_tr.values()), X_tr, cfg).to(device)
    A_tr = laplacian(list(hyper_tr.values()), X_tr, args.mediators).to(device)
    model.structure = A_tr
    for layer in model.layers: layer.reapproximate = False

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    criterion = nn.NLLLoss()
    print("—— Training baseline model ——")
    model = train_model(model, criterion, optimizer, scheduler,
                        fts_tr, lbls_tr,
                        num_epochs=args.epochs, print_freq=args.log_every)

    # 5) Evaluate on test set
    fts_te = torch.from_numpy(X_te).float().to(device)
    lbls_te = torch.tensor(y_te, dtype=torch.long, device=device)
    A_te = laplacian(list(hyper_te.values()), X_te, args.mediators).to(device)
    model.structure = A_te
    for layer in model.layers: layer.reapproximate = False

    print("—— Evaluating baseline model ——")
    test_obj = {"lap": fts_te, "y": lbls_te}
    acc = evaluate_hgcn_acc(model, test_obj)
    f1 = evaluate_hgcn_f1(model, test_obj)
    print(f"Baseline HyperGCN Test — Acc: {acc:.4f}, F1: {f1:.4f}")

if __name__ == "__main__":
    for run in range(1,5):
        print(f"=== Run {run} ===")
        main()
