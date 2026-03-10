# ───────────────────────────────────────────────────────────────
#  train_unlearning_HGCN_COLUMN.py （关键 main 函数节选）
#  依赖：config_HGCN / utils / data_preprocessing_column / GIF.GIF_HGCN_ROW
# ───────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

from Credit.HGCN.config import get_args
from Credit.HGCN.data_preprocessing_Credit_col import (
    preprocess_node_features, generate_hyperedge_dict,
    delete_feature_columns_hgcn
)
from Credit.HGCN.HGCN_utils import evaluate_hgcn_f1,evaluate_hgcn_acc
from Credit.HGCN.GIF_HGCN_COL_Credit import approx_gif, train_model
from Credit.HGCN.HGCN import HyperGCN, laplacian
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter

# ==============================================================
#                       Main  (re-organised)
# ==============================================================


def main():
    args = get_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {device}")

    # 1) Load & clean raw DataFrame
    df = pd.read_csv(args.train_csv, header=None, na_values='?')
    df = df.reset_index(drop=True)

    # 2) Split train/test, stratify on last column (class)
    df_tr, df_te = train_test_split(
        df,
        test_size=args.split_ratio,
        random_state=21,
        stratify=df.iloc[:, -1]
    )
    df_tr = df_tr.reset_index(drop=True)
    df_te = df_te.reset_index(drop=True)
    print(f"TRAIN samples: {len(df_tr)}, TEST samples: {len(df_te)}")
    print("– TRAIN label dist:", Counter(df_tr.iloc[:, -1]))
    print("– TEST  label dist:", Counter(df_te.iloc[:, -1]))

    # 3) Preprocess features
    X_tr, y_tr, df_tr_proc, transformer = preprocess_node_features(
        data=df_tr,
        transformer=None
    )
    X_te, y_te, df_te_proc, _ = preprocess_node_features(
        data=df_te,
        transformer=transformer
    )
    print(f"➤ TRAIN: X_tr={X_tr.shape}, y_tr={y_tr.shape}")
    print(f"➤ TEST:  X_te={X_te.shape}, y_te={y_te.shape}")

    # 4) Generate hyperedges
    feature_cols = args.cat_cols + args.cont_cols
    hyper_tr = generate_hyperedge_dict(
        df=df_tr_proc,
        feature_cols=feature_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        device=device
    )
    hyper_te = generate_hyperedge_dict(
        df=df_te_proc,
        feature_cols=feature_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        device=device
    )

    # 5) Train base HyperGCN model
    fts = torch.from_numpy(X_tr).float().to(device)
    lbls = torch.tensor(y_tr, dtype=torch.long, device=device)
    cfg = lambda: None
    cfg.d, cfg.c        = X_tr.shape[1], int(max(y_tr)) + 1
    cfg.depth, cfg.hidden = args.depth, args.hidden_dim
    cfg.dropout, cfg.fast  = args.dropout, args.fast
    cfg.mediators, cfg.cuda = args.mediators, args.cuda
    cfg.dataset = getattr(args, 'dataset', '')

    model = HyperGCN(
        X_tr.shape[0],
        list(hyper_tr.values()),
        X_tr,
        cfg
    ).to(device)
    A_before = laplacian(list(hyper_tr.values()), X_tr, args.mediators).to(device)
    model.structure = A_before
    for layer in model.layers:
        layer.reapproximate = False

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    criterion = torch.nn.NLLLoss()
    # 1. 明确标签列名
    print("所有列名：", df.columns.tolist())

    label_col = 15
    drop_col= 4




    print("—— Training base model ——")
    model = train_model(
        model, criterion, optimizer, scheduler,
        fts, lbls,
        num_epochs=args.epochs,
        print_freq=args.log_every
    )
    data_train = {"lap": fts, "y": lbls}

    # 6) Column-level Unlearning on train set
    cols = args.columns_to_unlearn
    he_tr_before = len(hyper_tr)
    X_zero, hyper_zero, A_zero = delete_feature_columns_hgcn(
        X_tensor=fts,
        transformer=transformer,
        column_names=cols,
        hyperedges=hyper_tr,
        df_proc=df_tr_proc,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        mediators=args.mediators,
        use_cuda=(device.type == 'cuda')
    )
    he_tr_after = len(hyper_zero)
    print(f"[Train Hyperedges] before: {he_tr_before}, after: {he_tr_after}, removed: {he_tr_before - he_tr_after}")

    fts_zero = X_zero.float().to(device) if isinstance(X_zero, torch.Tensor) else torch.from_numpy(X_zero).float().to(device)

    # 7) Retrain unlearned model
    model_un = HyperGCN(
        X_zero.shape[0],
        list(hyper_zero.values()),
        X_zero,
        cfg
    ).to(device)
    model_un.structure = A_zero.to(device)
    for layer in model_un.layers:
        layer.reapproximate = False

    opt_un = optim.Adam(model_un.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sch_un = optim.lr_scheduler.MultiStepLR(opt_un, milestones=args.milestones, gamma=args.gamma)
    print("—— Training unlearned model ——")
    model_un = train_model(
        model_un, criterion, opt_un, sch_un,
        fts_zero, lbls,
        num_epochs=args.epochs,
        print_freq=args.log_every
    )

    # 8) GIF update
    feat_names = transformer.get_feature_names_out()
    deleted_idx = [i for i, name in enumerate(feat_names) if any(c in name for c in cols)]
    gif_time = approx_gif(
        model, data_train,
        A_before, A_zero,
        deleted_column=deleted_idx,
        x_before=fts, x_after=fts_zero,
        iters=args.gif_iters,
        damp=args.gif_damp,
        scale=args.gif_scale
    )
    print(f"[GIF] time={gif_time:.2f}s")

    # 9) Column-level Unlearning & evaluation on test set
    he_te_before = len(hyper_te)
    X_te_zero, hyper_te_zero, A_te_zero = delete_feature_columns_hgcn(
        X_tensor=torch.from_numpy(X_te).float().to(device),
        transformer=transformer,
        column_names=cols,
        hyperedges=hyper_te,
        df_proc=df_te_proc,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        mediators=args.mediators,
        use_cuda=(device.type == 'cuda')
    )
    he_te_after = len(hyper_te_zero)
    print(f"[Test  Hyperedges] before: {he_te_before}, after: {he_te_after}, removed: {he_te_before - he_te_after}")

    fts_te_zero = X_te_zero.float().to(device) if isinstance(X_te_zero, torch.Tensor) else torch.from_numpy(X_te_zero).float().to(device)
    model_un.structure = A_te_zero.to(device)

    print("— After Column-Unlearn on Test —")
    test_obj = {"lap": fts_te_zero, "y": torch.tensor(y_te, dtype=torch.long, device=device)}
    print(f"F1={evaluate_hgcn_f1(model_un, test_obj):.4f}, Acc={evaluate_hgcn_acc(model_un, test_obj):.4f}")

if __name__ == "__main__":
    # main()
    for run in range(1,5):
        print("=== Run",run,"===")
        main()
