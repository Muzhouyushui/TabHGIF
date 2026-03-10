
import torch
import torch.nn as nn
import torch.optim as optim

from config_HGCN import get_args
from database.data_preprocessing.data_preprocessing_delete_column_retrain import (
    preprocess_node_features, generate_hyperedge_dict,
)
from utils.common_utils import evaluate_hgcn_f1,evaluate_hgcn_acc
from GIF.GIF_HGCN_COL import train_model
from HGCN.HyperGCN import HyperGCN, laplacian
import time

def main():
    args = get_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {device}")
    t0 = time.time()
    remove_ratio = args.remove_ratio
    print("数据加载与预处理")
    X_tr, y_tr, df_tr, transformer = preprocess_node_features(
        args.train_csv, is_test=False,ignore_cols=args.columns_to_unlearn
    )
    hyperedges_tr = list(generate_hyperedge_dict(
        df_tr, args.categate_cols,
        continuous_cols=args.categate_cols,
        ignore_cols=args.columns_to_unlearn,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        device=device
    ).values())

    fts = torch.from_numpy(X_tr).float().to(device)
    lbls = torch.tensor(y_tr, dtype=torch.long).to(device)
    train_mask = torch.ones(len(y_tr), dtype=torch.bool, device=device)
    data_train = {"x": fts, "y": lbls, "train_mask": train_mask}

    cfg = lambda: None
    cfg.d = X_tr.shape[1]
    cfg.depth = args.depth
    cfg.c = int(lbls.max().item() + 1)
    cfg.dropout = args.dropout
    cfg.fast = args.fast
    cfg.mediators = args.mediators
    cfg.cuda = args.cuda
    cfg.dataset = args.dataset

    model = HyperGCN(
        num_nodes=X_tr.shape[0],
        edge_list=hyperedges_tr,
        X_init=X_tr,
        args=cfg
    ).to(device)

    A_before = laplacian(hyperedges_tr, X_tr, args.mediators).to(device)
    model.structure = A_before
    for layer in model.layers:
        layer.reapproximate = False

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    criterion = nn.NLLLoss()
    model = train_model(
        model, criterion, optimizer, scheduler,
        fts, lbls,
        num_epochs=args.epochs,
        print_freq=args.log_every
    )
    t1 = time.time()

    X_te, y_te_raw, df_te, _ = preprocess_node_features(
        args.test_csv, is_test=True, transformer=transformer,ignore_cols=args.columns_to_unlearn
    )
    fts_te = torch.from_numpy(X_te).float().to(device)
    lbls_te = torch.tensor(y_te_raw, dtype=torch.long).to(device)
    data_test = {"x": fts_te, "y": lbls_te}

    hyperedges_te = list(generate_hyperedge_dict(
        df_te, args.categate_cols,continuous_cols=None,
        ignore_cols=args.columns_to_unlearn,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        device=device
    ).values())
    A_te = laplacian(hyperedges_te, X_te, args.mediators).to(device)
    model.structure = A_te
    for layer in model.layers:
        layer.reapproximate = False
    retrain_time = t1 - t0
    print(f"[Retrain Time] {retrain_time:.4f} s")

    print("— Before Unlearning —")
    print(f" Test F1: {evaluate_hgcn_f1(model, data_test):.4f}, "
          f"Acc: {evaluate_hgcn_acc(model, data_test):.4f}")

if __name__ == "__main__":
    main()

