import torch
import torch.nn as nn
import torch.optim as optim

# Config and utils
from config import get_args
from utils.common_utils import evaluate_test_acc, evaluate_test_f1

# User-defined preprocessing and hyperedge functions
from database.data_preprocessing.data_preprocessing_delete_column_retrain import (
    preprocess_node_features,
    generate_hyperedge_dict,
    build_incidence_matrix
)

# HGNN imports
from HGNNs_Model.HGNN.HGNN_2 import HGNN_implicit, build_incidence_matrix, compute_degree_vectors
from GIF.GIF_HGNN_COL import train_model as train_gif_hgnn

# HGNN+ imports
from HGNNs_Model.HGNNP import HGNNP_implicit
from GIF.GIF_HGNNP_COL import train_model as train_gif_hgnnp

import time
def retrain_column_hgnn(args, column_name: str):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    t1=time.time()
    X_tr, y_tr, df_tr, transformer = preprocess_node_features(
        args.train_csv, is_test=False, transformer=None, ignore_cols=[column_name]
    )
    hyperedges_tr = generate_hyperedge_dict(
        df_tr, args.cat_cols, args.continuous_cols,
        ignore_cols=[column_name], max_nodes_per_hyperedge=args.max_nodes_per_hyperedge_train,
        device=device
    )
    H_sp_tr = build_incidence_matrix(hyperedges_tr, X_tr.shape[0])
    dv_tr, de_tr = compute_degree_vectors(H_sp_tr)
    H_coo_tr = H_sp_tr.tocoo()
    idx_tr = torch.LongTensor([H_coo_tr.row, H_coo_tr.col])
    H_tr = torch.sparse_coo_tensor(idx_tr, torch.FloatTensor(H_coo_tr.data), H_coo_tr.shape).to(device)

    fts_tr = torch.FloatTensor(X_tr).to(device)
    lbls_tr = torch.LongTensor(y_tr).to(device)

    model = HGNN_implicit(
        fts_tr.shape[1], len(torch.unique(lbls_tr)), args.hidden_dim, dropout=args.dropout
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    criterion = nn.NLLLoss()

    model = train_gif_hgnn(
        model, criterion, optimizer, scheduler,
        fts_tr, lbls_tr, H_tr, torch.FloatTensor(dv_tr).to(device), torch.FloatTensor(de_tr).to(device),
        num_epochs=args.epochs, print_freq=args.log_every
    )
    t2=time.time()
    print("retrain time=",t2-t1)
    train_data = {
        'x': fts_tr,
        'H': H_tr,
        'dv_inv': torch.FloatTensor(dv_tr).to(device),
        'de_inv': torch.FloatTensor(de_tr).to(device),
        'y': lbls_tr
    }
    train_acc = evaluate_test_acc(model, train_data)
    train_f1 = evaluate_test_f1(model, train_data)
    print(f"[HGNN Retrain] Train Acc: {train_acc:.4f}, F1: {train_f1:.4f}")

    X_te, y_te, df_te, _ = preprocess_node_features(
        args.test_csv, is_test=True, transformer=transformer, ignore_cols=[column_name]
    )
    hyperedges_te = generate_hyperedge_dict(
        df_te, args.cat_cols, args.continuous_cols,
        ignore_cols=[column_name], max_nodes_per_hyperedge=args.max_nodes_per_hyperedge_test,
        device=device
    )
    H_sp_te = build_incidence_matrix(hyperedges_te, X_te.shape[0])
    dv_te, de_te = compute_degree_vectors(H_sp_te)
    H_coo_te = H_sp_te.tocoo()
    idx_te = torch.LongTensor([H_coo_te.row, H_coo_te.col])
    H_te = torch.sparse_coo_tensor(idx_te, torch.FloatTensor(H_coo_te.data), H_coo_te.shape).to(device)

    fts_te = torch.FloatTensor(X_te).to(device)
    lbls_te = torch.LongTensor(y_te).to(device)
    test_data = {
        'x': fts_te,
        'H': H_te,
        'dv_inv': torch.FloatTensor(dv_te).to(device),
        'de_inv': torch.FloatTensor(de_te).to(device),
        'y': lbls_te
    }
    test_acc = evaluate_test_acc(model, test_data)
    test_f1 = evaluate_test_f1(model, test_data)
    print(f"[HGNN Retrain] Test  Acc: {test_acc:.4f}, F1: {test_f1:.4f}")

    return model, transformer


def retrain_column_hgnn_plus(args, column_name: str):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    t1=time.time()
    X_tr, y_tr, df_tr, transformer = preprocess_node_features(
        args.train_csv, is_test=False, transformer=None, ignore_cols=[column_name]
    )
    hyperedges_tr = generate_hyperedge_dict(
        df_tr, args.cat_cols, args.continuous_cols,
        ignore_cols=[column_name], max_nodes_per_hyperedge=args.max_nodes_per_hyperedge_train,
        device=device
    )
    H_sp_tr = build_incidence_matrix(hyperedges_tr, X_tr.shape[0])
    dv_tr, de_tr = compute_degree_vectors(H_sp_tr)
    H_coo_tr = H_sp_tr.tocoo()
    idx_tr = torch.LongTensor([H_coo_tr.row, H_coo_tr.col])
    H_tr = torch.sparse_coo_tensor(idx_tr, torch.FloatTensor(H_coo_tr.data), H_coo_tr.shape).to(device)

    fts_tr = torch.FloatTensor(X_tr).to(device)
    lbls_tr = torch.LongTensor(y_tr).to(device)
    model = HGNNP_implicit(
        fts_tr.shape[1], len(torch.unique(lbls_tr)), args.hidden_dim, dropout=args.dropout
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    criterion = nn.NLLLoss()

    model = train_gif_hgnnp(
        model, criterion, optimizer, scheduler,
        fts_tr, lbls_tr, H_tr, torch.FloatTensor(dv_tr).to(device), torch.FloatTensor(de_tr).to(device),
        num_epochs=args.epochs, print_freq=args.log_every
    )
    t2=time.time()
    print("retrain time=",t2-t1)
    train_data = {
        'x': fts_tr,
        'H': H_tr,
        'dv_inv': torch.FloatTensor(dv_tr).to(device),
        'de_inv': torch.FloatTensor(de_tr).to(device),
        'y': lbls_tr
    }
    train_acc = evaluate_test_acc(model, train_data)
    train_f1 = evaluate_test_f1(model, train_data)
    print(f"[HGNN+ Retrain] Train Acc: {train_acc:.4f}, F1: {train_f1:.4f}")

    X_te, y_te, df_te, _ = preprocess_node_features(
        args.test_csv, is_test=True, transformer=transformer, ignore_cols=[column_name]
    )
    hyperedges_te = generate_hyperedge_dict(
        df_te, args.cat_cols, args.continuous_cols,
        ignore_cols=[column_name], max_nodes_per_hyperedge=args.max_nodes_per_hyperedge_test,
        device=device
    )
    H_sp_te = build_incidence_matrix(hyperedges_te, X_te.shape[0])
    dv_te, de_te = compute_degree_vectors(H_sp_te)
    H_coo_te = H_sp_te.tocoo()
    idx_te = torch.LongTensor([H_coo_te.row, H_coo_te.col])
    H_te = torch.sparse_coo_tensor(idx_te, torch.FloatTensor(H_coo_te.data), H_coo_te.shape).to(device)

    fts_te = torch.FloatTensor(X_te).to(device)
    lbls_te = torch.LongTensor(y_te).to(device)
    test_data = {
        'x': fts_te,
        'H': H_te,
        'dv_inv': torch.FloatTensor(dv_te).to(device),
        'de_inv': torch.FloatTensor(de_te).to(device),
        'y': lbls_te
    }
    test_acc = evaluate_test_acc(model, test_data)
    test_f1 = evaluate_test_f1(model, test_data)
    print(f"[HGNN+ Retrain] Test  Acc: {test_acc:.4f}, F1: {test_f1:.4f}")

    return model, transformer


if __name__ == "__main__":
    args = get_args()
    col = args.columns_to_unlearn[0]

    for run in range(1, 5):
        print(f"=== Run {run} ===")
        retrain_column_hgnn(args, col)
        retrain_column_hgnn_plus(args, col)