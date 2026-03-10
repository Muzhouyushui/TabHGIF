import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score
# Data tools
from bank.HGAT.data_preprocessing_bank_retrain import (
    preprocess_node_features,
    generate_hyperedge_dict
)
# HGAT model
from bank.HGAT.HGAT_new import HGAT_JK
# Common utils
from utils.common_utils import evaluate_test_f1, evaluate_test_acc
# GIF unlearning (in-place)
from bank.HGAT.GIF_HGAT_ROW_NEI import approx_gif, rebuild_structure_after_node_deletion,train_model
# MIA
from bank.HGAT.MIA_HGAT import train_shadow_model, membership_inference_hgat
import pandas as pd
from sklearn.model_selection import train_test_split
from config import get_args

def build_incidence_matrix(hyperedges: dict, num_nodes: int, device=None) -> torch.Tensor:
    """
    Build the sparse hypergraph incidence matrix H: E x N
    """
    H = torch.zeros((len(hyperedges), num_nodes), dtype=torch.float32, device=device)
    for i, nodes in enumerate(hyperedges.values()):
        H[i, nodes] = 1.0
    return H.to_sparse()


def main():
    # Arguments and device
    args = get_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {device}")

    # Data loading and splitting
    df_full = pd.read_csv(args.data_csv, sep=';', skipinitialspace=True)
    df_train, df_test = train_test_split(
        df_full,
        test_size=args.split_ratio,
        stratify=df_full[args.label_col],
        random_state=21
    )
    df_train = df_train.reset_index(drop=True)
    df_test  = df_test.reset_index(drop=True)
    print(f"TRAIN={len(df_train)} samples, TEST={len(df_test)} samples")

    # Preprocessing
    X_train, y_train, df_train_proc, transformer = preprocess_node_features(
        df_train, is_test=False
    )
    X_test, y_test, df_test_proc, _ = preprocess_node_features(
        df_test, is_test=True, transformer=transformer
    )

    # Build hyperedges & H
    train_edges = generate_hyperedge_dict(
        df_train_proc,
        args.cat_cols,
        args.cont_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        device=device
    )
    test_edges = generate_hyperedge_dict(
        df_test_proc,
        args.cat_cols,
        args.cont_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        device=device
    )
    H_train = build_incidence_matrix(train_edges, len(X_train), device)
    H_test  = build_incidence_matrix(test_edges,  len(X_test),  device)

    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.long,    device=device)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32, device=device)
    y_test_t  = torch.tensor(y_test,  dtype=torch.long,    device=device)

    # Model and training
    num_classes = int(y_train_t.max().item() + 1)
    model = HGAT_JK(
        in_dim=X_train_t.size(1),
        hidden_dim=args.hidden_dim,
        out_dim=num_classes,
        dropout=args.dropout,
        alpha=0.5,
        num_layers=3,
        use_jk=False
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[args.epochs//2, args.epochs*3//4],
        gamma=0.1
    )

    print("—— Training HGAT baseline ——")
    model = train_model(
        model, criterion, optimizer, scheduler,
        X_train_t, y_train_t, H_train,
        num_epochs=args.epochs, print_freq=args.log_every
    )

    # Test evaluation
    model.eval()
    with torch.no_grad():
        logits = model(X_test_t, H_test)
        preds  = logits.argmax(dim=1)
        acc    = accuracy_score(y_test_t.cpu(), preds.cpu())
        f1     = f1_score(y_test_t.cpu(), preds.cpu(), average='micro')
    print(f"Baseline HGAT test acc: {acc:.4f}, f1: {f1:.4f}")

if __name__ == "__main__":
    main()