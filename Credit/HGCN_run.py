import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

from GIF.GIF_HGCN_ROW import train_model
from utils.common_utils import evaluate_hgcn_f1, evaluate_hgcn_acc
from Credit.data_preprocessing_credit import (
    preprocess_node_features,
    generate_hyperedge_dict
)
from Credit.HGCN import HyperGCN, laplacian
from paths import CREDIT_DATA

def main():
    # 0) Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # 1) Load & preprocess the full dataset (without passing label_col)
    csv_path = CREDIT_DATA
    X_full, y_full, df_full, transformer = preprocess_node_features(
        data=csv_path,
        transformer=None
    )

    # 2) Split train / test
    df_tr, df_te, y_tr, y_te = train_test_split(
        df_full, y_full,
        test_size=0.2,
        stratify=y_full,
        random_state=42
    )
    df_tr = df_tr.reset_index(drop=True)
    df_te = df_te.reset_index(drop=True)

    # 3) Reuse the transformer to preprocess train / test again
    X_tr, y_tr, df_tr_proc, _ = preprocess_node_features(
        data=df_tr,
        transformer=transformer
    )
    X_te, y_te, df_te_proc, _ = preprocess_node_features(
        data=df_te,
        transformer=transformer
    )

    # 4) Construct hyperedges
    #    (only pass df, max_nodes_per_hyperedge, and device)
    hyper_tr = generate_hyperedge_dict(
        df=df_tr_proc,
        max_nodes_per_hyperedge=30,
        device=device
    )
    hyper_te = generate_hyperedge_dict(
        df=df_te_proc,
        max_nodes_per_hyperedge=30,
        device=device
    )
    he_tr_list = list(hyper_tr.values())
    he_te_list = list(hyper_te.values())

    # 5) Convert to tensors
    fts_tr  = torch.from_numpy(X_tr).float().to(device)
    lbls_tr = torch.tensor(y_tr, dtype=torch.long, device=device)
    fts_te  = torch.from_numpy(X_te).float().to(device)
    lbls_te = torch.tensor(y_te, dtype=torch.long, device=device)

    # 6) Loss function (label-balanced weighting)
    counts = torch.bincount(lbls_tr, minlength=2).float()
    crit   = nn.NLLLoss(weight=(1.0 / counts).to(device))

    # 7) Model & configuration
    class CFG: pass
    cfg = CFG()
    cfg.d         = X_tr.shape[1]
    cfg.depth     = 3
    cfg.c         = len(torch.unique(lbls_tr))
    cfg.hidden    = 80
    cfg.dropout   = 0.1
    cfg.fast      = False
    cfg.mediators = False
    cfg.cuda      = torch.cuda.is_available()

    model = HyperGCN(
        num_nodes = X_tr.shape[0],
        edge_list = he_tr_list,
        X_init    = fts_tr,
        args      = cfg
    ).to(device)

    # 8) Compute Laplacian on the training set and train
    A_tr = laplacian(he_tr_list, fts_tr.cpu(), cfg.mediators).to(device)
    model.structure = A_tr
    for layer in model.layers:
        layer.reapproximate = False

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[100, 150], gamma=0.1
    )

    model = train_model(
        model, crit, optimizer, scheduler,
        fts_tr, lbls_tr,
        num_epochs=120,
        print_freq=10
    )

    # 9) Switch to the test-set Laplacian and evaluate
    A_te = laplacian(he_te_list, fts_te.cpu(), cfg.mediators).to(device)
    model.structure = A_te
    for layer in model.layers:
        layer.reapproximate = False

    f1  = evaluate_hgcn_f1(model, {"x": fts_te, "y": lbls_te})
    ac  = evaluate_hgcn_acc(model, {"x": fts_te, "y": lbls_te})
    print("— Test Performance —")
    print(f" F1: {f1:.4f}, Acc: {ac:.4f}")

if __name__ == "__main__":
    main()