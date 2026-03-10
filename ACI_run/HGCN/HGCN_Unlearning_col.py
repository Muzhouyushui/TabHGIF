# ───────────────────────────────────────────────────────────────
#  train_unlearning_HGCN_COLUMN.py （关键 main 函数节选）
#  依赖：config_HGCN / utils / data_preprocessing_column / GIF.GIF_HGCN_ROW
# ───────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.optim as optim

from config_HGCN import get_args
from database.data_preprocessing.data_preprocessing_column import (
    preprocess_node_features, generate_hyperedge_dict,
    delete_feature_columns_hgcn
)
from utils.common_utils import evaluate_hgcn_f1,evaluate_hgcn_acc
from GIF.GIF_HGCN_COL import approx_gif, train_model
from HGCN.HyperGCN import HyperGCN, laplacian

# ==============================================================
#                       Main  (re-organised)
# ==============================================================

def main():
    args   = get_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {device}")

    remove_ratio = args.remove_ratio

    X_tr, y_tr, df_tr, transformer = preprocess_node_features(
        args.train_csv, is_test=False
    )

    hyperedges_tr = generate_hyperedge_dict(
        df_tr, args.categate_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        device=device
    )
    hyperedges_tr_list = list(generate_hyperedge_dict(
        df_tr, args.categate_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        device=device
    ).values())


    fts  = torch.from_numpy(X_tr).float().to(device)
    lbls = torch.tensor(y_tr, dtype=torch.long).to(device)
    train_mask = torch.ones(len(y_tr), dtype=torch.bool, device=device)
    data_train = {"x": fts, "y": lbls, "train_mask": train_mask}

    cfg = lambda: None
    cfg.d         = X_tr.shape[1]
    cfg.depth     = args.depth
    cfg.c         = int(lbls.max().item() + 1)
    cfg.dropout   = args.dropout
    cfg.fast      = args.fast
    cfg.mediators = args.mediators
    cfg.cuda      = args.cuda
    cfg.dataset   = args.dataset

    model = HyperGCN(
        num_nodes=X_tr.shape[0],
        edge_list=hyperedges_tr_list,
        X_init=X_tr,
        args=cfg
    ).to(device)

    A_before = laplacian(hyperedges_tr_list, X_tr, args.mediators).to(device)
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

    A_tr_before = A_before

    X_te, y_te_raw, df_te, _ = preprocess_node_features(
        args.test_csv, is_test=True, transformer=transformer
    )
    fts_te = torch.from_numpy(X_te).float().to(device)
    lbls_te = torch.tensor(y_te_raw, dtype=torch.long).to(device)
    data_test = {"x": fts_te, "y": lbls_te}

    hyperedges_te = list(generate_hyperedge_dict(
        df_te, args.categate_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        device=device
    ).values())
    A_te = laplacian(hyperedges_te, X_te, args.mediators).to(device)
    model.structure = A_te
    for layer in model.layers:
        layer.reapproximate = False

    print("— Before Unlearning —")
    print(f" Test F1: {evaluate_hgcn_f1(model, data_test):.4f}, "
          f"Acc: {evaluate_hgcn_acc(model, data_test):.4f}")

    column_to_delete = args.columns_to_unlearn
    X_tensor_after, hyperedges_after, A_after = delete_feature_columns_hgcn(
        X_tensor=fts,
        transformer=transformer,
        column_names=column_to_delete,
        hyperedges=hyperedges_tr,
        mediators=args.mediators,
        use_cuda=(device.type == 'cuda')
    )
    feat_names = transformer.get_feature_names_out()
    deleted_columns = [
        i for i, name in enumerate(feat_names)
        if column_to_delete in name.split("_")
    ]
    runtime = approx_gif(
        model,
        data_train,
        A_before=A_tr_before,
        A_after=A_after,
        deleted_column=deleted_columns,
        x_before=fts,
        x_after=X_tensor_after,
        iters=args.if_iters,
        damp=args.if_damp,
        scale=args.if_scale
    )


    edge_dict_te = generate_hyperedge_dict(
        df_te, args.categate_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        device=device
    )

    fts_te_after, edge_dict_te, A_test_after = delete_feature_columns_hgcn(
        X_tensor=fts_te,
        transformer=transformer,
        column_names=column_to_delete,
        hyperedges=edge_dict_te,
        mediators=args.mediators,
        use_cuda=(device.type == 'cuda')
    )


    model.structure = A_test_after
    for layer in model.layers:
        layer.reapproximate = False


    print("—— After Column-Unlearning on Test ——")
    print(f"Test F1: {evaluate_hgcn_f1(model, {'x': fts_te_after, 'y': lbls_te}):.4f}, "
          f"Acc: {evaluate_hgcn_acc(model, {'x': fts_te_after, 'y': lbls_te}):.4f}")

if __name__ == "__main__":
    main()
