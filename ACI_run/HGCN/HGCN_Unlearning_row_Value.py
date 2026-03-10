
import numpy as np, torch

from GIF.GIF_HGCN_ROW_NEI import approx_gif,train_model,apply_node_deletion_unlearning,find_hyperneighbors
import torch.nn as nn
import torch.optim as optim

from utils.common_utils import evaluate_hgcn_f1,evaluate_hgcn_acc
from database.data_preprocessing.data_preprocessing_K import preprocess_node_features, generate_hyperedge_dict
from HGCN.HyperGCN import HyperGCN ,laplacian # 你自定义的 HyperGCN 模型
from config_HGCN import get_args
from MIA.MIA_HGCN import train_shadow_model, membership_inference_hgcn
from torch_geometric.data import Data
import time
def retrain_after_prune(
        X_tr, y_tr, df_tr, hyperedges_tr,
        deleted: torch.LongTensor,
        transformer, args, device,A_tr_before):

    all_idx  = torch.arange(X_tr.shape[0], device=device)
    keep_idx = all_idx[~torch.isin(all_idx, deleted)]

    X_keep = X_tr[keep_idx.cpu().numpy()]
    y_arr   = np.array(y_tr, dtype=int)
    y_keep  = y_arr[keep_idx.cpu().numpy()]
    df_keep = df_tr.loc[keep_idx.cpu().numpy()].reset_index(drop=True)

    hyper_keep_dict = generate_hyperedge_dict(
        df_keep,
        args.categate_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        device=device
    )
    hyper_keep_list = list(hyper_keep_dict.values())
    A_keep = laplacian(hyper_keep_list, X_keep, args.mediators).to(device)

    fts_keep  = torch.from_numpy(X_keep).float().to(device)
    lbls_keep = torch.tensor(y_keep, dtype=torch.long, device=device)
    mask_keep = torch.ones(len(y_keep), dtype=torch.bool, device=device)

    cfg = lambda: None
    cfg.d         = X_keep.shape[1]
    cfg.c         = int(lbls_keep.max().item()) + 1
    cfg.depth     = args.depth
    cfg.dropout   = args.dropout
    cfg.fast      = args.fast
    cfg.mediators = args.mediators
    cfg.cuda      = True
    cfg.dataset   = args.dataset

    model_re = HyperGCN(
        num_nodes=X_keep.shape[0],
        edge_list=hyper_keep_list,
        X_init=X_keep,
        args=cfg
    ).to(device)
    model_re.structure = A_keep
    for layer in model_re.layers:
        layer.reapproximate = False

    opt = optim.Adam(model_re.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sch = optim.lr_scheduler.MultiStepLR(opt, args.milestones, gamma=args.gamma)
    crit = nn.NLLLoss()
    model_re = train_model(
        model_re, crit, opt, sch,
        fts_keep, lbls_keep,
        num_epochs=args.epochs,
        print_freq=args.log_every
    )

    f1_rem = evaluate_hgcn_f1(model_re, {"x": fts_keep, "y": lbls_keep, "train_mask": mask_keep})
    acc_rem = evaluate_hgcn_acc(model_re, {"x": fts_keep, "y": lbls_keep, "train_mask": mask_keep})
    print(f"[Retrain] Keep‐set    F1={f1_rem:.4f}, Acc={acc_rem:.4f}")

    X_te, y_te, df_te, _ = preprocess_node_features(
        args.test_csv, is_test=True, transformer=transformer
    )
    fts_te = torch.from_numpy(X_te).float().to(device)
    lbls_te = torch.tensor(y_te, dtype=torch.long, device=device)
    hyper_te_dict = generate_hyperedge_dict(
        df_te, args.categate_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        device=device
    )
    A_te = laplacian(list(hyper_te_dict.values()), X_te, args.mediators).to(device)
    model_re.structure = A_te
    for layer in model_re.layers:
        layer.reapproximate = False

    f1_test = evaluate_hgcn_f1(model_re, {"x": fts_te, "y": lbls_te})
    acc_test = evaluate_hgcn_acc(model_re, {"x": fts_te, "y": lbls_te})
    print(f"[Retrain] Test‐set    F1={f1_test:.4f}, Acc={acc_test:.4f}")


    model_re.structure = A_tr_before
    for layer in model_re.layers:
        layer.reapproximate = False


    member_mask = np.zeros(len(y_tr), dtype=bool)
    member_mask[keep_idx.cpu().numpy()] = True


    _, (_, _), (auc_re, f1_re) = membership_inference_hgcn(
        X_train     = X_tr,
        y_train     = y_tr,           #
        hyperedges  = hyperedges_tr,  #
        target_model= model_re,       #
        args        = args,
        device      = device,
        member_mask = member_mask     #
    )
    print(f"[Retrain MIA] AUC={auc_re:.4f}, F1={f1_re:.4f}")

    return {
        'f1_rem':   f1_rem,
        'acc_rem':  acc_rem,
        'f1_test':  f1_test,
        'acc_test': acc_test,
        'mia_auc':  auc_re,
        'mia_f1':   f1_re
    }

def mia_with_hgcn(X_train, y_train, hyperedges, args, device):

    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X = np.asarray(X_train)
    y = np.asarray(y_train)
    N = X.shape[0]

    cfg = lambda: None
    cfg.d = X.shape[1]
    cfg.depth = args.depth
    cfg.c = int(y.max()) + 1
    cfg.dropout = args.dropout
    cfg.fast = args.fast
    cfg.mediators = args.mediators
    cfg.cuda = args.cuda
    cfg.dataset = args.dataset

    full_mask = torch.ones(N, dtype=torch.bool, device=device)
    data_full = Data(
        x=torch.from_numpy(X).float().to(device),
        y=torch.from_numpy(y).long().to(device),
        train_mask=full_mask
    )

    full_model = HyperGCN(
        num_nodes=N,
        edge_list=hyperedges,
        X_init=X,
        args=cfg
    ).to(device)
    full_model = train_shadow_model(
        full_model,
        data_full,
        lr=getattr(args, "full_lr", args.lr),
        epochs=getattr(args, "full_epochs", args.epochs)
    )

    attack_model, (_, _), (auc_tgt, f1_tgt) = membership_inference_hgcn(
        X_train=X,
        y_train=y,
        hyperedges=hyperedges,
        args=args,
        device=device,
        target_model=full_model,
        member_mask=None
    )

    return auc_tgt, f1_tgt


def main():
    args   = get_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {device}")

    remove_ratio = args.remove_ratio
    print("pre-ready")
    X_tr, y_tr, df_tr, transformer = preprocess_node_features(
        args.train_csv, is_test=False
    )
    hyperedges_tr = list(generate_hyperedge_dict(
        df_tr, args.categate_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        device=device
    ).values())

    fts  = torch.from_numpy(X_tr).float().to(device)
    lbls = torch.tensor(y_tr, dtype=torch.long).to(device)
    data_train = {"x": fts, "y": lbls, "train_mask": torch.ones(len(y_tr), dtype=torch.bool, device=device)}

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
        edge_list=hyperedges_tr,
        X_init=X_tr,
        args=cfg
    ).to(device)

    A_before = laplacian(hyperedges_tr, X_tr, args.mediators).to(device)
    model.structure = A_before
    for layer in model.layers:
        layer.reapproximate = False

    # —— 4) 训练 Full Model —— #
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

    print("— Membership Inference Attack on Full Model —")
    auc_full, f1_full = mia_with_hgcn(
        X_train    = X_tr,
        y_train    = y_tr,
        hyperedges = hyperedges_tr,
        args       = args,
        device     = device
    )
    print(f" Full Model MIA  AUC = {auc_full:.4f},  F1 = {f1_full:.4f}")

    X_te, y_te_raw, df_te, _ = preprocess_node_features(
        args.test_csv, is_test=True, transformer=transformer
    )
    y_te_fixed = [1 if v.strip().rstrip('.') == ">50K" else 0 for v in df_te["income"].values]
    fts_te   = torch.from_numpy(X_te).float().to(device)
    lbls_te  = torch.tensor(y_te_fixed, dtype=torch.long).to(device)
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


    df_tr = df_tr.reset_index(drop=True)

    mask = (df_tr['sex'] == 'Male') & (df_tr['relationship'] == 'Husband')

    deleted_idx_np = np.where(mask)[0]

    deleted = torch.tensor(deleted_idx_np, dtype=torch.long, device=device)

    print(f"Value-based deletion: {deleted.numel()} nodes in total, first 5 indices: {deleted_idx_np[:5]}")

    baseline = retrain_after_prune(
        X_tr, y_tr, df_tr, hyperedges_tr,
        deleted, transformer, args, device,A_tr_before
    )

    print("[Baseline Retraining]")
    print(f" Retained subset  F1/Acc = {baseline['f1_rem']:.4f}/{baseline['acc_rem']:.4f}")
    print(f" Test set         F1/Acc = {baseline['f1_test']:.4f}/{baseline['acc_test']:.4f}")


    t2 = time.perf_counter()
    deleted_list = deleted.cpu().tolist()
    K = getattr(args, "neighbor_K", 14)
    deleted_neighbors = find_hyperneighbors(
        hyperedges_tr,
        deleted_list,
        K
    )
    print(f"Found {len(deleted_neighbors)} neighbors sharing ≥{K} hyperedges")
    t2_n = time.perf_counter()

    model.structure = A_tr_before
    for layer in model.layers:
        layer.reapproximate = False

    mask_del = torch.zeros_like(data_train["y"], dtype=torch.bool, device=device)
    mask_del[deleted] = True
    data_del_before = {"x": fts, "y": lbls, "train_mask": mask_del}
    print(f" Deleted-nodes F1 before: {evaluate_hgcn_f1(model, data_del_before):.4f}, "
          f"Acc: {evaluate_hgcn_acc(model, data_del_before):.4f}")

    fts_new, edge_list_new, A_after = apply_node_deletion_unlearning(
        fts, edge_list=hyperedges_tr,
        deleted_nodes=deleted,
        mediators=args.mediators, device=device
    )
    print("Check whether all features of the deleted nodes are zero:", (fts_new[deleted] != 0).sum().item())
    print(f" Deleted {len(deleted)} nodes; {len(edge_list_new)} hyperedges remain")

    gif_time = approx_gif(
        model, data_train,
        A_before, A_after,
        deleted_nodes=deleted,
        deleted_neighbors=deleted_neighbors,
        x_before=fts,
        x_after=fts_new,
        iters=80,
        damp=0.01,
        scale=1e7
    )
    t3 = time.perf_counter()

    neighbor_search_time = t2_n - t2
    gif_update_time = t3 - t2_n
    unlearn_time = t3 - t2  # = neighbor_search_time + gif_update_time

    print(
        f"Neighbor search: {neighbor_search_time:.4f}s, GIF update: {gif_update_time:.4f}s, Total unlearn_time: {unlearn_time:.4f}s")


    data_del_after = {"x": fts_new, "y": lbls, "train_mask": mask_del}
    print(f" Deleted-nodes F1 after: {evaluate_hgcn_f1(model, data_del_after):.4f}, "
          f"Acc: {evaluate_hgcn_acc(model, data_del_after):.4f}")

    model.structure = A_te
    for layer in model.layers:
        layer.reapproximate = False
    print("— After Unlearning —")
    print(f" Test F1: {evaluate_hgcn_f1(model, data_test):.4f}, "
          f"Acc: {evaluate_hgcn_acc(model, data_test):.4f}")

    model.structure = A_tr_before
    for layer in model.layers:
        layer.reapproximate = False

    member_mask = np.ones(len(y_tr), dtype=bool)
    member_mask[deleted.cpu().numpy()] = False

    print("— Membership Inference Attack on Unlearned Model —")
    res = membership_inference_hgcn(
        X_train=X_tr,
        y_train=y_tr,
        hyperedges=hyperedges_tr,
        target_model=model,
        args=args,
        device=device,
        member_mask=member_mask
    )
    # res = (attack_model, (auc_s, f1_s), (auc_t, f1_t))
    attack_model, _, (auc_un, f1_un) = res
    print(f" Unlearned Model MIA  AUC = {auc_un:.4f},  F1 = {f1_un:.4f}")

if __name__ == "__main__":
    for run in range(1,5):
        print("=== Run",run,"===")
        main()

