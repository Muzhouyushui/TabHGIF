import torch
import torch.nn as nn
import torch.optim as optim
from utils.common_utils import evaluate_hgcn_f1
from HGCN.HyperGCN import HyperGCN
from database.data_preprocessing.data_preprocessing_K import generate_hyperedge_dict
from GIF.GIF_HGCN_ROW import train_model

def train_hgcn_on_remaining_data(X_tr, y_tr, hyperedges_tr, deleted, device, cfg, args, fts, lbls):
    """
    Function to train HGCN on the remaining dataset after deleting nodes.

    Parameters:
        X_tr (numpy.ndarray): Training node feature matrix.
        y_tr (List[int]): Labels for the training set.
        hyperedges_tr (List[List[int]]): Hyperedges for the training set.
        deleted (np.ndarray): Array of indices of nodes to be deleted.
        device (torch.device): Device to run the model on.
        cfg: Configuration containing hyperparameter values.
        args: Arguments that define model configuration.
        fts (torch.Tensor): Node features for training.
        lbls (torch.Tensor): Labels for training.

    Returns:
        f1_score (float): F1 score after re-training the model on the remaining dataset.
    """

    remaining_nodes = torch.setdiff1d(torch.arange(X_tr.shape[0]), deleted)


    X_remaining = X_tr[remaining_nodes]
    y_remaining = y_tr[remaining_nodes]


    hyperedges_remaining = generate_hyperedge_dict(
        X_remaining, args.cat_cols, max_nodes_per_hyperedge=args.max_nodes_per_hyperedge, device=device
    )


    model = HyperGCN(
        num_nodes=X_remaining.shape[0],
        edge_list=hyperedges_remaining,
        X_init=X_remaining,
        args=cfg
    ).to(device)


    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.NLLLoss()


    model = train_model(
        model,
        criterion,
        optimizer,
        None,
        fts[remaining_nodes], lbls[remaining_nodes],
        num_epochs=args.epochs,
        print_freq=args.log_every
    )


    data_remaining = {
        "x": fts[remaining_nodes],
        "y": lbls[remaining_nodes],
        "train_mask": torch.ones(len(remaining_nodes), dtype=torch.bool, device=device)
    }

    f1_score_remaining = evaluate_hgcn_f1(model, data_remaining)
    print(f"F1 score on remaining nodes after re-training: {f1_score_remaining:.4f}")

    return f1_score_remaining
