import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score
from torch_geometric.data import Data
from bank.HGCN.HGCN import HyperGCN,laplacian
from types import SimpleNamespace


class AttackModel(nn.Module):
    """
    Simple two-layer binary attack model
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.fc1  = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2  = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x)).squeeze(-1)


def train_shadow_model(model: nn.Module,
                       data: Data,
                       lr: float = 5e-3,
                       epochs: int = 100) -> nn.Module:
    """
    Train a shadow HGCN with NLLLoss, and ensure that its structure (Laplacian)
    is on the same device.
    """
    device = data.x.device
    model.to(device)

    # —— Ensure model.structure is a sparse tensor and on GPU —— #
    if isinstance(model.structure, list):
        # data.x.cpu().numpy() is the original feature matrix X_init
        X_np = data.x.cpu().numpy()
        # Recompute the Laplacian and move it to GPU
        structure = laplacian(model.structure, X_np, model.mediators)
        model.structure = structure.to(device)
    else:
        # If it is already a tensor, just .to(device)
        model.structure = model.structure.to(device)

    # (Optional) If your HyperGCN caches structure inside layers, update them as well
    for layer in getattr(model, "layers", []):
        if hasattr(layer, "reapproximate"):
            # We use fast mode, so we keep it fixed and do not recompute dynamically
            layer.reapproximate = False

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn   = nn.NLLLoss()

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x)  # model.structure is already on GPU here
        loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    return model


def collect_shadow_outputs(model: nn.Module,
                           data: Data,
                           num_samples: int = None) -> (np.ndarray, np.ndarray):
    """
    Randomly sample the same number of members / non-members from the shadow HGCN,
    and return softmax probabilities (2k, C) and corresponding labels y (2k,)

    - model.forward only accepts data.x
    - indices are extracted from data.train_mask/test_mask
    """
    model.eval()
    with torch.no_grad():
        # HGCN only needs x as input
        logits = model(data.x)
        probs = F.softmax(logits, dim=1).cpu().numpy()  # [N, C]

    # Extract indices from boolean masks
    train_idx = data.train_mask.nonzero(as_tuple=True)[0].cpu().tolist()
    test_idx = data.test_mask.nonzero(as_tuple=True)[0].cpu().tolist()

    if num_samples is None:
        num_samples = min(len(train_idx), len(test_idx))

    # Random sampling
    pos = random.sample(train_idx, num_samples)
    neg = random.sample(test_idx, num_samples)

    X = np.vstack([probs[pos], probs[neg]])  # (2k, C)
    y = np.hstack([np.ones(num_samples), np.zeros(num_samples)])  # (2k,)

    # Global shuffle
    perm = np.random.permutation(len(y))
    return X[perm], y[perm]

def train_attack_model(
        X: np.ndarray,
        y: np.ndarray,
        test_split: float = 0.3,
        lr: float = 1e-2,
        epochs: int = 100,
        device=None
    ) -> (AttackModel, float, float):
    """
    Train the attack model and evaluate AUC/F1 on the shadow-test set, with support for a specified device.
    The mismatch issue between prediction/label shapes has already been fixed.
    """
    # 1) Split training/test sets
    idx = np.arange(len(y))
    np.random.shuffle(idx)
    X, y = X[idx], y[idx]
    split = int(len(y) * (1 - test_split))
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    # 2) Device setup
    if isinstance(device, str):
        device = torch.device(device)
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 3) Build model, optimizer, and loss
    feat_dim = X_tr.shape[1]
    atk = AttackModel(feat_dim).to(device)
    opt = optim.Adam(atk.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    # 4) Convert to tensors and move to device
    Xt = torch.from_numpy(X_tr).float().to(device)
    # Note: do not unsqueeze, keep shape=[N]
    yt = torch.from_numpy(y_tr).float().to(device)

    # 5) Training
    atk.train()
    for _ in range(epochs):
        opt.zero_grad()
        pred = atk(Xt)
        # If the model output is [N,1], squeeze it
        if pred.dim() == 2 and pred.size(1) == 1:
            pred = pred.squeeze(1)   # becomes [N]
        loss = loss_fn(pred, yt)
        loss.backward()
        opt.step()

    # 6) Test-set evaluation
    atk.eval()
    with torch.no_grad():
        Xte = torch.from_numpy(X_te).float().to(device)
        pred = atk(Xte)
        if pred.dim() == 2 and pred.size(1) == 1:
            pred = pred.squeeze(1)
        pred = pred.cpu().numpy()

    # 7) Compute AUC
    auc = roc_auc_score(y_te, pred)
    if auc < 0.5:
        pred = 1.0 - pred
        auc = roc_auc_score(y_te, pred)

    # 8) Compute best F1
    prec, rec, _  = precision_recall_curve(y_te, pred)
    f1s = 2 * prec * rec / (prec + rec + 1e-12)
    best_f1 = float(np.max(f1s))

    return atk, auc, best_f1


# def membership_inference(
#         X_train,
#         y_train,
#         hyperedges,
#         target_model,
#         args,
#         device=None
#     ):

def membership_inference_hgcn(
        X_train,
        y_train,
        hyperedges,
        target_model,
        args,
        device=None,
        member_mask=None
    ):
    """
    Execute the full MIA pipeline on HGCN:
      0) Construct cfg synchronized with main(), ensuring all fields required by HyperGCN initialization are complete
      1) Split member vs non-member → Data(x, y, train_mask, test_mask)
      2) Multiple shadow rounds: instantiate shadow HGCN → train_shadow_model → collect_shadow_outputs
      3) Concatenate all shadow outputs → train_attack_model → print shadow AUC/F1
      4) Perform the same inference on target_model → print target AUC/F1 → return results
    """
    # —— 0) Homogeneous cfg —— #
    cfg = SimpleNamespace()
    cfg.d         = X_train.shape[1]
    cfg.depth     = getattr(args, "depth", 1)
    cfg.c         = int(np.max(y_train)) + 1
    cfg.dropout   = getattr(args, "dropout", 0.5)
    cfg.fast      = getattr(args, "fast", False)
    cfg.mediators = getattr(args, "mediators", False)
    cfg.cuda      = getattr(args, "cuda", False)
    cfg.dataset   = getattr(args, "dataset", None)

    # Device & convert to np
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X = np.asarray(X_train)
    y = np.asarray(y_train)
    N = X.shape[0]

    # —— 1) Split member / non-member —— #
    ratio   = getattr(args, "shadow_test_ratio", 0.3)
    n_test  = int(N * ratio)
    all_idx = np.arange(N)
    test_idx  = np.random.choice(all_idx, n_test, replace=False)
    train_idx = np.setdiff1d(all_idx, test_idx)

    train_mask = torch.zeros(N, dtype=torch.bool, device=device)
    test_mask  = torch.zeros(N, dtype=torch.bool, device=device)
    train_mask[torch.from_numpy(train_idx)] = True
    test_mask[ torch.from_numpy(test_idx) ] = True

    data = Data(
        x          = torch.from_numpy(X).float().to(device),
        y          = torch.from_numpy(y).long().to(device),
        train_mask = train_mask,
        test_mask  = test_mask
    )

    # —— 2) Multiple shadow rounds —— #
    num_shadows = getattr(args, "num_shadows", 5)
    num_samp    = getattr(args, "num_attack_samples", None)
    all_X, all_y = [], []

    for _ in range(num_shadows):
        # Instantiate shadow HGCN
        shadow = HyperGCN(
            num_nodes = N,
            edge_list = hyperedges,
            X_init    = X,
            args      = cfg
        ).to(device)

        # Ensure the Laplacian is on the same device
        if isinstance(shadow.structure, list):
            shadow.structure = laplacian(hyperedges, X, cfg.mediators).to(device)
        else:
            shadow.structure = shadow.structure.to(device)

        # Train shadow model
        train_shadow_model(
            shadow, data,
            lr     = getattr(args, "shadow_lr", 5e-3),
            epochs = getattr(args, "shadow_epochs", 100)
        )
        # Collect outputs
        Xi, yi = collect_shadow_outputs(shadow, data, num_samp)
        all_X.append(Xi)
        all_y.append(yi)

    X_attack = np.vstack(all_X)
    y_attack = np.concatenate(all_y)

    # —— 3) Train attacker & print shadow metrics —— #
    attack_model, auc_s, f1_s = train_attack_model(
        X_attack, y_attack,
        test_split = getattr(args, "attack_test_split", 0.3),
        lr         = getattr(args, "attack_lr", 1e-2),
        epochs     = getattr(args, "attack_epochs", 50),
        device     = device
    )
    print(f"[Shadow MIA] AUC={auc_s:.4f}, F1={f1_s:.4f}")

    # —— 4) Evaluate on target_model —— #
    if target_model is not None:
        target_model.eval()
        # Forward only needs x
        with torch.no_grad():
            logits_t = target_model(data.x)
            probs_t  = F.softmax(logits_t, dim=1).cpu().numpy()

        # Prepare attack input & ground-truth labels
        if member_mask is not None:
            X_tgt = probs_t
            y_tgt = member_mask.astype(int)
        else:
            pos   = train_idx.tolist()
            neg   = test_idx.tolist()
            X_tgt = np.vstack([probs_t[pos], probs_t[neg]])
            y_tgt = np.hstack([np.ones(len(pos)), np.zeros(len(neg))])

        # Attacker inference
        attack_model.eval()
        with torch.no_grad():
            inp = torch.from_numpy(X_tgt).float().to(device)
            out = attack_model(inp)
            if out.dim()==2 and out.size(1)==1:
                out = out.squeeze(1)
            pred = out.cpu().numpy()

        # Compute AUC/F1
        auc_t = roc_auc_score(y_tgt, pred)
        if auc_t < 0.5:
            pred  = 1.0 - pred
            auc_t = roc_auc_score(y_tgt, pred)
        labels = (pred >= 0.5).astype(int)
        f1_t   = f1_score(y_tgt, labels)

        print(f"[Target MIA] AUC={auc_t:.4f}, F1={f1_t:.4f}")
        return attack_model, (auc_s, f1_s), (auc_t, f1_t)

    return attack_model, (auc_s, f1_s), None