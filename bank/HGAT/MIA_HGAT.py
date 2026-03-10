import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score
from torch_geometric.data import Data

from HGNNs_Model.HGAT import HGAT_JK

class AttackModel(nn.Module):
    """
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
    """
    device = data.x.device
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn   = nn.NLLLoss()

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        out  = model(data.x, data.H, data.dv_inv, data.de_inv)
        loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
    return model

def collect_shadow_outputs(model: nn.Module,
                           data: Data,
                           num_samples: int = None) -> (np.ndarray, np.ndarray):
    """
    """
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.H, data.dv_inv, data.de_inv)
        probs  = F.softmax(logits, dim=1).cpu().numpy()  # [N, C]

    train_idx = data.train_indices
    test_idx  = data.test_indices
    if num_samples is None:
        num_samples = min(len(train_idx), len(test_idx))

    pos = random.sample(train_idx, num_samples)
    neg = random.sample(test_idx,  num_samples)

    X = np.vstack([probs[pos], probs[neg]])            # (2k, C)
    y = np.hstack([np.ones(num_samples), np.zeros(num_samples)])  # (2k,)

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
    """
    idx = np.arange(len(y))
    np.random.shuffle(idx)
    X, y = X[idx], y[idx]
    split = int(len(y) * (1 - test_split))
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    if isinstance(device, str):
        device = torch.device(device)
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    feat_dim = X_tr.shape[1]
    atk = AttackModel(feat_dim).to(device)
    opt = optim.Adam(atk.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    Xt = torch.from_numpy(X_tr).float().to(device)
    yt = torch.from_numpy(y_tr).float().to(device)

    # 5) 训练
    atk.train()
    for _ in range(epochs):
        opt.zero_grad()
        pred = atk(Xt)
        if pred.dim() == 2 and pred.size(1) == 1:
            pred = pred.squeeze(1)
        loss = loss_fn(pred, yt)
        loss.backward()
        opt.step()

    atk.eval()
    with torch.no_grad():
        Xte = torch.from_numpy(X_te).float().to(device)
        pred = atk(Xte)
        if pred.dim() == 2 and pred.size(1) == 1:
            pred = pred.squeeze(1)
        pred = pred.cpu().numpy()

    auc = roc_auc_score(y_te, pred)
    if auc < 0.5:
        pred = 1.0 - pred
        auc = roc_auc_score(y_te, pred)

    prec, rec, _  = precision_recall_curve(y_te, pred)
    f1s = 2 * prec * rec / (prec + rec + 1e-12)
    best_f1 = float(np.max(f1s))

    return atk, auc, best_f1
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
def _build_incidence_matrix(hyperedges: dict, num_nodes: int, device):
    rows, cols = [], []
    for e_idx, nodes in enumerate(hyperedges.values()):
        rows.extend(nodes)
        cols.extend([e_idx] * len(nodes))
    idx   = torch.tensor([rows, cols], dtype=torch.long, device=device)
    vals  = torch.ones(len(rows), dtype=torch.float32, device=device)
    H_ne  = torch.sparse_coo_tensor(idx, vals,
                                    size=(num_nodes, len(hyperedges)),
                                    device=device)
    return H_ne.coalesce()                         # (N, E)

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
class _HGATWrapper(nn.Module):
    def __init__(self, core: nn.Module, H_EN):
        super().__init__()
        self.core = core
        self.register_buffer('H', H_EN)           # (E, N)

    def forward(self, x, *_):
        return self.core(x, self.H)

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
def _train_shadow_hgat(model: nn.Module,
                       data: Data,
                       lr: float = 5e-3,
                       epochs: int = 100):
    optimiser = optim.Adam(model.parameters(), lr=lr)
    loss_fn   = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        optimiser.zero_grad()
        logits = model(data.x)
        loss   = loss_fn(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimiser.step()
    return model

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
def membership_inference_hgat(
        X_train,
        y_train,
        hyperedges,
        target_model,
        args,
        device=None,
        member_mask=None):
    """

    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X, y   = np.asarray(X_train), np.asarray(y_train)
    N      = X.shape[0]

    H_NE   = _build_incidence_matrix(hyperedges, N, device)   # (N, E)
    H_EN   = H_NE.transpose(0, 1)                             # (E, N) for HGAT

    C          = int(np.max(y)) + 1
    hidden_dim = getattr(args, 'hidden_dim', 64)
    num_layers = getattr(args, 'num_layers', 2)
    dropout    = getattr(args, 'dropout', 0.5)
    alpha      = getattr(args, 'alpha',   0.2)

    ratio     = getattr(args, 'shadow_test_ratio', 0.30)
    n_test    = int(N * ratio)
    all_idx   = np.arange(N)
    test_idx  = np.random.choice(all_idx, n_test, replace=False)
    train_idx = np.setdiff1d(all_idx, test_idx)

    train_mask = torch.zeros(N, dtype=torch.bool, device=device)
    test_mask  = torch.zeros(N, dtype=torch.bool, device=device)
    train_mask[torch.from_numpy(train_idx)] = True
    test_mask[ torch.from_numpy(test_idx) ] = True

    dv_inv = torch.ones(N,          device=device)
    de_inv = torch.ones(H_EN.size(0), device=device)

    data = Data(
        x          = torch.from_numpy(X).float().to(device),
        y          = torch.from_numpy(y).long().to(device),
        H          = H_EN,
        dv_inv     = dv_inv,
        de_inv     = de_inv,
        train_mask = train_mask,
        test_mask  = test_mask
    )
    data.train_indices = train_idx.tolist()
    data.test_indices  = test_idx.tolist()

    from MIA.MIA_utils import collect_shadow_outputs, train_attack_model
    num_shadows = getattr(args, 'num_shadows', 5)
    num_samp    = getattr(args, 'num_attack_samples', None)

    all_X, all_y = [], []
    for _ in range(num_shadows):
        core = HGAT_JK(
            in_dim     = X.shape[1],
            hidden_dim = hidden_dim,
            out_dim    = C,
            dropout    = dropout,
            alpha      = alpha,
            num_layers = num_layers,
            use_jk     = True
        ).to(device)
        shadow = _HGATWrapper(core, H_EN)

        _train_shadow_hgat(
            shadow,
            data,
            lr     = getattr(args, 'shadow_lr',     5e-3),
            epochs = getattr(args, 'shadow_epochs', 100)
        )

        Xi, yi = collect_shadow_outputs(shadow, data, num_samp)
        all_X.append(Xi)
        all_y.append(yi)

    X_attack = np.vstack(all_X)
    y_attack = np.hstack(all_y)

    # attack_model, (auc_s, f1_s) = train_attack_model(
    attack_model, auc_s, f1_s = train_attack_model(
        X_attack, y_attack,
        test_split = getattr(args, 'attack_test_split', 0.3),
        lr         = getattr(args, 'attack_lr', 1e-2),
        epochs     = getattr(args, 'attack_epochs', 100),
        device     = device
    )
    print(f"[Shadow MIA] AUC={auc_s:.4f}, F1={f1_s:.4f}")

    if target_model is None:
        return attack_model, (auc_s, f1_s), None

    target = _HGATWrapper(target_model.to(device), H_EN)
    target.eval()
    with torch.no_grad():
        probs_t = F.softmax(target(data.x), dim=1).cpu().numpy()

    if member_mask is not None:
        X_tgt, y_tgt = probs_t, member_mask.astype(int)
    else:
        X_tgt = np.vstack([probs_t[train_idx], probs_t[test_idx]])
        y_tgt = np.hstack([np.ones(len(train_idx)), np.zeros(len(test_idx))])

    attack_model.eval()
    with torch.no_grad():
        preds = attack_model(torch.from_numpy(X_tgt).float().to(device)).squeeze().cpu().numpy()

    auc_t = roc_auc_score(y_tgt, preds)
    f1_t  = f1_score     (y_tgt, (preds >= 0.5).astype(int))
    print(f"[Target MIA]  AUC={auc_t:.4f}, F1={f1_t:.4f}")

    return attack_model, (auc_s, f1_s), (auc_t, f1_t)