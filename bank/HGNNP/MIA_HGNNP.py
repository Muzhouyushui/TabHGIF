import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score
from torch_geometric.data import Data

from bank.HGNNP.HGNNP      import HGNNP_implicit,build_incidence_matrix, compute_degree_vectors

class AttackModel(nn.Module):
    """
    简单的两层二分类攻击器
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
    用交叉熵训练单个 Shadow HGNN
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
    从 Shadow 模型中随机抽取同样多的成员 / 非成员，
    返回 softmax 概率 (2k, C) 和对应标签 y (2k,)
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

    # 全局打乱
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
    训练攻击模型并在 Shadow-test 上评估 AUC/F1，支持指定 device。
    已修正预测/标签 shape 不匹配问题。
    """
    # 1) 划分训练/测试集
    idx = np.arange(len(y))
    np.random.shuffle(idx)
    X, y = X[idx], y[idx]
    split = int(len(y) * (1 - test_split))
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    # 2) 设备准备
    if isinstance(device, str):
        device = torch.device(device)
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 3) 构建模型、优化器、Loss
    feat_dim = X_tr.shape[1]
    atk = AttackModel(feat_dim).to(device)
    opt = optim.Adam(atk.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    # 4) 转 Tensor 并搬到 device
    Xt = torch.from_numpy(X_tr).float().to(device)
    # 注意：不要 unsqueeze，保持 shape=[N]
    yt = torch.from_numpy(y_tr).float().to(device)

    # 5) 训练
    atk.train()
    for _ in range(epochs):
        opt.zero_grad()
        pred = atk(Xt)
        # 如果模型输出是 [N,1]，就 squeeze 一下
        if pred.dim() == 2 and pred.size(1) == 1:
            pred = pred.squeeze(1)   # 变成 [N]
        loss = loss_fn(pred, yt)
        loss.backward()
        opt.step()

    # 6) 测试集评估
    atk.eval()
    with torch.no_grad():
        Xte = torch.from_numpy(X_te).float().to(device)
        pred = atk(Xte)
        if pred.dim() == 2 and pred.size(1) == 1:
            pred = pred.squeeze(1)
        pred = pred.cpu().numpy()

    # 7) 计算 AUC
    auc = roc_auc_score(y_te, pred)
    if auc < 0.5:
        pred = 1.0 - pred
        auc = roc_auc_score(y_te, pred)

    # 8) 计算最优 F1
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
def membership_inference(
            X_train, y_train, hyperedges,
            target_model, args, device,
            member_mask=None  # 新增参数：同 X_attack 一一对应，True 表示 member
        ):
    """
    完整 MIA 流程：
      1) 用 X_train, y_train, hyperedges 构造一个 Shadow 数据集 Data
      2) 重复 num_shadows 次：训练 Shadow HGNN → 收集 Shadow 输出 logits/probs
      3) 将所有 Shadow 输出拼成 X_attack, y_attack，训练攻击模型
      4) 在 Shadow 自身上评估 AUC/F1，并打印
      5) 如果传入了 target_model，则在 target_model 上做同样的推理，评估 AUC/F1，并返回
    """
    # 设备
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 确保 numpy array
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    N = X_train.shape[0]

    # 1) 构造超图 incidence matrix + degree vectors
    H_sp         = build_incidence_matrix(hyperedges, N)
    dv_inv, de_inv = compute_degree_vectors(H_sp)
    Hc = H_sp.tocoo()
    idx = torch.LongTensor([Hc.row, Hc.col])
    val = torch.FloatTensor(Hc.data)
    H_tensor = torch.sparse_coo_tensor(idx, val, Hc.shape).to(device)

    # Shadow：member vs non-member 划分
    shadow_test_ratio = getattr(args, "shadow_test_ratio", 0.3)
    test_size = int(N * shadow_test_ratio)
    all_idx = np.arange(N)
    test_idx = np.random.choice(all_idx, test_size, replace=False).tolist()
    train_idx = list(set(all_idx.tolist()) - set(test_idx))

    train_mask = torch.zeros(N, dtype=torch.bool, device=device)
    train_mask[torch.LongTensor(train_idx)] = True

    data = Data(
        x             = torch.from_numpy(X_train).float().to(device),
        y             = torch.from_numpy(y_train).long().to(device),
        H             = H_tensor,
        dv_inv        = torch.from_numpy(dv_inv).to(device),
        de_inv        = torch.from_numpy(de_inv).to(device),
        train_mask    = train_mask,
        train_indices = train_idx,
        test_indices  = test_idx
    )

    # 2) 多轮 Shadow
    num_shadows = getattr(args, "num_shadows", 5)
    num_samp    = getattr(args, "num_attack_samples", None)
    all_X, all_y = [], []
    for _ in range(num_shadows):
        shadow = HGNNP_implicit(
            in_ch   = X_train.shape[1],
            n_class = int(y_train.max()) + 1,
            n_hid   = getattr(args, "shadow_hidden", 64),
            dropout = getattr(args, "shadow_dropout", 0.5)
        ).to(device)

        train_shadow_model(
            shadow, data,
            lr     = getattr(args, "shadow_lr", 5e-3),
            epochs = getattr(args, "shadow_epochs", 100)
        )
        Xi, yi = collect_shadow_outputs(shadow, data, num_samp)
        all_X.append(Xi)
        all_y.append(yi)

    X_attack = np.vstack(all_X)
    y_attack = np.concatenate(all_y)

    # 3) 训练攻击模型
    attack_model, auc_shadow, f1_shadow = train_attack_model(
        X_attack,
        y_attack,
        test_split = getattr(args, "attack_test_split", 0.3),
        lr         = getattr(args, "attack_lr", 1e-2),
        epochs     = getattr(args, "attack_epochs", 50),
        device     = device
    )
    print(f"[Shadow MIA] AUC={auc_shadow:.4f}, F1={f1_shadow:.4f}")

    # 4) （可选）在 target_model 上评估
    if target_model is not None:
        target_model.eval()
        with torch.no_grad():
            logits_t = target_model(data.x, data.H, data.dv_inv, data.de_inv)
            probs_t  = F.softmax(logits_t, dim=1).cpu().numpy()

        # —— 新增对 member_mask 的判断 —— #
        if member_mask is not None:
            # member_mask 长度必须等于 N，True 表示 member，False 表示 non-member
            # 直接用全部 probs_t 作为攻击器输入，用 member_mask 作为标签
            X_tgt = probs_t
            y_tgt = member_mask.astype(int)   # 1/0
        else:
            # 原来的正负例拆分逻辑
            X_pos = probs_t[train_idx]
            X_neg = probs_t[test_idx]
            X_tgt = np.vstack([X_pos, X_neg])
            y_tgt = np.hstack([np.ones(len(train_idx)), np.zeros(len(test_idx))])

        # 然后用 attack_model 预测 X_tgt，算 AUC/F1
        attack_model.eval()
        with torch.no_grad():
            X_tgt_tensor = torch.from_numpy(X_tgt).float().to(device)
            out = attack_model(X_tgt_tensor)
            if out.dim() == 2 and out.size(1) == 1:
                out = out.squeeze(1)
            pred = out.cpu().numpy()

        auc_tgt = roc_auc_score(y_tgt, pred)
        if auc_tgt < 0.5:
            pred = 1.0 - pred
            auc_tgt = roc_auc_score(y_tgt, pred)
        pred_label = (pred >= 0.5).astype(int)
        f1_tgt = f1_score(y_tgt, pred_label)

        print(f"[Target MIA]  AUC={auc_tgt:.4f}, F1={f1_tgt:.4f}")
        return attack_model, (auc_shadow, f1_shadow), (auc_tgt, f1_tgt)
