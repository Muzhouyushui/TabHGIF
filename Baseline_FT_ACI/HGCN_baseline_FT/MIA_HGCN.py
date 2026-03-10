import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score
from torch_geometric.data import Data
from HGCN.HyperGCN import HyperGCN,laplacian
from types import SimpleNamespace


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
    用 NLLLoss 训练一个 Shadow HGCN，并确保它的 structure（Laplacian）在同一 device 上。
    """
    device = data.x.device
    model.to(device)

    # —— 确保 model.structure 是稀疏张量并在 GPU —— #
    if isinstance(model.structure, list):
        # data.x.cpu().numpy() 就是原始特征 X_init
        X_np = data.x.cpu().numpy()
        # 重算 Laplacian，并搬到 GPU
        structure = laplacian(model.structure, X_np, model.mediators)
        model.structure = structure.to(device)
    else:
        # 如果它本来就是个张量，直接 .to(device)
        model.structure = model.structure.to(device)

    # （可选）如果你的 HyperGCN 在 layer 里缓存了 structure，也一并更新
    for layer in getattr(model, "layers", []):
        if hasattr(layer, "reapproximate"):
            # 我们用 fast 模式，所以固定不再动态重算
            layer.reapproximate = False

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn   = nn.NLLLoss()

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x)  # 这时 model.structure 已在 GPU
        loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    return model


def collect_shadow_outputs(model: nn.Module,
                           data: Data,
                           num_samples: int = None) -> (np.ndarray, np.ndarray):
    """
    从 Shadow HGCN 中随机抽取同样多的成员 / 非成员，
    返回 softmax 概率 (2k, C) 和对应标签 y (2k,)

    - model.forward 只接受 data.x
    - 从 data.train_mask/test_mask 提取索引
    """
    model.eval()
    with torch.no_grad():
        # HGCN 只要传入 x
        logits = model(data.x)
        probs = F.softmax(logits, dim=1).cpu().numpy()  # [N, C]

    # 从布尔 mask 中找出索引
    train_idx = data.train_mask.nonzero(as_tuple=True)[0].cpu().tolist()
    test_idx = data.test_mask.nonzero(as_tuple=True)[0].cpu().tolist()

    if num_samples is None:
        num_samples = min(len(train_idx), len(test_idx))

    # 随机抽样
    pos = random.sample(train_idx, num_samples)
    neg = random.sample(test_idx, num_samples)

    X = np.vstack([probs[pos], probs[neg]])  # (2k, C)
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
    在 HGCN 上执行完整 MIA 流程：
      0) 构造与 main() 同步的 cfg，保证 HyperGCN 初始化所需字段齐全
      1) 划分 member vs non-member → Data(x, y, train_mask, test_mask)
      2) 多轮 Shadow：实例化 Shadow HGCN → train_shadow_model → collect_shadow_outputs
      3) 拼接所有 Shadow 输出 → train_attack_model → 打印 Shadow AUC/F1
      4) 在 target_model 上做同样推理 → 打印 Target AUC/F1 → 返回结果
    """
    # —— 0) 同构 cfg —— #
    cfg = SimpleNamespace()
    cfg.d         = X_train.shape[1]
    cfg.depth     = getattr(args, "depth", 1)
    cfg.c         = int(np.max(y_train)) + 1
    cfg.dropout   = getattr(args, "dropout", 0.5)
    cfg.fast      = getattr(args, "fast", False)
    cfg.mediators = getattr(args, "mediators", False)
    cfg.cuda      = getattr(args, "cuda", False)
    cfg.dataset   = getattr(args, "dataset", None)

    # 设备 & 转 np
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X = np.asarray(X_train)
    y = np.asarray(y_train)
    N = X.shape[0]

    # —— 1) 划分 member / non-member —— #
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

    # —— 2) 多轮 Shadow —— #
    num_shadows = getattr(args, "num_shadows", 5)
    num_samp    = getattr(args, "num_attack_samples", None)
    all_X, all_y = [], []

    for _ in range(num_shadows):
        # 实例化 Shadow HGCN
        shadow = HyperGCN(
            num_nodes = N,
            edge_list = hyperedges,
            X_init    = X,
            args      = cfg
        ).to(device)

        # 确保 Laplacian 在同一 device
        if isinstance(shadow.structure, list):
            shadow.structure = laplacian(hyperedges, X, cfg.mediators).to(device)
        else:
            shadow.structure = shadow.structure.to(device)

        # 训练 Shadow
        train_shadow_model(
            shadow, data,
            lr     = getattr(args, "shadow_lr", 5e-3),
            epochs = getattr(args, "shadow_epochs", 100)
        )
        # 收集输出
        Xi, yi = collect_shadow_outputs(shadow, data, num_samp)
        all_X.append(Xi)
        all_y.append(yi)

    X_attack = np.vstack(all_X)
    y_attack = np.concatenate(all_y)

    # —— 3) 训练攻击器 & 打印 Shadow 指标 —— #
    attack_model, auc_s, f1_s = train_attack_model(
        X_attack, y_attack,
        test_split = getattr(args, "attack_test_split", 0.3),
        lr         = getattr(args, "attack_lr", 1e-2),
        epochs     = getattr(args, "attack_epochs", 50),
        device     = device
    )
    print(f"[Shadow MIA] AUC={auc_s:.4f}, F1={f1_s:.4f}")

    # —— 4) 在 target_model 上评估 —— #
    if target_model is not None:
        target_model.eval()
        # 前向只需 x
        with torch.no_grad():
            logits_t = target_model(data.x)
            probs_t  = F.softmax(logits_t, dim=1).cpu().numpy()

        # 准备攻击输入 & 真实标签
        if member_mask is not None:
            X_tgt = probs_t
            y_tgt = member_mask.astype(int)
        else:
            pos   = train_idx.tolist()
            neg   = test_idx.tolist()
            X_tgt = np.vstack([probs_t[pos], probs_t[neg]])
            y_tgt = np.hstack([np.ones(len(pos)), np.zeros(len(neg))])

        # 攻击器推理
        attack_model.eval()
        with torch.no_grad():
            inp = torch.from_numpy(X_tgt).float().to(device)
            out = attack_model(inp)
            if out.dim()==2 and out.size(1)==1:
                out = out.squeeze(1)
            pred = out.cpu().numpy()

        # 计算 AUC/F1
        auc_t = roc_auc_score(y_tgt, pred)
        if auc_t < 0.5:
            pred  = 1.0 - pred
            auc_t = roc_auc_score(y_tgt, pred)
        labels = (pred >= 0.5).astype(int)
        f1_t   = f1_score(y_tgt, labels)

        print(f"[Target MIA] AUC={auc_t:.4f}, F1={f1_t:.4f}")
        return attack_model, (auc_s, f1_s), (auc_t, f1_t)

    return attack_model, (auc_s, f1_s), None