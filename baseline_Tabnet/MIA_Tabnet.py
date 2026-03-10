import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score
from pytorch_tabnet.tab_model import TabNetClassifier

def train_attack_model(
        X: np.ndarray,
        y: np.ndarray,
        test_split: float = 0.3,
        lr: float = 1e-2,
        epochs: int = 100,
        device=None
    ) -> (nn.Module, float, float):
    """
    训练二分类攻击器并评估 AUC/F1。
    """
    # 1) 划分攻击集
    idx = np.arange(len(y))
    np.random.shuffle(idx)
    X, y = X[idx], y[idx]
    split = int(len(y) * (1 - test_split))
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    # 2) 设备
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 3) 模型、优化器、Loss
    feat_dim = X_tr.shape[1]
    atk = nn.Sequential(
        nn.Linear(feat_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
        nn.Sigmoid()
    ).to(device)
    opt = optim.Adam(atk.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    # 4) 训练
    Xt = torch.from_numpy(X_tr).float().to(device)
    yt = torch.from_numpy(y_tr).float().to(device)
    atk.train()
    for _ in range(epochs):
        opt.zero_grad()
        pred = atk(Xt).squeeze(-1)
        loss = loss_fn(pred, yt)
        loss.backward()
        opt.step()

    # 5) 测试评估
    atk.eval()
    with torch.no_grad():
        Xte = torch.from_numpy(X_te).float().to(device)
        pred = atk(Xte).squeeze(-1).cpu().numpy()

    auc = roc_auc_score(y_te, pred)
    if auc < 0.5:
        pred = 1 - pred
        auc = roc_auc_score(y_te, pred)
    prec, rec, _ = precision_recall_curve(y_te, pred)
    f1 = float(np.max(2 * prec * rec / (prec + rec + 1e-12)))

    return atk, auc, f1

def membership_inference(
        X_train: np.ndarray,
        y_train: np.ndarray,
        hyperedges,                    # unused for TabNet
        target_model: TabNetClassifier,
        args,
        device=None,
        member_mask=None,              # True 表示 member, False 表示 non-member
        attack_model: torch.nn.Module = None  # 如果已有攻击器就复用
    ):
    """
    兼容 HGNN/MIA_utils 签名，用 TabNetClassifier 做 Shadow-Model MIA。

    Args:
        X_train, y_train:     原始训练数据
        hyperedges:           保留但不使用
        target_model:         TabNetClassifier（原模型或 unlearned 模型）
        args:                 包含 num_shadows, shadow_epochs,
                              attack_test_split, attack_lr, attack_epochs
        device:               torch.device
        member_mask:          长度等于 len(X_train)，True=保留/正例，False=删除/负例
        attack_model:         如果传入，就跳过 Shadow 阶段，直接复用

    Returns:
        attack_model,
        (auc_shadow, f1_shadow),
        (auc_target, f1_target)
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if member_mask is None:
        raise ValueError("TabNet MIA: 必须提供 member_mask 来划分正负例")

    # 划分正例 (members) / 负例 (non-members)
    mem_idx = np.where(member_mask)[0].tolist()
    non_idx = np.where(~member_mask)[0].tolist()
    X_mem, y_mem = X_train[mem_idx], y_train[mem_idx]
    X_non, y_non = X_train[non_idx], y_train[non_idx]

    # —— 1) Shadow 阶段 —— 仅当没有传入 attack_model 时进行
    if attack_model is None:
        all_feats, all_labels = [], []
        for _ in range(args.num_shadows):
            # 创建并训练 shadow 模型
            shadow = TabNetClassifier(
                n_d               = getattr(args, "shadow_n_d", 8),
                n_a               = getattr(args, "shadow_n_a", 8),
                n_steps           = getattr(args, "shadow_n_steps", 3),
                gamma             = getattr(args, "shadow_gamma", 1.3),
                lambda_sparse     = getattr(args, "shadow_lambda", 1e-3),
                optimizer_fn      = torch.optim.Adam,
                optimizer_params  = {"lr": getattr(args, "shadow_lr", 1e-2)},
                scheduler_fn      = torch.optim.lr_scheduler.StepLR,
                scheduler_params  = {"step_size": 50, "gamma": 0.9},
                mask_type         = "sparsemax",
                device_name       = device.type,
                verbose           = 0
            )
            # 用 members 训练，non-members 验证
            shadow.fit(
                X_mem, y_mem,
                eval_set            = [(X_non, y_non)],
                eval_name           = ["val"],
                eval_metric         = ["accuracy"],
                max_epochs          = getattr(args, "shadow_epochs", 150),
                patience            = 70,
                batch_size          = getattr(args, "shadow_batch_size", 1024),
                virtual_batch_size  = getattr(args, "shadow_virtual_batch_size", 256),
                num_workers         = 0,
                drop_last           = False
            )

            # 收集置信度（max softmax）
            pm = F.softmax(torch.tensor(shadow.predict_proba(X_mem)), dim=1).max(dim=1)[0].cpu().numpy().reshape(-1,1)
            pn = F.softmax(torch.tensor(shadow.predict_proba(X_non)), dim=1).max(dim=1)[0].cpu().numpy().reshape(-1,1)

            feats  = np.vstack([pm, pn])
            labels = np.hstack([np.ones(len(pm)), np.zeros(len(pn))])
            all_feats.append(feats)
            all_labels.append(labels)

        X_attack = np.vstack(all_feats)
        y_attack = np.concatenate(all_labels)

        # 训练攻击模型
        attack_model, auc_shadow, f1_shadow = train_attack_model(
            X_attack, y_attack,
            test_split = getattr(args, "attack_test_split", 0.3),
            lr         = getattr(args, "attack_lr",        1e-2),
            epochs     = getattr(args, "attack_epochs",    150),
            device     = device
        )
        print(f"[Shadow MIA] AUC={auc_shadow:.4f}, F1={f1_shadow:.4f}")
    else:
        # 复用已有攻击器，shadow 阶段跳过
        auc_shadow = None
        f1_shadow  = None

    # —— 2) 在 target_model 上评估 —— 始终使用 attack_model
    # 收集 target_model 的置信度
    pm_t = F.softmax(torch.tensor(target_model.predict_proba(X_mem)), dim=1).max(dim=1)[0].cpu().numpy().reshape(-1,1)
    pn_t = F.softmax(torch.tensor(target_model.predict_proba(X_non)), dim=1).max(dim=1)[0].cpu().numpy().reshape(-1,1)

    X_tgt = np.vstack([pm_t, pn_t])
    y_tgt = np.hstack([np.ones(len(pm_t)), np.zeros(len(pn_t))])

    attack_model.eval()
    with torch.no_grad():
        preds = attack_model(torch.from_numpy(X_tgt).float().to(device)).squeeze().cpu().numpy()

    auc_target = roc_auc_score(y_tgt, preds)
    f1_target = f1_score(y_tgt, (preds > 0.5).astype(int))
    print(f"[Target MIA] AUC={auc_target:.4f}, F1={f1_target:.4f}")

    return attack_model, (auc_shadow, f1_shadow), (auc_target, f1_target)
