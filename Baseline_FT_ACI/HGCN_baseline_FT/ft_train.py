# ft_train.py
import copy
import time
import numpy as np
import torch
import torch.nn as nn

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def freeze_all_but_head(model):
    for p in model.parameters():
        p.requires_grad = False
    # 你 HyperGCN 的最后一层通常是 layers[-1]
    for p in model.layers[-1].parameters():
        p.requires_grad = True

def train_full(model, x, y, *,
               epochs: int,
               lr: float,
               weight_decay: float,
               milestones,
               gamma: float):
    """
    Full 训练：用最后一个 epoch 的参数（不做“train 上挑 best”）
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sch = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=gamma)
    crit = nn.NLLLoss()

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        out = model(x)
        loss = crit(out, y)
        loss.backward()
        opt.step()
        sch.step()
    return model

def finetune_steps(model, x_after, y, retain_mask, *,
                   steps: int,
                   lr: float,
                   weight_decay: float):
    """
    在 edited features (x_after) 上做 K steps，
    loss 只在 retain_mask 上计算（deleted 不参与）。
    """
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    crit = nn.NLLLoss()

    model.train()
    for _ in range(steps):
        opt.zero_grad()
        out = model(x_after)
        loss = crit(out[retain_mask], y[retain_mask])
        loss.backward()
        opt.step()
    return model
