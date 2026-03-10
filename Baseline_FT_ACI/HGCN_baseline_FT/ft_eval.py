# ft_eval.py
import numpy as np
import torch
from sklearn.metrics import f1_score

def _set_structure(model, A):
    model.structure = A
    for l in getattr(model, "layers", []):
        if hasattr(l, "reapproximate"):
            l.reapproximate = False

@torch.no_grad()
def eval_split(model, x, y, A, mask=None):
    """
    返回 micro-F1 和 ACC（对二分类 micro-F1=ACC，但我们统一算）
    """
    model.eval()
    _set_structure(model, A)
    pred = model(x).argmax(dim=1)

    if mask is not None:
        pred = pred[mask]
        y = y[mask]

    acc = (pred == y).float().mean().item()

    pred_np = pred.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    f1 = f1_score(y_np, pred_np, average="micro")

    return f1, acc

def build_masks(N, deleted_idx_torch, device):
    retain_mask = torch.ones(N, dtype=torch.bool, device=device)
    retain_mask[deleted_idx_torch] = False
    del_mask = torch.zeros(N, dtype=torch.bool, device=device)
    del_mask[deleted_idx_torch] = True
    return retain_mask, del_mask

def member_mask_from_retain(retain_mask_torch):
    # retained=1, deleted=0
    return retain_mask_torch.detach().cpu().numpy().astype(int)

def member_mask_from_deleted(deleted_idx_torch, N):
    mm = np.zeros(N, dtype=int)
    mm[deleted_idx_torch.detach().cpu().numpy()] = 1
    return mm
