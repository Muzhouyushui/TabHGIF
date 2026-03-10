# ft_metrics.py
import torch

def accuracy(model, x, y, mask=None):
    model.eval()
    with torch.no_grad():
        out = model(x).argmax(dim=1)
        if mask is None:
            return (out == y).float().mean().item()
        return (out[mask] == y[mask]).float().mean().item()

def micro_f1_binary(model, x, y, mask=None):
    """
    二分类 micro-F1 = accuracy；这里给个统一接口。
    """
    return accuracy(model, x, y, mask=mask)
