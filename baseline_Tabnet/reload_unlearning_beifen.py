
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from torch.utils.data import DataLoader

class RELOADUnlearner:
    """
    Blind unlearning per Newatia et al. (2024), Algorithm 1.
    """
    def __init__(self, eta_p: float = 0.1, reset_frac: float = 0.1, eps: float = 1e-8):
        self.eta_p = eta_p
        self.alpha = reset_frac
        self.eps = eps

    def compute_grad(self, model: nn.Module, loader: DataLoader, loss_fn) -> dict:
        """
        Compute summed gradients over dataset in `loader`.
        Returns a dict name->gradient tensor on the model's device.
        """
        model.eval()
        device = next(model.parameters()).device
        grads = {
            name: torch.zeros_like(param, device=device)
            for name, param in model.named_parameters()
        }
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            model.zero_grad()
            out = model(X)
            logits = out[0] if isinstance(out, tuple) else out
            loss_fn(logits, y).backward()
            for name, param in model.named_parameters():
                g = param.grad if param.grad is not None else torch.zeros_like(param, device=device)
                grads[name] += g.detach()
        return grads

    def unlearn(
        self,
        model: nn.Module,
        grads_all: dict,
        loader_retain: DataLoader,
        loss_fn,
        ft_epochs: int = 10,
        ft_lr: float = 1e-3
    ) -> nn.Module:
        """
        Perform RELOAD unlearning on `model` given:
          - grads_all: gradients over the full training set
          - loader_retain: DataLoader over the retain set D_retain
        Returns a new model with forgotten parameters.
        """
        device = next(model.parameters()).device

        # 1) Compute gradients on the retain set
        grads_retain = self.compute_grad(model, loader_retain, loss_fn)

        # 2) Gradient difference: g_forget = g_all - g_retain
        grads_forget = {
            name: grads_all[name] - grads_retain[name]
            for name in grads_all
        }

        # 3) Priming ascent: θ' = θ + η_p * g_forget
        model_p = deepcopy(model)
        with torch.no_grad():
            for name, param in model_p.named_parameters():
                param.add_(self.eta_p * grads_forget[name])

        # 4) Compute Knowledge Values (KV) and reinit lowest-α fraction
        #    KV_k = |g_forget_k| / (|g_all_k| + eps)
        all_kv = torch.cat([
            (grads_forget[name].abs() + self.eps)
            .div(grads_all[name].abs() + self.eps)
            .flatten()
            for name in grads_forget
        ])
        thresh = torch.quantile(all_kv, self.alpha)

        with torch.no_grad():
            for name, param in model_p.named_parameters():
                kv = (grads_forget[name].abs() + self.eps) \
                     .div(grads_all[name].abs() + self.eps) \
                     .flatten()
                mask = kv <= thresh
                flat = param.view(-1)
                # reinitialize masked elements from normal(dist of flat)
                rnd = torch.randn_like(flat) * flat.std() + flat.mean()
                flat[mask] = rnd[mask]
                param.copy_(flat.view(param.shape))

        # 5) Fine-tune on the retain set D_retain
        optimizer = optim.Adam(model_p.parameters(), lr=ft_lr)
        model_p.train()
        for _ in range(ft_epochs):
            for X, y in loader_retain:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                out = model_p(X)
                logits = out[0] if isinstance(out, tuple) else out
                loss_fn(logits, y).backward()
                optimizer.step()

        return model_p


def reload_unlearn_tabnet(
    model: nn.Module,
    train_loader: DataLoader,
    retain_loader: DataLoader,
    loss_fn,
    eta_p: float = 0.1,
    reset_frac: float = 0.1,
    ft_epochs: int = 10,
    ft_lr: float = 1e-3,
    eps: float = 1e-8
) -> nn.Module:
    """
    Interface to apply RELOAD blind unlearning to a TabNet model.

    Args:
        model:         pretrained nn.Module (e.g., TabNet)
        train_loader:  DataLoader for full training set D
        retain_loader: DataLoader for retain set D_retain
        loss_fn:       loss function (e.g. nn.CrossEntropyLoss())
        eta_p:         priming learning rate η_p
        reset_frac:    fraction α of params to reset
        ft_epochs:     number of fine-tuning epochs
        ft_lr:         fine-tune learning rate
        eps:           small constant for stability

    Returns:
        model_unlearn: the unlearned model
    """
    unlearner = RELOADUnlearner(eta_p, reset_frac, eps)
    # 1) Compute full-set gradients
    grads_all = unlearner.compute_grad(model, train_loader, loss_fn)
    # 2) Run unlearning
    model_unlearn = unlearner.unlearn(
        model, grads_all, retain_loader,
        loss_fn, ft_epochs=ft_epochs, ft_lr=ft_lr
    )
    return model_unlearn
