import time
import copy

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import grad


# ——— A. 训练函数 ———
def train_model(model, criterion, optimizer, scheduler, fts, lbls, num_epochs=200, print_freq=10):
    since = time.time()
    best_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
        model.train()
        optimizer.zero_grad()
        out = model(fts)
        loss = criterion(out, lbls)
        loss.backward()
        optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            preds = model(fts).argmax(dim=1)
            acc = (preds == lbls).float().mean().item()
        model.train()
        if acc > best_acc:
            best_acc = acc
            best_wts = copy.deepcopy(model.state_dict())
        if epoch % print_freq == 0:
            tqdm.write(f"[{epoch}/{num_epochs-1}] loss={loss:.4f} acc={acc:.4f}")
    print(f"Training complete | Best Train Acc: {best_acc:.4f}")
    model.load_state_dict(best_wts)
    return model


def get_grad_hgcn_column(
    model,
    data,
    A_before,
    A_after,
    x_before,
    x_after,
    deleted_columns: list
):
    """
    Compute gradients for GIF under column-level feature deletion:
      - g_all: full-graph gradient at pre-deletion
      - g1   : gradient with original features (before deleting columns)
      - g2   : gradient after setting specified columns to zero
    """

    y = data["y"]
    mask_all = data.get("train_mask", torch.ones_like(y, dtype=torch.bool, device=y.device))

    # # 检查删除列在 x_before 中确实非零，在 x_after 中是零
    # with torch.no_grad():
    #     before_vals = x_before[:, deleted_columns]
    #     after_vals  = x_after[:, deleted_columns]
    #     # 删除前应有至少一个非零
    #     assert (before_vals.abs().sum(dim=1) > 0).any().item(), (
    #         "Error: Deleted columns in x_before are already zero."
    #     )
    #     # 删除后应全为零
    #     assert (after_vals.abs().sum(dim=1) == 0).all().item(), (
    #         "Error: Deleted columns in x_after are not fully zero."
    #     )

    # Pre-deletion pass
    model.structure = A_before
    for l in model.layers: l.reapproximate = False
    out1 = model(x_before)
    params = [p for p in model.parameters() if p.requires_grad]
    loss_all = nn.functional.nll_loss(out1[mask_all], y[mask_all], reduction='sum')
    g_all = grad(loss_all, params, retain_graph=True, create_graph=True)

    # Deletion (zero columns) pass
    model.structure = A_after
    for l in model.layers: l.reapproximate = False
    out2 = model(x_after)
    loss_all_after = nn.functional.nll_loss(out2[mask_all], y[mask_all], reduction='sum')
    g2 = grad(loss_all_after, params, retain_graph=True, create_graph=True)

    # g1 就是 g_all，g2 是置零后的
    g1 = g_all

    # # 检查 g2 是否接近零
    # is_g2_zero = torch.all(g2[0].abs() < 1e-6)
    # print(f"[GIF-Col] g2 is zero: {is_g2_zero.item()}")
    # print("[GIF-Col] g2 values:", g2[0])

    return g_all, g1, g2


def hvp(g_all, model, vs):
    elem = sum((g * v).sum() for g, v in zip(g_all, vs))
    return grad(elem, [p for p in model.parameters() if p.requires_grad], create_graph=True)

# ------------------------------------------------------------------
# approx_gif  —— LiSSA 近似逆 Hessian 更新
# ------------------------------------------------------------------
def approx_gif(model, data, A_before, A_after, deleted_column, x_before, x_after, iters=20, damp=1e-2, scale=1e6):
    """
    LiSSA-based approximate GIF update.
    Explicitly takes x_before and x_after features.
    带动态 scale 和 NaN 检测，防止 LiSSA 发散。
    """
    import numpy as np
    t0 = time.time()
    # Compute gradients
    g_all, g1, g2 = get_grad_hgcn_column(model, data, A_before, A_after, x_before, x_after, deleted_column)
    v = [a - b for a, b in zip(g1, g2)]
    h = [vi.clone() for vi in v]
    params = [p for p in model.parameters() if p.requires_grad]

    # Hessian-vector product helper
    def _hvp(vs):
        s = sum((g * vi).sum() for g, vi in zip(g_all, vs))
        return grad(s, params, create_graph=True)

    cur_scale = scale
    # LiSSA iterations（带动态 scale 和 NaN 检测）
    for it in range(iters):
        hv = _hvp(h)

        v_norm  = sum(vi.norm().item()  for vi  in v)
        hv_norm = sum(hvi.norm().item() for hvi in hv)

        if not np.isfinite(hv_norm) or hv_norm == 0:
            print(f"[GIF-Col] hv_norm={hv_norm:.3e} at iter {it}, stopping LiSSA early")
            break

        cur_scale = max(scale, hv_norm / (v_norm + 1e-12))

        with torch.no_grad():
            for i in range(len(h)):
                h[i] = v[i] + (1.0 - damp) * h[i] - hv[i] / cur_scale

        h_norm = sum(hi.norm().item() for hi in h)
        if not np.isfinite(h_norm):
            print(f"[GIF-Col] NaN/Inf in h at iter {it}, resetting to v")
            h = [vi.clone() for vi in v]
            cur_scale = scale
            break

    # Parameter update
    params_change = [hi / cur_scale for hi in h]
    with torch.no_grad():
        for p, delta in zip(params, params_change):
            p.sub_(delta)

    delta_norm = sum(d.norm().item() for d in params_change)
    print(f"[GIF-Col] ||Δparam||₂ = {delta_norm:.4e}, final_scale = {cur_scale:.3e}")
    return time.time() - t0


