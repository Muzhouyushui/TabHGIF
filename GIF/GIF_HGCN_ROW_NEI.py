import time
import copy

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import grad

from HGCN.HyperGCN import laplacian
from collections import defaultdict


# ——— A. 训练函数 ———
def apply_node_deletion_unlearning(X, edge_list, deleted_nodes,
                                   mediators=True, device="cpu"):
    """
    · new_X           : 置零后的特征        (Tensor, on `device`)
    · edge_list_new   : 删点后的超边列表     (python list)
    · A_new           : 删点后的 Laplacian  (torch.sparse, on `device`)
    """
    new_X = X.clone()
    new_X[deleted_nodes] = 0.

    del_set = set(deleted_nodes.tolist())
    edge_list_new = [ [v for v in e if v not in del_set]   # 剔除 deleted
                      for e in edge_list ]
    edge_list_new = [e for e in edge_list_new if len(e) >= 2]

    A_new = laplacian(edge_list_new, new_X.cpu().numpy(), mediators).to(device)

    return new_X.to(device), edge_list_new, A_new


def train_model(model, criterion, optimizer, scheduler,
                fts, lbls, num_epochs=200, print_freq=None):
    # 保存最佳权重与准确率
    best_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for _ in tqdm(range(num_epochs)):
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

        # 更新最佳权重
        if acc > best_acc:
            best_acc = acc
            best_wts = copy.deepcopy(model.state_dict())

    # 加载最佳权重并输出最终最佳准确率
    model.load_state_dict(best_wts)
    print(f"Best Train Acc: {best_acc:.4f}")
    return model


def get_grad_hgcn(model, data,
                  A_before, A_after,
                  x_before, x_after,
                  deleted_nodes, deleted_neighbors):
    """
    Compute GIF gradients for both deleted nodes and their neighbors.
    Returns:
      g_all      : list of grads for full‐graph loss before deletion
      g_del_diff : list of gradient diffs for deleted nodes (pre − post)
      g_nei_diff : list of gradient diffs for neighbors     (pre − post)
    """
    y      = data["y"]
    device = y.device

    # Masks
    mask_all = data.get("train_mask",
                        torch.ones_like(y, dtype=torch.bool, device=device))
    mask_del = torch.zeros_like(mask_all);  mask_del[deleted_nodes] = True
    mask_nei = torch.zeros_like(mask_all);  mask_nei[deleted_neighbors] = True

    # Collect trainable parameters
    params = [p for p in model.parameters() if p.requires_grad]

    # --- Pre-deletion pass ---
    model.structure = A_before
    for l in model.layers: l.reapproximate = False
    out1 = model(x_before)

    loss_all  = nn.functional.nll_loss(out1[mask_all],  y[mask_all],  reduction='sum')
    loss_del1 = nn.functional.nll_loss(out1[mask_del],   y[mask_del],   reduction='sum')
    loss_nei1 = nn.functional.nll_loss(out1[mask_nei],   y[mask_nei],   reduction='sum')

    g_all  = grad(loss_all,  params, retain_graph=True, create_graph=True)
    g_del1 = grad(loss_del1, params, retain_graph=True, create_graph=True)
    g_nei1 = grad(loss_nei1, params, retain_graph=True, create_graph=True)

    # --- Post-deletion pass ---
    model.structure = A_after
    for l in model.layers: l.reapproximate = False
    out2 = model(x_after)

    loss_del2 = nn.functional.nll_loss(out2[mask_del], y[mask_del], reduction='sum')
    loss_nei2 = nn.functional.nll_loss(out2[mask_nei], y[mask_nei], reduction='sum')

    g_del2 = grad(loss_del2, params, retain_graph=True, create_graph=True)
    g_nei2 = grad(loss_nei2, params, retain_graph=True, create_graph=True)

    # Gradient differences
    g_del_diff = [d1 - d2 for d1, d2 in zip(g_del1, g_del2)]
    g_nei_diff = [n1 - n2 for n1, n2 in zip(g_nei1, g_nei2)]

    return g_all, g_del_diff, g_nei_diff


# ------------------------------------------------------------------
# approx_gif  —— LiSSA 近似逆 Hessian 更新
# ------------------------------------------------------------------
def approx_gif(model, data,
               A_before, A_after,
               deleted_nodes, deleted_neighbors,
               x_before, x_after,
               iters=20, damp=1e-2, scale=1e6):
    """
    LiSSA-based approximate GIF update that compensates both
    deleted nodes and their neighbors.
    """
    t0 = time.time()

    # 1) Compute base gradients and diffs
    g_all, g_del_diff, g_nei_diff = get_grad_hgcn(
        model, data, A_before, A_after,
        x_before, x_after,
        deleted_nodes, deleted_neighbors
    )


    # ← 新增这一段，打印所有参数上邻居梯度差的总 L2 范数
    total_nei_norm = torch.sqrt(sum((diff.norm() ** 2) for diff in g_nei_diff))
    print(f"Total neighbor gradient change norm: {total_nei_norm.item():.6e}")

    # 2) Combine deleted‐node and neighbor diffs into one vector v
    v = [gd + gn for gd, gn in zip(g_del_diff, g_nei_diff)]
    h = [vi.clone() for vi in v]

    # 3) Hessian-vector product w.r.t. g_all
    params = [p for p in model.parameters() if p.requires_grad]
    def hvp(vs):
        hvp_scalar = sum((g * v).sum() for g, v in zip(g_all, vs))
        return grad(hvp_scalar, params, create_graph=True)

    # 4) LiSSA iterations
    for _ in range(iters):
        hv = hvp(h)
        with torch.no_grad():
            for i in range(len(h)):
                h[i] = v[i] + (1 - damp) * h[i] - hv[i] / scale

    # 5) Apply approximate inverse-Hessian update
    with torch.no_grad():
        for p, delta in zip(params, h):
            p.sub_(delta / scale)

    print(f"GIF update done in {time.time() - t0:.3f}s, ||Δparam[0]||₂ = {torch.norm(h[0]).item():.4e}")
    return time.time() - t0



def find_hyperneighbors(hyperedges, deleted, K):
    """
    找到所有与 deleted 中任一节点共享至少 K 条不同超边的邻居节点。
    优化：按每个被删节点单独统计，再取并集，使用反向索引加速。

    参数：
      hyperedges: List[List[int]]    超边列表，每个超边是节点列表
      deleted:    Iterable[int]       被删节点列表
      K:          int                 阈值，共享至少 K 条超边

    返回：
      neighbors: List[int]
    """
    # 1) 构建反向索引：node -> 所属超边索引列表
    node2edges = defaultdict(list)
    for eid, hedge in enumerate(hyperedges):
        for node in hedge:
            node2edges[node].append(eid)

    neighbors = set()
    deleted_set = set(deleted)

    # 2) 对每个被删节点，扫描它关联的超边
    for d in deleted:
        cnt = defaultdict(int)
        for eid in node2edges.get(d, []):
            for node in hyperedges[eid]:
                if node != d:
                    cnt[node] += 1
        # 3) 阈值过滤，并集累加
        for node, c in cnt.items():
            if c >= K:
                neighbors.add(node)

    return list(neighbors)