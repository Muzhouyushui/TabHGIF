import time
import copy

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import grad

from Credit.HGCN import laplacian
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

def get_grad_hgcn(model, data, A_before, A_after, x_before, x_after, deleted_nodes,deleted_neighbors):
    """
    Compute gradients for GIF using explicit pre-/post-deletion features:
      - g_all: full-graph gradient at pre-deletion
      - g1   : deletion-subset gradient at pre-deletion (A_before + x_before)
      - g2   : deletion-subset gradient at post-deletion (A_after + x_after)
    """

    # —— 新增：检查特征置零情况 —— #
    # 确保删除节点在 x_before 上至少有一个非零元素
    nonzero_before = (x_before[deleted_nodes] != 0).any(dim=1)
    assert nonzero_before.all().item(), (
        "Error: Some deleted_nodes in x_before are already zero."
    )
    # 确保删除节点在 x_after 上全为零
    nonzero_after = (x_after[deleted_nodes] != 0).any(dim=1)
    assert not nonzero_after.any().item(), (
        "Error: Some deleted_nodes in x_after are not zero."
    )



    y = data["y"]
    mask_all = data.get("train_mask", torch.ones_like(y, dtype=torch.bool, device=y.device))
    mask_del = torch.zeros_like(mask_all)
    mask_del[deleted_nodes] = True

    # Pre-deletion pass
    model.structure = A_before
    for l in model.layers: l.reapproximate = False
    out1 = model(x_before)
    params = [p for p in model.parameters() if p.requires_grad]
    loss_all = nn.functional.nll_loss(out1[mask_all], y[mask_all], reduction='sum')
    loss_d1  = nn.functional.nll_loss(out1[mask_del], y[mask_del], reduction='sum')
    g_all = grad(loss_all, params, retain_graph=True, create_graph=True)
    g1    = grad(loss_d1,  params, retain_graph=True, create_graph=True)

    # Post-deletion pass
    model.structure = A_after
    for l in model.layers: l.reapproximate = False
    out2 = model(x_after)
    loss_d2 = nn.functional.nll_loss(out2[mask_del], y[mask_del], reduction='sum')
    g2 = grad(loss_d2, params, retain_graph=True, create_graph=True)
    # 检查 g2 是否为零
    is_g2_zero = torch.all(g2[0].abs() < 1e-6)  # 使用一个非常小的阈值来判断是否接近零
    print(f"g2 is zero: {is_g2_zero.item()}")

    return g_all, g1, g2


def hvp(g_all, model, vs):
    elem = sum((g * v).sum() for g, v in zip(g_all, vs))
    return grad(elem, [p for p in model.parameters() if p.requires_grad], create_graph=True)

# ------------------------------------------------------------------
# approx_gif  —— LiSSA 近似逆 Hessian 更新
# ------------------------------------------------------------------
def approx_gif(model, data, A_before, A_after, deleted_neighbors,x_before, x_after, deleted_nodes, iters=20, damp=1e-2, scale=1e6):
    """
    LiSSA-based approximate GIF update.
    Explicitly takes x_before and x_after features.
    """
    t0 = time.time()
    # Compute gradients
    g_all, g1, g2 = get_grad_hgcn(model, data, A_before, A_after, x_before, x_after, deleted_nodes,deleted_neighbors)
    v = [a - b for a, b in zip(g1, g2)]
    h = [vi.clone() for vi in v]
    # Hessian-vector product helper
    def hvp(vs):
        s = sum((g * v).sum() for g, v in zip(g_all, vs))
        return grad(s, [p for p in model.parameters() if p.requires_grad], create_graph=True)
    # LiSSA iterations
    for _ in range(iters):
        hv = hvp(h)
        with torch.no_grad():
            for i in range(len(h)):
                h[i] = v[i] + (1 - damp) * h[i] - hv[i] / scale
    # Parameter update
    with torch.no_grad():
        for p, delta in zip([p for p in model.parameters() if p.requires_grad], h):
            p.sub_(delta / scale)
    print("delta", delta)
    return time.time() - t0



def find_hyperneighbors(hyperedges, deleted, K):
    """
    找到所有与 deleted 中节点共享至少 K 条超边的邻居节点。

    参数：
      hyperedges: List[List[int]]
          超边列表，每个超边是节点索引的列表。
      deleted: Iterable[int]
          被删除节点的索引集合或列表。
      K: int
          共享超边的阈值。

    返回：
      neighbors: List[int]
          所有满足条件的邻居节点索引列表。
    """

    # 转成集合加速判断
    deleted_set = set(deleted)

    # —— 1) 构建节点 → 超边倒排表 —— #
    # node2edges[node] = [eid1, eid2, ...]
    node2edges = defaultdict(list)
    for eid, hedge in enumerate(hyperedges):
        for node in hedge:
            node2edges[node].append(eid)

    # —— 2) 统计候选节点的“共享超边数” —— #
    counter = defaultdict(int)
    for d in deleted_set:
        # 遍历每条与被删节点相连的超边
        for eid in node2edges.get(d, []):
            # 将这条超边上的其它节点计数
            for node in hyperedges[eid]:
                if node not in deleted_set:
                    counter[node] += 1

    # —— 3) 筛选出共享超边数 ≥ K 的节点 —— #
    neighbors = [node for node, cnt in counter.items() if cnt >= K]
    return neighbors