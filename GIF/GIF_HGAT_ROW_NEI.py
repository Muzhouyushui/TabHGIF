#####################
# ========== B. GIF/IF 相关辅助函数 ==========
#####################
import time
import numpy as np
import torch
from torch.autograd import grad
# 如果你把 hypergraph_utils.py 放在同目录下：
from HGNNs_Model.HGNNP import build_incidence_matrix, compute_degree_vectors
import copy
from tqdm import tqdm
import torch.nn.functional as F
import torch.autograd as autograd # Explicitly import autograd
import torch.nn as nn
from collections import defaultdict
import scipy.sparse as sp

def train_model(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    fts: torch.Tensor,
    lbls: torch.Tensor,
    H: torch.sparse.FloatTensor,
    num_epochs: int = 200,
    print_freq: int = 10
) -> torch.nn.Module:

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in tqdm(range(1, num_epochs + 1), desc="Training HGAT", unit="epoch"):
        model.train()
        optimizer.zero_grad()

        # HGAT 前向：只需要特征和超图结构
        output = model(fts, H)
        loss = criterion(output, lbls)
        loss.backward()

        # 可选梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        # 计算训练准确率
        with torch.no_grad():
            preds = output.argmax(dim=1)
            train_acc = (preds == lbls).float().mean().item()

        # 打印训练进度
        if epoch % print_freq == 0:
            tqdm.write(f"Epoch {epoch}/{num_epochs} | "
                       f"Loss: {loss.item():.4f} | Acc: {train_acc:.4f}")

        # 更新最佳权重
        if train_acc > best_acc:
            best_acc = train_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s")
    print(f"Best Train Acc: {best_acc:.4f}")

    # 加载最佳权重
    model.load_state_dict(best_model_wts)
    return model


def find_hyperneighbors(hyperedges: dict, deleted: list, K: int):

    # 1) 构建反向索引：node → 所属超边ID列表
    node2edges = defaultdict(list)
    for eid, hedge in hyperedges.items():
        for node in hedge:
            node2edges[node].append(eid)

    neighbors = set()
    # 2) 对每个被删节点，扫描它所在的超边
    for d in deleted:
        cnt = {}
        for eid in node2edges.get(d, []):
            for node in hyperedges[eid]:
                if node != d:
                    cnt[node] = cnt.get(node, 0) + 1
        # 3) 阈值过滤
        for node, c in cnt.items():
            if c >= K:
                neighbors.add(node)

    return list(neighbors)

def rebuild_structure_after_node_deletion(hyperedges_orig, deleted_nodes, num_nodes, device):

    # 1) 构造删除集合
    del_set = set(deleted_nodes.tolist() if isinstance(deleted_nodes, np.ndarray) else deleted_nodes)
    # 2) 筛选出保留超边（保持原 ID→节点 列表映射以便外部追踪）
    new_hyperedges = {
        hid: [n for n in nodes if n not in del_set]
        for hid, nodes in hyperedges_orig.items()
        if any(n not in del_set for n in nodes)
    }

    # 3) 将 new_hyperedges 的值列表枚举为连续行号
    edges_list = list(new_hyperedges.values())
    rows, cols, vals = [], [], []
    for row_idx, nodes in enumerate(edges_list):
        for n in nodes:
            rows.append(row_idx)   # 连续的行号
            cols.append(n)         # 原始节点索引
            vals.append(1.0)

    # 4) 构建 SciPy COO 矩阵
    if rows:
        rows_arr = np.array(rows, dtype=int)
        cols_arr = np.array(cols, dtype=int)
        vals_arr = np.array(vals, dtype=float)
        H_sp = sp.coo_matrix((vals_arr, (rows_arr, cols_arr)),
                              shape=(len(edges_list), num_nodes))
    else:
        H_sp = sp.coo_matrix(([], ([], [])), shape=(len(edges_list), num_nodes))

    # 5) 转为 torch.sparse_coo_tensor
    idx = torch.LongTensor(np.vstack((H_sp.row, H_sp.col))).to(device)
    val = torch.FloatTensor(H_sp.data).to(device)
    H_t = torch.sparse_coo_tensor(idx, val, size=H_sp.shape).coalesce().to(device)

    return H_t, new_hyperedges


def get_grad_hgat(model, data, unlearn_info):

    deleted_nodes, hyperedges, K = unlearn_info
    x      = data["x"]
    y      = data["y"]
    H      = data["H_orig"]
    device = x.device

    # 找邻居
    deleted_list = deleted_nodes.tolist() if isinstance(deleted_nodes, torch.Tensor) else deleted_nodes
    neighbors = find_hyperneighbors(hyperedges, deleted_list, K)
    print(f"[GIF] Found {len(neighbors)} neighbor nodes (K={K})")

    # 构造 mask
    mask_all = data.get("train_mask", torch.ones_like(y, dtype=torch.bool, device=device))
    mask_del = torch.zeros_like(mask_all); mask_del[deleted_list] = True
    mask_nei = torch.zeros_like(mask_all); mask_nei[neighbors]    = True

    # 准备参数列表和名称
    named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    param_names = [n for n, _ in named_params]
    params      = [p for _, p in named_params]

    # —— Pre-deletion —— #
    out1      = model(x, H)
    loss_all  = F.cross_entropy(out1[mask_all], y[mask_all], reduction='sum')
    loss_del1 = F.cross_entropy(out1[mask_del],  y[mask_del],  reduction='sum')
    loss_nei1 = F.cross_entropy(out1[mask_nei],  y[mask_nei],  reduction='sum')

    g_all  = grad(loss_all,  params, retain_graph=True, create_graph=True)
    g_del1 = grad(loss_del1, params, retain_graph=True, create_graph=True)
    g_nei1 = grad(loss_nei1, params, retain_graph=True, create_graph=True)

    # —— Post-deletion —— #
    x2 = x.clone()
    x2[deleted_list] = 0.0
    H2, _ = rebuild_structure_after_node_deletion(
        hyperedges, deleted_list, x.size(0), device
    )
    out2      = model(x2, H2)
    loss_del2 = F.cross_entropy(out2[mask_del], y[mask_del], reduction='sum')
    loss_nei2 = F.cross_entropy(out2[mask_nei], y[mask_nei], reduction='sum')

    g_del2 = grad(loss_del2, params, retain_graph=True, create_graph=True)
    g_nei2 = grad(loss_nei2, params, retain_graph=True, create_graph=True)

    # 取差分
    g_del_diff = [d1 - d2 for d1, d2 in zip(g_del1, g_del2)]
    g_nei_diff = [n1 - n2 for n1, n2 in zip(g_nei1, g_nei2)]


    return g_all, g_del_diff, g_nei_diff

def hvps(grad_all, model, vs):

    # 构造标量 inner = ∑_i grad_all[i] · vs[i]
    inner = sum((g * v).sum() for g, v in zip(grad_all, vs))
    # params 与 get_grad_hgat 保持一致
    params = [p for p in model.parameters() if p.requires_grad]
    # 对 inner 求梯度：得到 H·v
    # return autograd.grad(
    #     inner, params,
    #     create_graph=True,
    #     retain_graph=True
    # )
    return autograd.grad(
        inner, params,
        create_graph=False,
        # create_graph=True,
        retain_graph=True
    )



def approx_gif(model, data, unlearn_info,
               iteration=20, damp=1e-2, scale=1e6):


    # 1) 先拿到所有一次梯度
    g_all, g_del, g_nei = get_grad_hgat(model, data, unlearn_info)
    t0 = time.time()

    # v = g_del + g_nei
    v      = [gd + gn for gd, gn in zip(g_del, g_nei)]
    # h 初始化为 v 的拷贝
    h      = [vi.clone()       for vi in v]
    params = [p for p in model.parameters() if p.requires_grad]

    # 2) LiSSA 迭代
    for _ in range(iteration):
        hv = hvps(g_all, model, h)   # 计算 H·h

        for i in range(len(h)):
            # h_{t+1} = v + (1–damp)·h_t – hv/scale
            h[i] = v[i] + (1 - damp) * h[i] - hv[i] / scale

    # 3) 将近似的 H^{-1}·v 应用到模型参数上
    with torch.no_grad():
        for p, hi in zip(params, h):
            p.sub_(hi / scale)
    t1=time.time()
    elapsed = time.time() - t0
    print(t1-t0)
    return elapsed, h