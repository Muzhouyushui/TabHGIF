#####################
# ========== B. GIF/IF 相关辅助函数 ==========
#####################
import time
import numpy as np
import torch
from torch.autograd import grad
# 如果你把 hypergraph_utils.py 放在同目录下：
from Credit.HGNN.HGNN import build_incidence_matrix, compute_degree_vectors
import copy
from tqdm import tqdm
import torch.nn.functional as F

def get_grad_hgnnp(model, data, unlearn_info):
    """
    原签名不变：
    unlearn_info: (deleted_nodes, hyperedges, K)
    计算全图梯度，以及被删节点和其超边邻居的梯度差分。
    返回: g_all, g_del_diff, g_nei_diff
    """
    deleted_nodes, hyperedges, K = unlearn_info
    x      = data["x"]
    y      = data["y"]
    H      = data["H"]
    dv_inv = data["dv_inv"]
    de_inv = data["de_inv"]
    device = x.device

    # 找到邻居
    deleted_neighbors = find_hyperneighbors(hyperedges, deleted_nodes, K)
    # —— 新增输出 —— #
    print(f"[GIF] 共找到 {len(deleted_neighbors)} 个邻居节点 (K={K})")

    # 构造 mask
    mask_all = data.get("train_mask", torch.ones_like(y, dtype=torch.bool, device=device))
    mask_del = torch.zeros_like(mask_all); mask_del[deleted_nodes] = True
    mask_nei = torch.zeros_like(mask_all); mask_nei[deleted_neighbors] = True

    params = [p for p in model.parameters() if p.requires_grad]

    # —— Pre-deletion pass —— #
    out1 = model(x, H, dv_inv, de_inv)
    loss_all  = F.nll_loss(out1[mask_all], y[mask_all], reduction='sum')
    loss_del1 = F.nll_loss(out1[mask_del],  y[mask_del],  reduction='sum')
    loss_nei1 = F.nll_loss(out1[mask_nei],  y[mask_nei],  reduction='sum')

    g_all  = grad(loss_all,  params, retain_graph=True, create_graph=True)
    g_del1 = grad(loss_del1, params, retain_graph=True, create_graph=True)
    g_nei1 = grad(loss_nei1, params, retain_graph=True, create_graph=True)

    # —— Post-deletion pass —— #
    # 置零被删节点特征
    x2 = x.clone()
    x2[deleted_nodes] = 0.0
    # 重建结构
    H2, dv2_np, de2_np, _ = rebuild_structure_after_node_deletion(
        hyperedges, deleted_nodes, x.shape[0], device
    )
    dv2 = torch.as_tensor(dv2_np, device=device, dtype=torch.float)
    de2 = torch.as_tensor(de2_np, device=device, dtype=torch.float)

    out2 = model(x2, H2, dv2, de2)
    loss_del2 = F.nll_loss(out2[mask_del], y[mask_del], reduction='sum')
    loss_nei2 = F.nll_loss(out2[mask_nei], y[mask_nei], reduction='sum')

    g_del2 = grad(loss_del2, params, retain_graph=True, create_graph=True)
    g_nei2 = grad(loss_nei2, params, retain_graph=True, create_graph=True)

    # 差分
    g_del_diff = [d1 - d2 for d1, d2 in zip(g_del1, g_del2)]
    g_nei_diff = [n1 - n2 for n1, n2 in zip(g_nei1, g_nei2)]

    return g_all, g_del_diff, g_nei_diff


def hvps(grad_all, model, vs):
    """Hessian-Vector Product (HVP)"""
    elem_product = 0
    model_params = [p for p in model.parameters() if p.requires_grad]
    for g, v in zip(grad_all, vs):
        elem_product += torch.sum(g * v)
    return_grads = grad(elem_product, model_params, create_graph=True)
    return return_grads
# ---------- 工具：删除节点并重建 H / dv_inv / de_inv ----------
def rebuild_structure_after_node_deletion(hyperedges_orig, deleted_nodes, num_nodes, device):
    """
    根据 deleted_nodes 重新生成:
      • H_sparse  (scipy COO)
      • dv_inv / de_inv (numpy)
      • H_tensor / dv_inv_t / de_inv_t (torch)

    hyperedges_orig : dict  {hyperedge_id: [node_id, ...]}
    deleted_nodes   : 1-D np.ndarray/int list
    num_nodes       : 原节点总数 (不缩减行数，只把被删行置零)
    """
    del_set = set(deleted_nodes.tolist() if isinstance(deleted_nodes, np.ndarray) else deleted_nodes)

    # 1) 过滤掉被删节点；若某条超边被删空则直接丢掉
    new_hyperedges = {}
    for he_id, nodes in hyperedges_orig.items():
        kept = [n for n in nodes if n not in del_set]
        if kept:                      # 至少还剩 1 个节点
            new_hyperedges[he_id] = kept

    # 2) incidence matrix  &  degree vectors
    H_sp = build_incidence_matrix(new_hyperedges, num_nodes)
    dv_inv_np, de_inv_np = compute_degree_vectors(H_sp)

    # 3) 转成 torch.sparse 形式
    H_coo = H_sp.tocoo()
    idx   = torch.LongTensor(np.vstack((H_coo.row, H_coo.col))).to(device)
    val   = torch.FloatTensor(H_coo.data).to(device)
    H_t   = torch.sparse_coo_tensor(idx, val, size=H_coo.shape).coalesce().to(device)

    dv_inv_t = torch.FloatTensor(dv_inv_np).to(device)
    de_inv_t = torch.FloatTensor(de_inv_np).to(device)

    return H_t, dv_inv_t, de_inv_t, new_hyperedges


def approx_gif(model, data, unlearn_info, iteration=5, damp=0.01, scale=1e7):
    """
    GIF 逆向更新（LiSSA 近似），带动态 scale 和 NaN 检测防止发散：
    unlearn_info: (deleted_nodes, hyperedges, K)
    返回: (unlearning_time, f1_unlearn)
    """
    import numpy as np
    start_time = time.time()

    # 1) 梯度差分
    grad_all, grad_del, grad_nei = get_grad_hgnnp(model, data, unlearn_info)
    # 合并 v 向量
    v = [gd + gn for gd, gn in zip(grad_del, grad_nei)]
    h = [vi.clone() for vi in v]
    params = [p for p in model.parameters() if p.requires_grad]

    # 温和移除系数
    alpha = 0.5
    cur_scale = scale

    # 2) LiSSA 迭代（带动态 scale 和 NaN 检测）
    for it in range(iteration):
        hv = hvps(grad_all, model, h)

        # 动态自适应 scale
        v_norm  = sum(vi.norm().item()  for vi  in v)
        hv_norm = sum(hvi.norm().item() for hvi in hv)

        if not np.isfinite(hv_norm) or hv_norm == 0:
            print(f"[GIF] hv_norm={hv_norm:.3e} at iter {it}, stopping LiSSA early")
            break

        cur_scale = hv_norm / (v_norm + 1e-12)

        with torch.no_grad():
            for i in range(len(h)):
                h[i] = (
                    v[i]
                    + (1.0 - damp) * h[i]
                    - alpha * hv[i] / cur_scale
                )

        # NaN 检测：若 h 出现 NaN/Inf 则回退到 v 并提前退出
        h_norm = sum(hi.norm().item() for hi in h)
        if not np.isfinite(h_norm):
            print(f"[GIF] NaN/Inf in h at iter {it}, resetting to v")
            h = [vi.clone() for vi in v]
            cur_scale = scale
            break

    # 3) 应用到模型参数
    with torch.no_grad():
        for p, hi in zip(params, h):
            p.sub_(hi / cur_scale)

    delta_norm = sum(hi.norm().item() / cur_scale for hi in h)
    print(f"[GIF] ||Δparam||₂ = {delta_norm:.4e}, scale = {cur_scale:.3e}")

    unlearning_time = time.time() - start_time

    # 保持原 placeholder，由外部 fill-in
    f1_unlearn = -1

    return unlearning_time, f1_unlearn
#####################
# ========== A. 常规训练函数 ==========
#####################
def train_model(model, criterion, optimizer, scheduler, fts, lbls, H, dv_inv, de_inv,
                num_epochs=200, print_freq=10):
    """
    训练 HGNN 模型。
    返回加载最佳权重后的模型
    """
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in tqdm(range(num_epochs), desc="Training Epochs", unit="epoch"):
        model.train()
        optimizer.zero_grad()

        # 隐式传播在模型内部完成
        output = model(fts, H, dv_inv, de_inv)
        loss = criterion(output, lbls)
        loss.backward()
        optimizer.step()
        scheduler.step()

        _, preds = torch.max(output, 1)
        train_acc = (preds == lbls).float().mean().item()

        # if epoch % print_freq == 0:
        #     tqdm.write(f"Epoch {epoch}/{num_epochs-1} | Loss: {loss.item():.4f} | Train Acc: {train_acc:.4f}")

        if train_acc > best_acc:
            best_acc = train_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s")
    print(f"Best Train Acc: {best_acc:.4f}")

    model.load_state_dict(best_model_wts)
    return model



# def find_hyperneighbors(hyperedges: dict, deleted: list, K: int):
#     """
#     找到所有与 deleted 中节点共享至少 K 条不同超边的邻居节点。
#     """
#     deleted_set = set(deleted)
#     counter = {}
#     for hedge in hyperedges.values():
#         if deleted_set.intersection(hedge):
#             for node in hedge:
#                 if node not in deleted_set:
#                     counter[node] = counter.get(node, 0) + 1
#     return [n for n, cnt in counter.items() if cnt >= K]
from collections import defaultdict

def find_hyperneighbors(hyperedges: dict, deleted: list, K: int):
    """
    找到所有与任一被删节点共享至少 K 条超边的邻居节点。
    优化：先构建 node→edge 反向索引，再针对每个 deleted 扫描相关超边。

    参数:
      hyperedges: Dict[int, List[int]]  超边字典，键为超边ID，值为节点列表
      deleted:    List[int]             被删节点列表
      K:          int                   阈值，共享至少 K 条超边

    返回:
      List[int]   邻居节点列表
    """
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