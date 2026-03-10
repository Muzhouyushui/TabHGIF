#####################
# ========== B. GIF/IF 相关辅助函数 ==========
#####################
import time
import numpy as np
import torch
from torch.autograd import grad
import torch.nn as nn
# 如果你把 hypergraph_utils.py 放在同目录下：
from HGNNs_Model.HGNNP import build_incidence_matrix, compute_degree_vectors
import copy
from tqdm import tqdm

def get_grad_hgnn(model, data, unlearn_info=None):
    """
    计算 (grad_all, grad_del1, grad_del2) 三组梯度，供 GIF/IF 使用。
    假设:
      model.reason_once(data)         -> 删除前前向
      model.reason_once_unlearn(data) -> 删除后前向

    data 中至少包含:
      data["x"], data["y"], data["train_mask"]    # 用于训练/计算loss
      data["H"], data["dv_inv"], data["de_inv"]    # 超图信息

    unlearn_info: (deleted_nodes, ..., ...)
    """
    # 1) 正常状态前向传播
    out1 = model.reason_once(data)
    # 2) unlearn 状态前向传播
    out2 = model.reason_once_unlearn(data)

    # unlearn_info[0] = 要删除的节点
    deleted_nodes = unlearn_info[0]
    device = data["x"].device

    # 构造“删除节点”掩码
    mask_del = torch.zeros_like(data["train_mask"], dtype=torch.bool, device=device)
    mask_del[deleted_nodes] = True
    mask_del = mask_del & data["train_mask"]  # 只在训练集中考虑

    # ========== 计算 loss ==========
    # 全部训练集的 loss
    loss_all = nn.functional.nll_loss(out1[data["train_mask"]], data["y"][data["train_mask"]], reduction='sum')

    # 删除节点(删除前) loss
    if mask_del.sum() > 0:
        loss_del1 = nn.functional.nll_loss(out1[mask_del], data["y"][mask_del], reduction='sum')
    else:
        loss_del1 = 0.0

    # 删除节点(删除后) loss
    if mask_del.sum() > 0:
        loss_del2 = nn.functional.nll_loss(out2[mask_del], data["y"][mask_del], reduction='sum')
    else:
        loss_del2 = 0.0

    # ========== 求梯度 ==========
    model_params = [p for p in model.parameters() if p.requires_grad]
    grad_all = grad(loss_all, model_params, retain_graph=True, create_graph=True)

    if isinstance(loss_del1, float):
        grad_del1 = [torch.zeros_like(p) for p in model_params]
    else:
        grad_del1 = grad(loss_del1, model_params, retain_graph=True, create_graph=True)

    if isinstance(loss_del2, float):
        grad_del2 = [torch.zeros_like(p) for p in model_params]
    else:
        grad_del2 = grad(loss_del2, model_params, retain_graph=True, create_graph=True)

    return grad_all, grad_del1, grad_del2


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
    GIF 逆向更新过程（带动态自适应 scale 和温和移除系数 alpha）
    unlearn_info: (deleted_nodes, ...)
    返回: (unlearning_time, f1_unlearn)
    """
    start_time = time.time()

    # 1) 计算所有样本梯度与删除前后差分
    grad_all, grad_del1, grad_del2 = get_grad_hgnn(model, data, unlearn_info)
    v = [g1 - g2 for (g1, g2) in zip(grad_del1, grad_del2)]
    model_params = [p for p in model.parameters() if p.requires_grad]
    h_estimate = [vi.clone() for vi in v]

    # 2) 迭代求解 H^{-1} v 近似
    for it in range(iteration):
        hv = hvps(grad_all, model, h_estimate)

        # 动态自适应 scale
        v_norm  = sum(vi.norm().item() for vi in v)
        hv_norm = sum(hvi.norm().item() for hvi in hv)
        scale   = hv_norm / (v_norm + 1e-12)
        print(f"[GIF] iter {it}: ‖v‖={v_norm:.3e}, ‖Hv‖={hv_norm:.3e}, 使用 scale={scale:.3e}")

        # 温和移除系数
        alpha = 0.5

        with torch.no_grad():
            for i in range(len(h_estimate)):
                # h = v + (1−damp)h_prev − alpha * (Hv/scale)
                h_estimate[i] = (
                    v[i]
                    + (1.0 - damp) * h_estimate[i]
                    - alpha * hv[i] / scale
                )

    # 3) 将近似影响量应用到模型参数上
    params_change = [h / scale for h in h_estimate]
    with torch.no_grad():
        for p, delta in zip(model_params, params_change):
            p -= delta

    unlearning_time = time.time() - start_time
    f1_unlearn = -1  # 由调用方在 main() 里评估填充
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

        if epoch % print_freq == 0:
            tqdm.write(f"Epoch {epoch}/{num_epochs-1} | Loss: {loss.item():.4f} | Train Acc: {train_acc:.4f}")

        if train_acc > best_acc:
            best_acc = train_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s")
    print(f"Best Train Acc: {best_acc:.4f}")

    model.load_state_dict(best_model_wts)
    return model


