#####################
# ========== B. GIF/IF 相关辅助函数 ==========
#####################
import time
import torch
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import grad
import copy
from tqdm import tqdm
import re
def rebuild_structure_after_column_deletion(hyperedges_orig, deleted_names, num_nodes, device):
    """
    删除指定特征列后，移除对应超边并重建稀疏 H。
    返回：(H_t, new_hyperedges)
    """
    orig_count = len(hyperedges_orig)
    # 1) 筛掉属于被删列的超边
    new_hyperedges = {
        hid: nodes
        for hid, nodes in hyperedges_orig.items()
        if hid[0] not in deleted_names
    }
    new_count = len(new_hyperedges)
    removed = orig_count - new_count

    # 构造稀疏张量
    rows, cols, vals = [], [], []
    for i, nodes in enumerate(new_hyperedges.values()):
        for n in nodes:
            rows.append(i); cols.append(n); vals.append(1.0)

    if rows:
        import numpy as np, scipy.sparse as sp
        H_sp = sp.coo_matrix(
            (np.array(vals), (np.array(rows), np.array(cols))),
            shape=(new_count, num_nodes)
        )
        idx = torch.LongTensor(np.vstack((H_sp.row, H_sp.col))).to(device)
        val = torch.FloatTensor(H_sp.data).to(device)
        H_t = torch.sparse_coo_tensor(idx, val, size=H_sp.shape).coalesce().to(device)
    else:
        H_t = torch.sparse_coo_tensor(
            torch.empty((2,0), dtype=torch.int64, device=device),
            torch.tensor([], device=device),
            size=(0, num_nodes)
        ).coalesce()

    # 在这里打印删除结果
    print(f"[ColumnDeletion] Hyperedges before/after: {orig_count} → {new_count} (removed {removed})")
    return H_t, new_hyperedges
def delete_feature_columns_hgat(
    X_tensor: torch.Tensor,
    transformer,
    column_names: list,
    hyperedges: dict,
    device: torch.device
):
    """
    1) 零化 X_tensor 中所有 column_names 对应的编码后特征维度
    2) 调用 rebuild_structure_after_column_deletion 删除 hyperedges + 重建 H
    返回: (X_new, new_hyperedges, H_new)
    """
    # 1) 特征名映射
    try:
        feat_names = transformer.get_feature_names_out().tolist()
    except AttributeError:
        feat_names = transformer.get_feature_names()

    del_idx = []
    for col in column_names:
        idxs = [i for i,f in enumerate(feat_names)
                if col in re.split(r'__|_', f)]
        if idxs:
            print(f"[zero] '{col}' → zero {len(idxs)} dims: {idxs}")
            del_idx += idxs
        else:
            print(f"[warn] '{col}' 未匹配到任何编码特征")
    del_idx = sorted(set(del_idx))
    if del_idx:
        X_tensor[:, del_idx] = 0.0
        print(f"[verify] 零化了 {len(del_idx)} 个特征维度")
    else:
        print("[verify] 无特征被零化")

    # 2) 调用 rebuild_structure_after_column_deletion
    H_new, new_hyp = rebuild_structure_after_column_deletion(
        hyperedges, column_names, X_tensor.size(0), device
    )

    return X_tensor, new_hyp, H_new
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
    """
    训练 HGAT 模型（forward 接收 fts, H），移除了不必要的 dv_inv/de_inv 参数。
    返回：加载了最佳训练准确率权重的模型。
    """
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in tqdm(range(1, num_epochs + 1), desc="Training HGAT", unit="epoch"):
        model.train()
        optimizer.zero_grad()

        output = model(fts, H)
        loss = criterion(output, lbls)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            preds = output.argmax(dim=1)
            train_acc = (preds == lbls).float().mean().item()

        if epoch % print_freq == 0:
            tqdm.write(f"Epoch {epoch}/{num_epochs} | Loss: {loss.item():.4f} | Acc: {train_acc:.4f}")

        if train_acc > best_acc:
            best_acc = train_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s")
    print(f"Best Train Acc: {best_acc:.4f}")

    model.load_state_dict(best_model_wts)
    return model

def get_grad_hgat_col(model, data, unlearn_info):
    """
    计算列删除的梯度差分，并打印置零列数与超边删除数。
    unlearn_info = (deleted_names, deleted_idxs, hyperedges)
    data = {"x","y","H_orig"}
    """
    deleted_names, deleted_idxs, hyperedges = unlearn_info
    x  = data["x"]           # [N, F]
    y  = data["y"]           # [N]
    H  = data["H_orig"]      # [E, N]
    dev = x.device

    params = [p for p in model.parameters() if p.requires_grad]
    mask_all = torch.ones_like(y, dtype=torch.bool, device=dev)

    # —— 原始梯度 g_all —— #
    out1 = model(x, H)
    loss_all = F.cross_entropy(out1[mask_all], y[mask_all], reduction="sum")
    g_all = grad(loss_all, params, create_graph=True, retain_graph=True)

    # —— Post-deletion —— #
    # 1) 把特征矩阵对应列置零
    x2 = x.clone()
    x2[:, deleted_idxs] = 0.0
    print(f"[ColumnDeletion] Zeroed feature columns: {len(deleted_idxs)} → {deleted_names}")

    # 2) 重新构建超图 H2，并打印
    H2, new_hyp = rebuild_structure_after_column_deletion(
        hyperedges, deleted_names, x.size(0), dev
    )

    out2 = model(x2, H2)
    loss_del = F.cross_entropy(out2[mask_all], y[mask_all], reduction="sum")
    g_del = grad(loss_del, params, create_graph=True, retain_graph=True)

    # 差分
    g_del_diff = [ga - gd for ga, gd in zip(g_all, g_del)]
    g_nei_diff = [torch.zeros_like(ga) for ga in g_all]

    return g_all, g_del_diff, g_nei_diff

def hvps(grad_all, model, vs):
    """
    Hessian–vector product: H(grad_all)·vs
    """
    inner = sum((g * v).sum() for g, v in zip(grad_all, vs))
    params = [p for p in model.parameters() if p.requires_grad]
    return autograd.grad(inner, params, create_graph=False, retain_graph=True)


def approx_gif_col(model, data, unlearn_info,
                   iteration=20, damp=1e-2, scale=1e6):
    """
    列删除的 GIF 更新：
      1) g_all, g_del_diff = get_grad_hgat_col(...)
      2) 用 LiSSA 迭代近似 H^{-1}·g_del_diff（带动态 scale 和 NaN 检测）
      3) 应用到模型参数上
    """
    import numpy as np
    t0 = time.time()

    g_all, g_del, _ = get_grad_hgat_col(model, data, unlearn_info)
    # v 初始为差分
    v = [gd.clone() for gd in g_del]
    h = [vi.clone() for vi in v]
    params = [p for p in model.parameters() if p.requires_grad]

    cur_scale = scale
    # LiSSA 迭代（带动态 scale 和 NaN 检测）
    for it in range(iteration):
        hv = hvps(g_all, model, h)

        v_norm  = sum(vi.norm().item()  for vi  in v)
        hv_norm = sum(hvi.norm().item() for hvi in hv)

        if not np.isfinite(hv_norm) or hv_norm == 0:
            print(f"[GIF-Col] hv_norm={hv_norm:.3e} at iter {it}, stopping LiSSA early")
            break

        # 取 max(给定 scale, 自适应 scale)，防止步长过大
        cur_scale = max(scale, hv_norm / (v_norm + 1e-12))

        with torch.no_grad():
            for i in range(len(h)):
                h[i] = v[i] + (1.0 - damp) * h[i] - hv[i] / cur_scale

        # NaN 检测：若 h 出现 NaN/Inf 则回退到 v 并退出
        h_norm = sum(hi.norm().item() for hi in h)
        if not np.isfinite(h_norm):
            print(f"[GIF-Col] NaN/Inf in h at iter {it}, resetting to v")
            h = [vi.clone() for vi in v]
            cur_scale = scale
            break

    # 更新模型参数
    params_change = [hi / cur_scale for hi in h]
    with torch.no_grad():
        for p, delta in zip(params, params_change):
            p.sub_(delta)

    delta_norm = sum(d.norm().item() for d in params_change)
    print(f"[GIF-Col] ||Δparam||₂ = {delta_norm:.4e}, final_scale = {cur_scale:.3e}")

    return time.time() - t0, h
