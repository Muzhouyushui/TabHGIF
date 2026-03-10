import copy
from tqdm import tqdm
####################
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
        #
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





import time
import torch
from torch.autograd import grad

def hvps(grad_all, model, vs):
    """
    Compute Hessian-vector products via gradient of the dot product <grad_all, vs>.
    Args:
        grad_all: list of gradients of the full-loss w.r.t. model parameters
        model   : the Hypergraph model
        vs      : list of vectors (one per parameter) to multiply with Hessian
    Returns:
        list of Hessian-vector products, one per model parameter
    """
    model_params = [p for p in model.parameters() if p.requires_grad]
    # accumulate scalar element = sum_i <grad_all[i], vs[i]>
    elem = None
    for g, v in zip(grad_all, vs):
        prod = torch.sum(g * v)
        elem = prod if elem is None else elem + prod
    # compute gradient of elem w.r.t. parameters = Hv
    hv = torch.autograd.grad(elem, model_params, retain_graph=True, create_graph=False)
    return hv


def approx_gif_col(model, criterion, batch_before, batch_after,
                   cg_iters=80, damping=0.01, scale=1e7):
    """
    Column-level GIF unlearning: approximate parameter update v = H^{-1} \Delta g.

    Args:
        model        : the Hypergraph model (e.g., HyperGCN)
        criterion    : loss function, e.g., nn.CrossEntropyLoss()
        batch_before : tuple (fts, H, dv_inv, de_inv, labels) before column removal
        batch_after  : tuple (fts, H, dv_inv, de_inv, labels) after column removal
        cg_iters     : number of LiSSA (CG-style) iterations
        damping      : damping factor for stability
        tol          : tolerance for early stopping (currently unused)
        scale        : scaling factor for Hessian-vector products

    Returns:
        v_estimate: list of parameter update tensors (same order as model.parameters())
    """
    # 1) Gradient before column removal
    model.zero_grad()
    fts_b, H_b, dv_inv_b, de_inv_b, labels_b = batch_before
    logits_b = model(fts_b, H_b, dv_inv_b, de_inv_b)
    loss_b = criterion(logits_b, labels_b)
    params = [p for p in model.parameters() if p.requires_grad]
    grad_before = grad(loss_b, params, retain_graph=True, create_graph=True)
    # set grad_all for Hessian estimation
    grad_all = [g.clone() for g in grad_before]

    # 2) Gradient after column removal
    model.zero_grad()
    fts_a, H_a, dv_inv_a, de_inv_a, labels_a = batch_after
    logits_a = model(fts_a, H_a, dv_inv_a, de_inv_a)
    loss_a = criterion(logits_a, labels_a)
    grad_after = grad(loss_a, params, retain_graph=True, create_graph=True)

    # 3) Delta gradient v = grad_before - grad_after
    v = [g1 - g2 for g1, g2 in zip(grad_before, grad_after)]

    # 4) LiSSA-style approximation of H^{-1} v
    v_est = [vi.clone() for vi in v]
    for _ in range(cg_iters):
        hv = hvps(grad_all, model, v_est)
        with torch.no_grad():
            for i in range(len(v_est)):
                v_est[i] = v[i] + (1.0 - damping) * v_est[i] - hv[i] / scale
    # return the double-precision update vectors
    # return v_est
    params_change = [vi / scale for vi in v_est]
    return params_change