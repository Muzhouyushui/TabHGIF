#####################
# ========== B. GIF/IF-related helper functions ==========
#####################
import time
import numpy as np
import torch
from torch.autograd import grad
# If you place hypergraph_utils.py in the same directory:
from bank.HGNN.HGNN import build_incidence_matrix, compute_degree_vectors
import copy
from tqdm import tqdm
import torch.nn.functional as F

def get_grad_hgnnp(model, data, unlearn_info):
    """
    Original signature unchanged:
    unlearn_info: (deleted_nodes, hyperedges, K)
    Compute the full-graph gradient, as well as the gradient differences
    for deleted nodes and their hyperedge neighbors.
    Return: g_all, g_del_diff, g_nei_diff
    """
    deleted_nodes, hyperedges, K = unlearn_info
    x      = data["x"]
    y      = data["y"]
    H      = data["H"]
    dv_inv = data["dv_inv"]
    de_inv = data["de_inv"]
    device = x.device

    # Find neighbors
    deleted_neighbors = find_hyperneighbors(hyperedges, deleted_nodes, K)
    # —— New output —— #
    print(f"[GIF] Found {len(deleted_neighbors)} neighbor nodes in total (K={K})")

    # Build masks
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
    # Zero out the features of deleted nodes
    x2 = x.clone()
    x2[deleted_nodes] = 0.0
    # Rebuild structure
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

    # Differences
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

# ---------- Utility: delete nodes and rebuild H / dv_inv / de_inv ----------
def rebuild_structure_after_node_deletion(hyperedges_orig, deleted_nodes, num_nodes, device):
    """
    Regenerate the following according to deleted_nodes:
      • H_sparse  (scipy COO)
      • dv_inv / de_inv (numpy)
      • H_tensor / dv_inv_t / de_inv_t (torch)

    hyperedges_orig : dict  {hyperedge_id: [node_id, ...]}
    deleted_nodes   : 1-D np.ndarray/int list
    num_nodes       : original total number of nodes (do not shrink rows, only zero out deleted rows)
    """
    del_set = set(deleted_nodes.tolist() if isinstance(deleted_nodes, np.ndarray) else deleted_nodes)

    # 1) Filter out deleted nodes; if a hyperedge becomes empty after deletion, discard it directly
    new_hyperedges = {}
    for he_id, nodes in hyperedges_orig.items():
        kept = [n for n in nodes if n not in del_set]
        if kept:                      # at least 1 node remains
            new_hyperedges[he_id] = kept

    # 2) incidence matrix & degree vectors
    H_sp = build_incidence_matrix(new_hyperedges, num_nodes)
    dv_inv_np, de_inv_np = compute_degree_vectors(H_sp)

    # 3) Convert to torch.sparse format
    H_coo = H_sp.tocoo()
    idx   = torch.LongTensor(np.vstack((H_coo.row, H_coo.col))).to(device)
    val   = torch.FloatTensor(H_coo.data).to(device)
    H_t   = torch.sparse_coo_tensor(idx, val, size=H_coo.shape).coalesce().to(device)

    dv_inv_t = torch.FloatTensor(dv_inv_np).to(device)
    de_inv_t = torch.FloatTensor(de_inv_np).to(device)

    return H_t, dv_inv_t, de_inv_t, new_hyperedges


def approx_gif(model, data, unlearn_info, iteration=5, damp=0.01, scale=1e7):
    """
    GIF inverse update (LiSSA approximation), original signature unchanged:
    unlearn_info: (deleted_nodes, hyperedges, K)
    Return: (unlearning_time, f1_unlearn)
    """
    start_time = time.time()

    # 1) Gradient differences
    grad_all, grad_del, grad_nei = get_grad_hgnnp(model, data, unlearn_info)
    # Merge v vector
    v = [gd + gn for gd, gn in zip(grad_del, grad_nei)]
    h = [vi.clone() for vi in v]
    params = [p for p in model.parameters() if p.requires_grad]

    # 2) LiSSA iteration
    for it in range(iteration):
        hv = hvps(grad_all, model, h)

        # Dynamic adaptive scale
        v_norm  = sum(vi.norm().item()  for vi  in v)
        hv_norm = sum(hvi.norm().item() for hvi in hv)
        scale   = hv_norm / (v_norm + 1e-12)
        # print(f"[GIF] iter {it}: ‖v‖={v_norm:.3e}, ‖Hv‖={hv_norm:.3e}, scale={scale:.3e}")

        # Mild removal coefficient
        alpha = 0.5

        with torch.no_grad():
            for i in range(len(h)):
                h[i] = (
                    v[i]
                    + (1.0 - damp) * h[i]
                    - alpha * hv[i] / scale
                )

    # 3) Apply to model parameters
    with torch.no_grad():
        for p, hi in zip(params, h):
            p.sub_(hi / scale)

    unlearning_time = time.time() - start_time

    # Keep original placeholder, filled in externally
    f1_unlearn = -1

    return unlearning_time, f1_unlearn

#####################
# ========== A. Standard training function ==========
#####################
def train_model(model, criterion, optimizer, scheduler, fts, lbls, H, dv_inv, de_inv,
                num_epochs=200, print_freq=10):
    """
    Train the HGNN model.
    Return the model loaded with the best weights.
    """
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in tqdm(range(num_epochs), desc="Training Epochs", unit="epoch"):
        model.train()
        optimizer.zero_grad()

        # Implicit propagation is completed inside the model
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
#     Find all neighbor nodes that share at least K different hyperedges with nodes in deleted.
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
    Find all neighbor nodes that share at least K hyperedges with any deleted node.
    Optimization: first build a reverse index node→edge, then scan related hyperedges for each deleted node.

    Parameters:
      hyperedges: Dict[int, List[int]]  Hyperedge dictionary, where the key is hyperedge ID and the value is the node list
      deleted:    List[int]             List of deleted nodes
      K:          int                   Threshold: share at least K hyperedges

    Returns:
      List[int]   List of neighbor nodes
    """
    # 1) Build reverse index: node → list of incident hyperedge IDs
    node2edges = defaultdict(list)
    for eid, hedge in hyperedges.items():
        for node in hedge:
            node2edges[node].append(eid)

    neighbors = set()
    # 2) For each deleted node, scan the hyperedges it belongs to
    for d in deleted:
        cnt = {}
        for eid in node2edges.get(d, []):
            for node in hyperedges[eid]:
                if node != d:
                    cnt[node] = cnt.get(node, 0) + 1
        # 3) Threshold filtering
        for node, c in cnt.items():
            if c >= K:
                neighbors.add(node)

    return list(neighbors)