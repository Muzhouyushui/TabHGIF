#####################
# ========== B. GIF/IF-related helper functions ==========
#####################
import time
import numpy as np
import torch
from torch.autograd import grad
# If you place hypergraph_utils.py in the same directory:
import copy
from tqdm import tqdm
import torch.nn.functional as F
import torch.autograd as autograd  # Explicitly import autograd
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
    """
    Train the HGAT model (forward takes fts and H), with unnecessary dv_inv/de_inv
    parameters removed.
    Return: the model loaded with the best training-accuracy weights.
    """
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in tqdm(range(1, num_epochs + 1), desc="Training HGAT", unit="epoch"):
        model.train()
        optimizer.zero_grad()

        # HGAT forward: only feature matrix and hypergraph structure are needed
        output = model(fts, H)
        loss = criterion(output, lbls)
        loss.backward()

        # Optional gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        # Compute training accuracy
        with torch.no_grad():
            preds = output.argmax(dim=1)
            train_acc = (preds == lbls).float().mean().item()

        # Print training progress
        if epoch % print_freq == 0:
            tqdm.write(f"Epoch {epoch}/{num_epochs} | "
                       f"Loss: {loss.item():.4f} | Acc: {train_acc:.4f}")

        # Update best weights
        if train_acc > best_acc:
            best_acc = train_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s")
    print(f"Best Train Acc: {best_acc:.4f}")

    # Load the best weights
    model.load_state_dict(best_model_wts)
    return model


def find_hyperneighbors(hyperedges: dict, deleted: list, K: int):
    """
    Find all neighbor nodes that share at least K hyperedges with any deleted node.
    Optimization: first build a reverse index node→edge, then scan related hyperedges for each deleted node.

    Parameters:
      hyperedges: Dict[int, List[int]]  Hyperedge dictionary, where the key is the hyperedge ID and the value is the node list
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

def rebuild_structure_after_node_deletion(hyperedges_orig, deleted_nodes, num_nodes, device):
    """
    Rebuild the sparse hypergraph tensor after node deletion, returning only H_tensor and the new hyperedges.
    """
    # 1) Construct the deletion set
    del_set = set(deleted_nodes.tolist() if isinstance(deleted_nodes, np.ndarray) else deleted_nodes)
    # 2) Filter out retained hyperedges (keep the original ID→node-list mapping for external tracking)
    new_hyperedges = {
        hid: [n for n in nodes if n not in del_set]
        for hid, nodes in hyperedges_orig.items()
        if any(n not in del_set for n in nodes)
    }

    # 3) Enumerate the value lists of new_hyperedges into consecutive row indices
    edges_list = list(new_hyperedges.values())
    rows, cols, vals = [], [], []
    for row_idx, nodes in enumerate(edges_list):
        for n in nodes:
            rows.append(row_idx)   # consecutive row index
            cols.append(n)         # original node index
            vals.append(1.0)

    # 4) Build SciPy COO matrix
    if rows:
        rows_arr = np.array(rows, dtype=int)
        cols_arr = np.array(cols, dtype=int)
        vals_arr = np.array(vals, dtype=float)
        H_sp = sp.coo_matrix((vals_arr, (rows_arr, cols_arr)),
                              shape=(len(edges_list), num_nodes))
    else:
        H_sp = sp.coo_matrix(([], ([], [])), shape=(len(edges_list), num_nodes))

    # 5) Convert to torch.sparse_coo_tensor
    idx = torch.LongTensor(np.vstack((H_sp.row, H_sp.col))).to(device)
    val = torch.FloatTensor(H_sp.data).to(device)
    H_t = torch.sparse_coo_tensor(idx, val, size=H_sp.shape).coalesce().to(device)

    return H_t, new_hyperedges

# def get_grad_hgat(model, data, unlearn_info):
#     """
#     Compute gradients:
#       g_all  - gradient on all training nodes
#       g_del  - gradient difference before/after deletion on deleted nodes
#       g_nei  - gradient difference before/after deletion on neighbor nodes
#
#     unlearn_info: (deleted_nodes:list, hyperedges:dict, K:int)
#     data: {"x", "y", "H_orig", "train_mask"(optional)}
#     Return: g_all, g_del_diff, g_nei_diff
#     """
#     deleted, hyperedges, K = unlearn_info
#     x = data["x"]
#     y = data["y"]
#     H = data["H_orig"]
#     device = x.device
#
#     train_mask = data.get("train_mask", torch.ones_like(y, dtype=torch.bool, device=device))
#     deleted_list = deleted.tolist() if isinstance(deleted, np.ndarray) else deleted
#     nei_list = find_hyperneighbors(hyperedges, deleted_list, K)
#     del_idx = torch.tensor(deleted_list, dtype=torch.long, device=device)
#     nei_idx = torch.tensor(nei_list, dtype=torch.long, device=device)
#
#     mask_del = torch.zeros_like(y, dtype=torch.bool, device=device)
#     mask_del[del_idx] = True
#     mask_nei = torch.zeros_like(y, dtype=torch.bool, device=device)
#     mask_nei[nei_idx] = True
#
#     params = [p for p in model.parameters() if p.requires_grad]
#
#     # Before deletion
#     out1 = model(x, H)
#     def loss_fn(logits, labels): return F.cross_entropy(logits, labels, reduction='sum') if logits.numel()>0 else torch.tensor(0., device=device)
#     loss_all  = loss_fn(out1[train_mask], y[train_mask])
#     loss_del1 = loss_fn(out1[mask_del],    y[mask_del])
#     loss_nei1 = loss_fn(out1[mask_nei],    y[mask_nei])
#
#     g_all  = autograd.grad(loss_all,  params, create_graph=True, retain_graph=True)
#     g_del1 = autograd.grad(loss_del1, params, retain_graph=True)
#     g_nei1 = autograd.grad(loss_nei1, params, retain_graph=True)
#
#     # After deletion
#     x2 = x.clone()
#     x2[del_idx] = 0.0
#     H2, new_hyperedges = rebuild_structure_after_node_deletion(hyperedges, deleted_list, x.size(0), device)
#     out2 = model(x2, H2)
#     loss_del2 = loss_fn(out2[mask_del], y[mask_del])
#     loss_nei2 = loss_fn(out2[mask_nei], y[mask_nei])
#
#     g_del2 = autograd.grad(loss_del2, params, retain_graph=True)
#     g_nei2 = autograd.grad(loss_nei2, params, retain_graph=True)
#
#     # Difference
#     g_del_diff = [d1 - d2 for d1, d2 in zip(g_del1, g_del2)]
#     g_nei_diff = [n1 - n2 for n1, n2 in zip(g_nei1, g_nei2)]
#
#     return g_all, g_del_diff, g_nei_diff
def get_grad_hgat(model, data, unlearn_info):
    """
    Keep consistent with the HGNNP / HGCN version:
    - use create_graph=True, retain_graph=True for all subset gradients
    - return g_all, g_del_diff, g_nei_diff
    Also print the gradient norms to help diagnose NaN / overly large / overly small values.
    """
    deleted_nodes, hyperedges, K = unlearn_info
    x      = data["x"]
    y      = data["y"]
    H      = data["H_orig"]
    device = x.device

    # Find neighbors
    deleted_list = deleted_nodes.tolist() if isinstance(deleted_nodes, torch.Tensor) else deleted_nodes
    neighbors = find_hyperneighbors(hyperedges, deleted_list, K)
    print(f"[GIF] Found {len(neighbors)} neighbor nodes (K={K})")

    # Build masks
    mask_all = data.get("train_mask", torch.ones_like(y, dtype=torch.bool, device=device))
    mask_del = torch.zeros_like(mask_all); mask_del[deleted_list] = True
    mask_nei = torch.zeros_like(mask_all); mask_nei[neighbors]    = True

    # Prepare parameter list and names
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

    # Take differences
    g_del_diff = [d1 - d2 for d1, d2 in zip(g_del1, g_del2)]
    g_nei_diff = [n1 - n2 for n1, n2 in zip(g_nei1, g_nei2)]

    return g_all, g_del_diff, g_nei_diff

def hvps(grad_all, model, vs):
    """
    Hessian–vector product: H(grad_all)·vs
      grad_all: list of ∇_θ L_all
      vs:       list of v vectors (same shapes)
    """
    # Build scalar inner = ∑_i grad_all[i] · vs[i]
    inner = sum((g * v).sum() for g, v in zip(grad_all, vs))
    # Keep params consistent with get_grad_hgat
    params = [p for p in model.parameters() if p.requires_grad]
    # Take gradient of inner to obtain H·v
    # return autograd.grad(
    #     inner, params,
    #     create_graph=True,
    #     retain_graph=True
    # )
    return autograd.grad(
        inner, params,
        create_graph=False,
        retain_graph=True
    )



def approx_gif(model, data, unlearn_info,
               iteration=20, damp=1e-2, scale=1e6):
    """
    LiSSA-based GIF update (HGNNP/HGCN style)
    """
    t0 = time.time()

    # 1) First obtain all first-order gradients
    g_all, g_del, g_nei = get_grad_hgat(model, data, unlearn_info)

    # v = g_del + g_nei
    v      = [gd + gn for gd, gn in zip(g_del, g_nei)]
    # Initialize h as a copy of v
    h      = [vi.clone()       for vi in v]
    params = [p for p in model.parameters() if p.requires_grad]

    # 2) LiSSA iterations
    for _ in range(iteration):
        hv = hvps(g_all, model, h)   # Compute H·h

        for i in range(len(h)):
            # h_{t+1} = v + (1–damp)·h_t – hv/scale
            h[i] = v[i] + (1 - damp) * h[i] - hv[i] / scale

    # 3) Apply the approximated H^{-1}·v to the model parameters
    with torch.no_grad():
        for p, hi in zip(params, h):
            p.sub_(hi / scale)

    elapsed = time.time() - t0
    return elapsed, h