import time
import copy

from torch.xpu import device
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import grad


def train_model(model, criterion, optimizer, scheduler,
                fts, lbls, num_epochs=200, print_freq=None):
    # Save the best weights and accuracy
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

        # Update the best weights
        if acc > best_acc:
            best_acc = acc
            best_wts = copy.deepcopy(model.state_dict())

    # Load the best weights and print the final best accuracy
    model.load_state_dict(best_wts)
    print(f"Best Train Acc: {best_acc:.4f}")
    return model

def get_grad_hgcn(model, data, A_before, A_after, x_before, x_after, deleted_nodes, deleted_neighbors):
    """
    Compute gradients for GIF using explicit pre-/post-deletion features:
      - g_all: full-graph gradient at pre-deletion
      - g1   : deletion-subset gradient at pre-deletion (A_before + x_before)
      - g2   : deletion-subset gradient at post-deletion (A_after + x_after)
    """

    # —— New: check feature zeroing status —— #
    # Ensure deleted nodes in x_before have at least one nonzero element
    nonzero_before = (x_before[deleted_nodes] != 0).any(dim=1)
    assert nonzero_before.all().item(), (
        "Error: Some deleted_nodes in x_before are already zero."
    )
    # Ensure deleted nodes in x_after are all zero
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
    for l in model.layers:
        l.reapproximate = False
    out1 = model(x_before)
    params = [p for p in model.parameters() if p.requires_grad]
    loss_all = nn.functional.nll_loss(out1[mask_all], y[mask_all], reduction='sum')
    loss_d1  = nn.functional.nll_loss(out1[mask_del], y[mask_del], reduction='sum')
    g_all = grad(loss_all, params, retain_graph=True, create_graph=True)
    g1    = grad(loss_d1,  params, retain_graph=True, create_graph=True)

    # Post-deletion pass
    model.structure = A_after
    for l in model.layers:
        l.reapproximate = False
    out2 = model(x_after)
    loss_d2 = nn.functional.nll_loss(out2[mask_del], y[mask_del], reduction='sum')
    g2 = grad(loss_d2, params, retain_graph=True, create_graph=True)

    # Check whether g2 is zero
    is_g2_zero = torch.all(g2[0].abs() < 1e-6)  # Use a very small threshold to determine whether it is close to zero
    print(f"g2 is zero: {is_g2_zero.item()}")

    return g_all, g1, g2


def hvp(g_all, model, vs):
    elem = sum((g * v).sum() for g, v in zip(g_all, vs))
    return grad(elem, [p for p in model.parameters() if p.requires_grad], create_graph=True)

# ------------------------------------------------------------------
# approx_gif  —— LiSSA approximate inverse Hessian update
# ------------------------------------------------------------------
def approx_gif(model, data, A_before, A_after, deleted_neighbors, x_before, x_after, deleted_nodes, iters=20, damp=1e-2, scale=1e6):
    """
    LiSSA-based approximate GIF update.
    Explicitly takes x_before and x_after features.
    """
    t0 = time.time()
    # Compute gradients
    g_all, g1, g2 = get_grad_hgcn(model, data, A_before, A_after, x_before, x_after, deleted_nodes, deleted_neighbors)
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


def get_neighbors(deleted_nodes, edge_list, K):
    """
    Compute neighbor nodes of deleted nodes, where neighbors are defined
    by sharing at least K hyperedges.

    Optimization: precompute the hyperedges each node belongs to,
    avoiding repeated traversal of the hyperedge list.

    Parameters:
      deleted_nodes (Tensor): Indices of deleted nodes (Tensor), containing multiple deleted nodes.
      edge_list (list of lists): Hyperedge list, where each hyperedge is a list containing multiple nodes.
      K (int): Threshold for defining neighbor nodes, requiring the number of shared hyperedges to be >= K.

    Returns:
      neighbors (dict): Neighbor node sets for each deleted node
                        (dictionary: key = deleted node, value = neighbor node set).
    """
    from scipy.sparse import csr_matrix
    start = time.time()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print("start neighbors getting neighbors")

    # Step 1: build the set of hyperedges each node participates in
    node_to_edges = {node: set() for node in range(len(edge_list))}

    # Move edge_list to GPU
    edge_list_gpu = [torch.tensor(edge, device=device) for edge in edge_list]

    # Use GPU to compute the hyperedge set for each node
    for edge_index, edge in enumerate(edge_list_gpu):
        for node in edge:
            node_to_edges[node.item()].add(edge_index)  # record the hyperedge indices each node participates in

    # Step 2: precompute the number of shared hyperedges for each node pair
    shared_edge_count = {}  # store the number of shared hyperedges for each node pair
    for node_a in node_to_edges:
        for node_b in node_to_edges:
            if node_a == node_b:
                continue
            # Compute the number of shared hyperedges between node_a and node_b
            shared_edges = len(node_to_edges[node_a] & node_to_edges[node_b])
            if shared_edges >= K:
                if node_a not in shared_edge_count:
                    shared_edge_count[node_a] = set()
                shared_edge_count[node_a].add(node_b)

    # Step 3: iterate through each deleted node
    neighbors = {}  # store the neighbor node set for each deleted node

    for deleted_node in deleted_nodes:
        neighbors[deleted_node.item()] = set()  # initialize neighbor node set

        # Step 4: get neighbors that share hyperedges with the deleted node
        if deleted_node.item() in shared_edge_count:
            neighbors[deleted_node.item()] = shared_edge_count[deleted_node.item()]

    return neighbors