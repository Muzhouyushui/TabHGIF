#####################
# ========== B. GIF/IF-related helper functions ==========
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
    After deleting the specified feature columns, remove the corresponding hyperedges
    and rebuild the sparse H.
    Return: (H_t, new_hyperedges)
    """
    orig_count = len(hyperedges_orig)
    # 1) Filter out hyperedges belonging to deleted columns
    # Support tuple keys (col, val) and integer keys
    # (when keys are integers, column-based filtering cannot be applied, so keep all)
    new_hyperedges = {}
    for hid, nodes in hyperedges_orig.items():
        try:
            keep = hid[0] not in deleted_names
        except TypeError:
            keep = True  # Cannot determine the source column for integer keys, so keep them
        if keep:
            new_hyperedges[hid] = nodes
    new_count = len(new_hyperedges)
    removed = orig_count - new_count

    # Build sparse tensor
    rows, cols, vals = [], [], []
    for i, nodes in enumerate(new_hyperedges.values()):
        for n in nodes:
            rows.append(i)
            cols.append(n)
            vals.append(1.0)

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
            torch.empty((2, 0), dtype=torch.int64, device=device),
            torch.tensor([], device=device),
            size=(0, num_nodes)
        ).coalesce()

    # Print deletion result here
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
    1) Zero out all encoded feature dimensions in X_tensor corresponding to column_names
    2) Call rebuild_structure_after_column_deletion to remove hyperedges and rebuild H
    Return: (X_new, new_hyperedges, H_new)
    """
    # 1) Feature name mapping
    try:
        feat_names = transformer.get_feature_names_out().tolist()
    except AttributeError:
        feat_names = transformer.get_feature_names()

    del_idx = []
    for col in column_names:
        idxs = [i for i, f in enumerate(feat_names)
                if col in re.split(r'__|_', f)]
        if idxs:
            print(f"[zero] '{col}' → zero {len(idxs)} dims: {idxs}")
            del_idx += idxs
        else:
            print(f"[warn] '{col}' did not match any encoded features")
    del_idx = sorted(set(del_idx))
    if del_idx:
        X_tensor[:, del_idx] = 0.0
        print(f"[verify] Zeroed out {len(del_idx)} feature dimensions")
    else:
        print("[verify] No features were zeroed out")

    # 2) Call rebuild_structure_after_column_deletion
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
    Compute gradient differences for column deletion, and print the number of zeroed columns
    and deleted hyperedges.
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

    # —— Original gradient g_all —— #
    out1 = model(x, H)
    loss_all = F.cross_entropy(out1[mask_all], y[mask_all], reduction="sum")
    g_all = grad(loss_all, params, create_graph=True, retain_graph=True)

    # —— Post-deletion —— #
    # 1) Zero out the corresponding feature columns in the feature matrix
    x2 = x.clone()
    x2[:, deleted_idxs] = 0.0
    print(f"[ColumnDeletion] Zeroed feature columns: {len(deleted_idxs)} → {deleted_names}")

    # 2) Rebuild hypergraph H2 and print info
    H2, new_hyp = rebuild_structure_after_column_deletion(
        hyperedges, deleted_names, x.size(0), dev
    )

    out2 = model(x2, H2)
    loss_del = F.cross_entropy(out2[mask_all], y[mask_all], reduction="sum")
    g_del = grad(loss_del, params, create_graph=True, retain_graph=True)

    # Difference
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
    GIF update for column deletion:
      1) g_all, g_del_diff = get_grad_hgat_col(...)
      2) Use LiSSA iterations to approximate H^{-1}·g_del_diff
      3) Apply the result to the model parameters
    """
    t0 = time.time()

    g_all, g_del, _ = get_grad_hgat_col(model, data, unlearn_info)
    # Initialize v as the difference
    v = [gd.clone() for gd in g_del]
    h = [vi.clone() for vi in v]
    params = [p for p in model.parameters() if p.requires_grad]

    # LiSSA iterations
    for _ in range(iteration):
        hv = hvps(g_all, model, h)
        for i in range(len(h)):
            h[i] = v[i] + (1 - damp) * h[i] - hv[i] / scale

    # Update model parameters
    with torch.no_grad():
        for p, hi in zip(params, h):
            p.sub_(hi / scale)

    return time.time() - t0, h