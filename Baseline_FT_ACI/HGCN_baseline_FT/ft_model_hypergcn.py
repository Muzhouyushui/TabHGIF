# ft_model_hypergcn.py
import math
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

def sym_normalise(mat: sp.csr_matrix) -> sp.csr_matrix:
    d = np.array(mat.sum(1)).flatten()
    d_inv_sqrt = np.power(d, -0.5, where=d > 0)
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    return D_inv_sqrt @ mat @ D_inv_sqrt

def ssm_to_torch(mat: sp.csr_matrix) -> torch.sparse.FloatTensor:
    mat = mat.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((mat.row, mat.col))).long()
    values = torch.from_numpy(mat.data)
    shape = torch.Size(mat.shape)
    return torch.sparse_coo_tensor(indices, values, shape)

class SparseMM(Function):
    @staticmethod
    def forward(ctx, sparse, dense):
        ctx.save_for_backward(sparse, dense)
        return torch.mm(sparse, dense)

    @staticmethod
    def backward(ctx, grad_output):
        sparse, dense = ctx.saved_tensors
        grad_sparse = grad_dense = None
        if ctx.needs_input_grad[0]:
            grad_sparse = torch.mm(grad_output, dense.t())
        if ctx.needs_input_grad[1]:
            grad_dense = torch.mm(sparse.t(), grad_output)
        return grad_sparse, grad_dense

def laplacian(edge_list, X: np.ndarray, mediators: bool = True):
    """
    HyperGCN Laplacian approximation (same spirit as你原文件里那份)：
    对每条超边用随机投影找 supremum/infimum 节点，并加 mediators 连接构图。
    """
    rows, cols, weights = [], [], {}
    rv = np.random.rand(X.shape[1])

    for e in edge_list:
        if len(e) < 2:
            continue
        proj = X[e] @ rv
        Se = e[int(np.argmax(proj))]
        Ie = e[int(np.argmin(proj))]
        c = 2 * len(e) - 3 if mediators else len(e)
        w = 1.0 / c

        def _add(u, v):
            rows.append(u); cols.append(v)
            weights[(u, v)] = weights.get((u, v), 0) + w

        _add(Se, Ie); _add(Ie, Se)
        if mediators:
            for m in e:
                if m in (Se, Ie):
                    continue
                _add(Se, m); _add(m, Se)
                _add(Ie, m); _add(m, Ie)

    n = X.shape[0]
    if weights:
        us, vs = zip(*weights.keys())
        data = list(weights.values())
    else:
        us = vs = data = []

    A = sp.coo_matrix((data, (us, vs)), shape=(n, n), dtype=np.float32)
    A = A.tocsr() + sp.eye(n, dtype=np.float32)
    A = sym_normalise(A)
    return ssm_to_torch(A)

class HyperGraphConvolution(nn.Module):
    def __init__(self, in_feats, out_feats, reapproximate=False, use_cuda=False):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_feats, out_feats))
        self.bias = nn.Parameter(torch.FloatTensor(out_feats))
        self.reapproximate = reapproximate
        self.use_cuda = use_cuda
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, structure, X, mediators: bool):
        HW = X @ self.weight
        if self.reapproximate:
            A = laplacian(structure, HW.detach().cpu().numpy(), mediators)
        else:
            A = structure

        if self.use_cuda:
            A = A.cuda()
        AX = SparseMM.apply(A, HW)
        return AX + self.bias

class HyperGCN(nn.Module):
    def __init__(self, num_nodes, edge_list, X_init, args):
        super().__init__()
        self.dropout = args.dropout
        self.mediators = args.mediators
        cuda_ok = args.cuda and torch.cuda.is_available()

        offset = 2 if getattr(args, "dataset", "") != "citeseer" else 4
        dims = [args.d] + [
            2 ** (args.depth - i + offset)
            for i in range(1, args.depth)
        ] + [args.c]

        if args.fast:
            reapp = False
            structure = laplacian(edge_list, X_init, args.mediators)
        else:
            reapp = True
            structure = edge_list

        self.structure = structure
        self.layers = nn.ModuleList([
            HyperGraphConvolution(
                in_feats=dims[i],
                out_feats=dims[i+1],
                reapproximate=reapp,
                use_cuda=cuda_ok
            )
            for i in range(len(dims) - 1)
        ])

    def forward(self, X):
        H = X
        for i, layer in enumerate(self.layers):
            H = layer(self.structure, H, self.mediators)
            if i != len(self.layers) - 1:
                H = F.relu(H)
                H = F.dropout(H, p=self.dropout, training=self.training)
        return F.log_softmax(H, dim=1)
