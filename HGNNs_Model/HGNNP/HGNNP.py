import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_incidence_matrix(hyperedges: dict, num_nodes: int) -> sp.coo_matrix:
    """
    Construct node-hyperedge incidence matrix H (COO) of shape (N, M).

    Parameters
    ----------
    hyperedges : dict
        Mapping each hyperedge key to list of node indices.
    num_nodes : int
        Total number of nodes N.

    Returns
    -------
    H : scipy.sparse.coo_matrix
    """
    hyperedge_list = list(hyperedges.values())
    num_hyperedges = len(hyperedge_list)
    H = sp.lil_matrix((num_nodes, num_hyperedges), dtype=np.float32)
    for j, indices in enumerate(hyperedge_list):
        for i in indices:
            H[i, j] = 1.0
    return H.tocoo()


def compute_degree_vectors(H: sp.spmatrix) -> (np.ndarray, np.ndarray):
    """
    Compute node degree inverse sqrt and hyperedge degree inverse vectors.

    Parameters
    ----------
    H : scipy.sparse.spmatrix
        Incidence matrix (N x M).

    Returns
    -------
    dv_inv : np.ndarray
        Node degree inverse sqrt, shape (N,).
    de_inv : np.ndarray
        Hyperedge degree inverse, shape (M,).
    """
    dv = np.array(H.sum(axis=1)).flatten()
    de = np.array(H.sum(axis=0)).flatten()
    eps = 1e-10
    dv_inv = 1.0 / np.sqrt(dv + eps)
    de_inv = 1.0 / (de + eps)
    return dv_inv, de_inv


def normalize_H(H: sp.spmatrix) -> torch.sparse.FloatTensor:
    """
    Compute normalized hypergraph propagation matrix:
        H_norm = Dv^{-1/2} H De^{-1} H^T Dv^{-1/2}

    Returns torch sparse tensor of shape (N, N).
    """
    H = sp.coo_matrix(H)
    N, M = H.shape
    dv = np.array(H.sum(axis=1)).flatten()
    de = np.array(H.sum(axis=0)).flatten()
    dv_inv_sqrt = sp.diags(np.power(dv, -0.5, where=(dv>0)), dtype=np.float32)
    de_inv = sp.diags(np.power(de, -1.0, where=(de>0)), dtype=np.float32)
    H_norm = dv_inv_sqrt.dot(H).dot(de_inv).dot(H.T).dot(dv_inv_sqrt)
    H_norm = H_norm.tocoo()
    indices = torch.LongTensor([H_norm.row, H_norm.col])
    values = torch.FloatTensor(H_norm.data)
    return torch.sparse_coo_tensor(indices, values, torch.Size(H_norm.shape))


class HGNNP_conv_implicit(nn.Module):
    """
    HGNN+ convolution layer (implicit style): map then aggregate.

    y = Dv^{-1} H De^{-1} H^T (X W + b)
    """
    def __init__(self, in_ft: int, out_ft: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / (self.weight.size(1) ** 0.5)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, H: torch.sparse.FloatTensor,
                dv_inv: torch.Tensor, de_inv: torch.Tensor) -> torch.Tensor:
         # 1) 先线性映射
        x_mapped = x.matmul(self.weight)

        if self.bias is not None:
            x_mapped = x_mapped + self.bias
      # 2) 聚合（map→aggregate）
        x_norm = x_mapped * dv_inv.unsqueeze(1)
        E = torch.spmm(H.t(), x_norm)
        E2 = E * de_inv.unsqueeze(1)
        F_out = torch.spmm(H, E2)
        y = F_out * dv_inv.unsqueeze(1)
         # 3) 加回 residual（自环）
        out = y + x_mapped

        return out

class HGNNP_implicit(nn.Module):
    """
    Two-layer implicit HGNN+ model for classification.
    """
    def __init__(self, in_ch: int, n_class: int, n_hid: int, dropout: float = 0.5):
        super().__init__()
        self.hgc1 = HGNNP_conv_implicit(in_ch, n_hid)
        self.hgc2 = HGNNP_conv_implicit(n_hid, n_class)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, H: torch.sparse.FloatTensor,
                dv_inv: torch.Tensor, de_inv: torch.Tensor) -> torch.Tensor:
        x = self.hgc1(x, H, dv_inv, de_inv)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.hgc2(x, H, dv_inv, de_inv)
        return F.log_softmax(x, dim=1)
