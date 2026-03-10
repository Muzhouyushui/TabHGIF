# -*- coding: utf-8 -*-
"""
model_hgat.py
=============
一个多层 HGAT 封装（与您原来 HGNN_ATT 的接口保持一致）
"""
import torch
import torch.nn as nn
from HGNNs_Model.HGAT.HGAT_new import HyperGraphAttentionLayerSparse
import numpy as np
import scipy.sparse as sp

# ----------------------------------------------------------------------
def build_incidence_matrix(hyperedges: dict, num_nodes: int, device=None) -> torch.Tensor:
    """
    将超边字典转成稀疏 incidence matrix，形状 [n_edges, n_nodes]
    """
    n_edges = len(hyperedges)
    H = torch.zeros((n_edges, num_nodes), dtype=torch.float32, device=device)
    for i, nodes in enumerate(hyperedges.values()):
        H[i, nodes] = 1.0
    return H.to_sparse()  # 返回稀疏矩阵
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






class HGAT_JK(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hidden_dim: int,
                 out_dim: int,
                 dropout: float,
                 alpha: float,
                 num_layers: int = 2,
                 use_jk: bool = True):
        super().__init__()
        self.num_layers = num_layers
        self.use_jk     = use_jk
        self.dropout    = nn.Dropout(dropout)

        # 动态创建：layers, res projections, norms, alphas
        self.layers = nn.ModuleList()
        self.res    = nn.ModuleList()
        self.norm   = nn.ModuleList()
        self.alphas = nn.ParameterList()
        for i in range(num_layers):
            # 输入/输出维度：中间层用 hidden_dim，首层 in_dim，尾层 out_dim
            in_ft  = in_dim  if i == 0 else hidden_dim
            out_ft = out_dim if i == num_layers-1 else hidden_dim
            # 1) 注意力层
            self.layers.append(
                HyperGraphAttentionLayerSparse(
                    in_ft, out_ft, dropout, alpha,
                    transfer=True, concat=True, bias=True
                )
            )
            # 2) 跨层残差对齐投影
            self.res.append(nn.Linear(in_ft, out_ft, bias=False))
            # 3) LayerNorm
            self.norm.append(nn.LayerNorm(out_ft))
            # 4) 可学残差权重
            self.alphas.append(nn.Parameter(torch.tensor(alpha)))

        # 如果启用 JK，就拼接所有层的输出再分类
        if use_jk:
            jk_dim = (num_layers-1)*hidden_dim + out_dim
            self.classifier = nn.Sequential(
                nn.Linear(jk_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim),
            )
        else:
            self.classifier = None

    def forward(self, X, H):
        x = X
        outputs = []
        for i in range(self.num_layers):
            # 注意力 + Dropout
            x_pre = self.layers[i](x, H)       # [N, out_ft]
            x_pre = self.dropout(x_pre)
            # LayerNorm（在残差前）
            x_norm = self.norm[i](x_pre)
            # 残差融合：α·new + (1−α)·proj(x)
            x = self.alphas[i] * x_norm + (1 - self.alphas[i]) * self.res[i](x)
            outputs.append(x)

        if self.use_jk:
            # 拼接所有层的输出
            jk  = torch.cat(outputs, dim=1)   # [N, jk_dim]
            out = self.classifier(jk)         # [N, out_dim]
        else:
            # 只用最后一层输出
            out = outputs[-1]                 # [N, out_dim]

        return out