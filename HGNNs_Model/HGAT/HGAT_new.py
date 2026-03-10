# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.checkpoint import checkpoint
#
# # -------- 两个不用 torch_scatter 的辅助函数 --------------------
# def segment_sum(src: torch.Tensor, index: torch.Tensor, num_seg: int):
#     if src.dim() == 1:
#         out = torch.zeros(num_seg, dtype=src.dtype, device=src.device)
#     else:
#         out = torch.zeros(num_seg, src.size(-1), dtype=src.dtype, device=src.device)
#     out.index_add_(0, index, src)
#     return out
#
# def segment_softmax(src: torch.Tensor, index: torch.Tensor, num_seg: int):
#     src_exp = torch.exp(src)
#     denom   = segment_sum(src_exp, index, num_seg)
#     return src_exp / (denom[index] + 1e-15)
#
# class HyperGraphAttentionLayerSparse(nn.Module):
#     """
#     接口与原来完全一致：forward(x, H) -> out
#     只在注意力+聚合这一步使用 checkpoint，以减小显存峰值。
#     """
#     def __init__(self,
#                  in_features,
#                  out_features,
#                  dropout,
#                  alpha,
#                  transfer,
#                  concat=True,
#                  bias=True):
#         super().__init__()
#         self.in_features  = in_features
#         self.out_features = out_features
#         self.dropout      = nn.Dropout(dropout)
#         self.leakyrelu    = nn.LeakyReLU(alpha)
#         self.transfer     = transfer
#         self.concat       = concat
#
#         # 线性映射参数（X @ W）
#         self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
#         # 注意力参数
#         self.a = nn.Parameter(torch.Tensor(2 * out_features, 1))
#         if bias:
#             self.bias = nn.Parameter(torch.Tensor(out_features))
#         else:
#             self.register_parameter('bias', None)
#
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         nn.init.xavier_uniform_(self.weight)
#         nn.init.xavier_uniform_(self.a)
#         if self.bias is not None:
#             nn.init.zeros_(self.bias)
#
#     def forward(self, x: torch.Tensor, H: torch.sparse.FloatTensor) -> torch.Tensor:
#         device = x.device
#         if H.device != device:
#             H = H.to(device)
#         # 自动修正 H 方向
#         N = x.size(0)
#         H_inc = H if H.size(0) == N else H.transpose(0, 1)
#
#         # 1) 计算度向量及其倒数
#         dv = torch.sparse.sum(H_inc, dim=1).to_dense()
#         de = torch.sparse.sum(H_inc, dim=0).to_dense()
#         dv_inv = dv.pow(-0.5)
#         de_inv = de.pow(-1.0)
#
#         # 2) 先线性映射
#         X_proj = x.matmul(self.weight)
#
#         # 3) 隐式超图聚合
#         x_norm = X_proj * dv_inv.unsqueeze(1)
#         E      = torch.spmm(H_inc.transpose(0,1), x_norm)
#         E2     = E * de_inv.unsqueeze(1)
#         F_out  = torch.spmm(H_inc, E2)
#         Y      = F_out * dv_inv.unsqueeze(1)
#
#         # 4) 显式自环
#         Y_hat = Y + X_proj
#
#         # 5) 注意力+显式聚合—重内存部分，使用 checkpoint
#         def attn_agg(Yh, Hm, Xp):
#             pair = Hm._indices()
#             h_v  = Yh[pair[0]]
#             h_u  = Yh[pair[1]]
#             cat  = torch.cat([h_v, h_u], dim=1)
#             e    = self.leakyrelu(cat @ self.a).squeeze(-1)
#             attn = torch.sparse_coo_tensor(pair, e, torch.Size([N, Hm.size(1)]))
#             attn_dense = self.dropout(F.softmax(attn.to_dense(), dim=1))
#             edge_feats = attn_dense.transpose(0,1) @ Xp
#             out = torch.spmm(Hm, edge_feats)
#             if self.bias is not None:
#                 out = out + self.bias
#             return out
#
#         out = checkpoint(attn_agg, Y_hat, H_inc, X_proj, use_reentrant=False)
#         return out
#
# class HGAT_JK(nn.Module):
#     """
#     与 model_hgat.py 中 HGAT_JK 完全一致
#     """
#     def __init__(self,
#                  in_dim: int,
#                  hidden_dim: int,
#                  out_dim: int,
#                  dropout: float,
#                  alpha: float,
#                  num_layers: int = 2,
#                  use_jk: bool = True):
#         super().__init__()
#         self.num_layers = num_layers
#         self.use_jk     = use_jk
#         self.dropout    = nn.Dropout(dropout)
#
#         self.layers = nn.ModuleList()
#         self.res    = nn.ModuleList()
#         self.norm   = nn.ModuleList()
#         self.alphas = nn.ParameterList()
#         for i in range(num_layers):
#             in_ft  = in_dim  if i == 0 else hidden_dim
#             out_ft = out_dim if i == num_layers-1 else hidden_dim
#             self.layers.append(
#                 HyperGraphAttentionLayerSparse(
#                     in_ft, out_ft, dropout, alpha,
#                     transfer=True, concat=True, bias=True
#                 )
#             )
#             self.res.append(nn.Linear(in_ft, out_ft, bias=False))
#             self.norm.append(nn.LayerNorm(out_ft))
#             self.alphas.append(nn.Parameter(torch.tensor(alpha)))
#
#         if use_jk:
#             jk_dim = (num_layers-1)*hidden_dim + out_dim
#             self.classifier = nn.Sequential(
#                 nn.Linear(jk_dim, hidden_dim),
#                 nn.ReLU(),
#                 nn.Dropout(dropout),
#                 nn.Linear(hidden_dim, out_dim),
#             )
#         else:
#             self.classifier = None
#
#     def forward(self, X, H):
#         x = X
#         outputs = []
#         for i in range(self.num_layers):
#             x_pre = self.layers[i](x, H)
#             x_pre = self.dropout(x_pre)
#             x_norm = self.norm[i](x_pre)
#             x = self.alphas[i] * x_norm + (1 - self.alphas[i]) * self.res[i](x)
#             outputs.append(x)
#
#         if self.use_jk:
#             jk  = torch.cat(outputs, dim=1)
#             out = self.classifier(jk)
#         else:
#             out = outputs[-1]
#
#         return out
# -*- coding: utf-8 -*-
"""
layers_hgat_single.py
---------------------
单图版 Hypergraph Attention Layer：
  ① Node ➜ Edge 注意力  ② Edge ➜ Node 注意力
输入:  X (N, Fin)
       H (E, N)  稀疏 / 稠密 0-1 矩阵
输出:  Z (N, Fout)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def segment_sum(src: torch.Tensor, index: torch.Tensor, num_seg: int):
    """
    对相同 index 的元素做求和，等价于 scatter_add。
    src   : (M, F) / (M,)      index: (M,)
    """
    if src.dim() == 1:
        out = torch.zeros(num_seg, dtype=src.dtype, device=src.device)
    else:
        out = torch.zeros(num_seg, src.size(-1),
                          dtype=src.dtype, device=src.device)
    out.index_add_(0, index, src)
    return out


def segment_softmax(src: torch.Tensor, index: torch.Tensor, num_seg: int):
    """
    group-wise softmax；无需先求 max，直接 e^x 再归一化即可。
    """
    src_exp = torch.exp(src)
    denom   = segment_sum(src_exp, index, num_seg)     # (num_seg,)
    return src_exp / (denom[index] + 1e-15)


class HyperGraphAttentionLayerSparse(nn.Module):
    """
    接口与原来完全一致：forward(x, H) -> out
    仅在度归一化时加了 eps，避免除零引发 inf/nan。
    """
    def __init__(self,
                 in_features,
                 out_features,
                 dropout,
                 alpha,
                 transfer,
                 concat=True,
                 bias=True):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.dropout      = nn.Dropout(dropout)
        self.leakyrelu    = nn.LeakyReLU(alpha)
        self.transfer     = transfer
        self.concat       = concat

        # 线性映射参数（X @ W）
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        # 注意力参数
        self.a = nn.Parameter(torch.Tensor(2 * out_features, 1))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.a)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, H: torch.sparse.FloatTensor) -> torch.Tensor:
        # —— 设备对齐 —— #
        device = x.device
        if H.device != device:
            H = H.to(device)
        if self.weight.device != device:
            self.weight.data = self.weight.data.to(device)
        if self.a.device != device:
            self.a.data = self.a.data.to(device)
        if self.bias is not None and self.bias.device != device:
            self.bias.data = self.bias.data.to(device)

        N = x.size(0)
        # —— 自动修正 H 方向 —— #
        H_inc = H if H.size(0) == N else H.transpose(0, 1)

        # —— 1) 计算度向量 —— #
        #    原版可能是 dv_inv = dv.pow(-0.5)，de_inv = de.pow(-1.0)
        #    这里加 eps 避免度为 0
        eps = 1e-10
        dv = torch.sparse.sum(H_inc, dim=1).to_dense()  # [N]
        de = torch.sparse.sum(H_inc, dim=0).to_dense()  # [E]
        dv_inv = (dv + eps).pow(-0.5)                   # [N]
        de_inv = (de + eps).pow(-1.0)                   # [E]

        # —— 2) 先线性映射 —— #
        X_proj = x.matmul(self.weight)                  # [N, out_features]

        # —— 3) 隐式超图聚合 —— #
        x_norm = X_proj * dv_inv.unsqueeze(1)           # Dv^-1/2·X_proj
        E      = torch.spmm(H_inc.transpose(0, 1), x_norm)  # [E, out]
        E2     = E * de_inv.unsqueeze(1)                # De^-1·(…)
        F_out  = torch.spmm(H_inc, E2)                  # [N, out]
        Y      = F_out * dv_inv.unsqueeze(1)            # Dv^-1/2·(…)

        # —— 4) 显式自环 —— #
        Y_hat = Y + X_proj                               # [N, out]

        # —— 5) 原 attention 逻辑 —— #
        pair = H_inc._indices()                         # [2, nnz]
        h_v  = Y_hat[pair[0]]                           # hyperedge 端
        h_u  = Y_hat[pair[1]]                           # node 端
        cat  = torch.cat([h_v, h_u], dim=1)             # [nnz, 2*out]
        e    = self.leakyrelu(cat @ self.a).squeeze(-1) # [nnz]

        attn = torch.sparse_coo_tensor(
            pair, e, torch.Size([N, H_inc.size(1)])
        )
        attn = self.dropout(F.softmax(attn.to_dense(), dim=1))  # [N, E]

        # —— 正确的两步聚合 —— #
        edge_feats = attn.transpose(0, 1) @ X_proj  # [E, F]
        H_inc2 = H if H.size(0) == x.size(0) else H.transpose(0, 1)
        out = torch.spmm(H_inc2, edge_feats)       # [N, F]

        if self.bias is not None:
            out = out + self.bias
        return out
class HGAT_JK(nn.Module):
    """
    与 model_hgat.py 中 HGAT_JK 完全一致
    """
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

        self.layers = nn.ModuleList()
        self.res    = nn.ModuleList()
        self.norm   = nn.ModuleList()
        self.alphas = nn.ParameterList()
        for i in range(num_layers):
            in_ft  = in_dim  if i == 0 else hidden_dim
            out_ft = out_dim if i == num_layers-1 else hidden_dim
            self.layers.append(
                HyperGraphAttentionLayerSparse(
                    in_ft, out_ft, dropout, alpha,
                    transfer=True, concat=True, bias=True
                )
            )
            self.res.append(nn.Linear(in_ft, out_ft, bias=False))
            self.norm.append(nn.LayerNorm(out_ft))
            self.alphas.append(nn.Parameter(torch.tensor(alpha)))

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
            x_pre = self.layers[i](x, H)
            x_pre = self.dropout(x_pre)
            x_norm = self.norm[i](x_pre)
            x = self.alphas[i] * x_norm + (1 - self.alphas[i]) * self.res[i](x)
            outputs.append(x)

        if self.use_jk:
            jk  = torch.cat(outputs, dim=1)
            out = self.classifier(jk)
        else:
            out = outputs[-1]

        return out