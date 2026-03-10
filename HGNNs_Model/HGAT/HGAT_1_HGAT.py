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
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------- 两个不用 torch_scatter 的辅助函数 --------------------
def segment_sum(src: torch.Tensor, index: torch.Tensor, num_seg: int):
    """
    对相同 index 的元素做求和，等价于 scatter_add。
    src   : (M, F) / (M,)      index: (M,)
    """
    if src.dim() == 1:
        out = torch.zeros(num_seg, dtype=src.dtype, device=src.device)
    else:
        out = torch.zeros(num_seg, src.size(-1), dtype=src.dtype,
                          device=src.device)
    out.index_add_(0, index, src)
    return out


def segment_softmax(src: torch.Tensor, index: torch.Tensor, num_seg: int):
    """
    group-wise softmax；无需先求 max，直接 e^x 再归一化即可，
    实践中 x≈[-20,20] 数值足够稳定。如担心可减去全局 max().
    """
    src_exp = torch.exp(src)
    denom   = segment_sum(src_exp, index, num_seg)     # (num_seg,)
    return src_exp / (denom[index] + 1e-15)


# ------------------------- HGAT Layer -------------------------

class HyperGraphAttentionLayerSparse(nn.Module):
    """
    接口与原来完全一致：forward(x, H) -> out
    只修改了聚合顺序：先线性→聚合＋显式自环，再走原 attention 逻辑，
    并自动处理 H 的方向（N×E 或 E×N 都可）。
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
        # —— 设备对齐：把 H 与所有参数都搬到 x 所在的 device —— #
        device = x.device
        if H.device != device:
            H = H.to(device)
        if self.weight.device != device:
            self.weight.data = self.weight.data.to(device)
        if self.a.device != device:
            self.a.data = self.a.data.to(device)
        if self.bias is not None and self.bias.device != device:
            self.bias.data = self.bias.data.to(device)
        # —— 设备对齐：LayerNorm 参数 —— #

        N = x.size(0)
        # —— 0) 自动修正 H 方向 —— #
        H_inc = H if H.size(0) == N else H.transpose(0, 1)

        # —— 1) 计算度向量及其倒数 —— #
        dv = torch.sparse.sum(H_inc, dim=1).to_dense()  # [N]
        de = torch.sparse.sum(H_inc, dim=0).to_dense()  # [E]
        dv_inv = dv.pow(-0.5)                           # [N]
        de_inv = de.pow(-1.0)                           # [E]

        # —— 2) 先线性映射 —— #
        X_proj = x.matmul(self.weight)                  # [N, out_features]

        # —— 3) 隐式超图聚合 —— #
        x_norm = X_proj * dv_inv.unsqueeze(1)           # Dv^-1/2·X_proj
        E      = torch.spmm(H_inc.transpose(0,1), x_norm)  # [E, out]
        E2     = E * de_inv.unsqueeze(1)                # De^-1·(…)
        F_out  = torch.spmm(H_inc, E2)                  # [N, out]
        Y      = F_out * dv_inv.unsqueeze(1)            # Dv^-1/2·(…)

        # —— 4) 显式自环 —— #
        Y_hat = Y + X_proj                               # [N, out]

        # —— 5) 原 attention 逻辑（只改输入） —— #
        pair = H_inc._indices()                         # [2, nnz]
        h_v  = Y_hat[pair[0]]                           # hyperedge 端
        h_u  = Y_hat[pair[1]]                           # node 端
        cat  = torch.cat([h_v, h_u], dim=1)             # [nnz, 2*out]
        e    = self.leakyrelu(cat @ self.a).squeeze(-1) # [nnz]
        attn = torch.sparse_coo_tensor(
            pair, e, torch.Size([N, H_inc.size(1)])
        )
        # … up to attention 计算，得到 attn:[N×E] X_proj:[N×F] …

        attn = self.dropout(F.softmax(attn.to_dense(), dim=1))  # [N, E]

        # —— 正确的两步聚合 —— #
        # 1) 节点 → 超边：每个超边聚合它所有节点的特征
        edge_feats = attn.transpose(0, 1) @ X_proj  # [E, F]
        # 2) 超边 → 节点：每个节点聚合它所连超边的特征
        H_inc = H if H.size(0) == x.size(0) else H.transpose(0, 1)
        out = torch.spmm(H_inc, edge_feats)  # [N, F]

        if self.bias is not None:
            out = out + self.bias
        return out