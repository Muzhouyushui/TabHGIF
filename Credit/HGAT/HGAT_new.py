"""
layers_hgat_single.py
---------------------
Single-graph Hypergraph Attention Layer:
  ① Node ➜ Edge attention  ② Edge ➜ Node attention
Input:  X (N, Fin)
        H (E, N)  sparse / dense 0-1 matrix
Output: Z (N, Fout)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def segment_sum(src: torch.Tensor, index: torch.Tensor, num_seg: int):
    """
    Sum elements with the same index, equivalent to scatter_add.
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
    Group-wise softmax; no need to compute the max first,
    directly normalize e^x.
    """
    src_exp = torch.exp(src)
    denom   = segment_sum(src_exp, index, num_seg)     # (num_seg,)
    return src_exp / (denom[index] + 1e-15)


class HyperGraphAttentionLayerSparse(nn.Module):
    """
    Interface remains exactly the same as before: forward(x, H) -> out
    Only adds eps during degree normalization to avoid inf/nan caused by division by zero.
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

        # Linear mapping parameters (X @ W)
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        # Attention parameters
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
        # —— Device alignment —— #
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
        # —— Automatically fix H orientation —— #
        H_inc = H if H.size(0) == N else H.transpose(0, 1)

        # —— 1) Compute degree vectors —— #
        #    In the original version, this may be dv_inv = dv.pow(-0.5), de_inv = de.pow(-1.0)
        #    Here eps is added to avoid zero degree
        eps = 1e-10
        dv = torch.sparse.sum(H_inc, dim=1).to_dense()  # [N]
        de = torch.sparse.sum(H_inc, dim=0).to_dense()  # [E]
        dv_inv = (dv + eps).pow(-0.5)                   # [N]
        de_inv = (de + eps).pow(-1.0)                   # [E]

        # —— 2) Linear mapping first —— #
        X_proj = x.matmul(self.weight)                  # [N, out_features]

        # —— 3) Implicit hypergraph aggregation —— #
        x_norm = X_proj * dv_inv.unsqueeze(1)           # Dv^-1/2·X_proj
        E      = torch.spmm(H_inc.transpose(0, 1), x_norm)  # [E, out]
        E2     = E * de_inv.unsqueeze(1)                # De^-1·(…)
        F_out  = torch.spmm(H_inc, E2)                  # [N, out]
        Y      = F_out * dv_inv.unsqueeze(1)            # Dv^-1/2·(…)

        # —— 4) Explicit self-loop —— #
        Y_hat = Y + X_proj                               # [N, out]

        # —— 5) Original attention logic —— #
        pair = H_inc._indices()                         # [2, nnz]
        h_v  = Y_hat[pair[0]]                           # hyperedge side
        h_u  = Y_hat[pair[1]]                           # node side
        cat  = torch.cat([h_v, h_u], dim=1)             # [nnz, 2*out]
        e    = self.leakyrelu(cat @ self.a).squeeze(-1) # [nnz]

        attn = torch.sparse_coo_tensor(
            pair, e, torch.Size([N, H_inc.size(1)])
        )
        attn = self.dropout(F.softmax(attn.to_dense(), dim=1))  # [N, E]

        # —— Correct two-step aggregation —— #
        edge_feats = attn.transpose(0, 1) @ X_proj  # [E, F]
        H_inc2 = H if H.size(0) == x.size(0) else H.transpose(0, 1)
        out = torch.spmm(H_inc2, edge_feats)       # [N, F]

        if self.bias is not None:
            out = out + self.bias
        return out


class HGAT_JK(nn.Module):
    """
    Exactly the same as HGAT_JK in model_hgat.py
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