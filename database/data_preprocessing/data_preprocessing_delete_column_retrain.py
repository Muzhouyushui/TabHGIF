"""
data_preprocessing.py

该模块主要用于从 CSV 文件中加载数据，并对数据进行预处理：
  - 对离散属性采用 One-Hot 编码；
  - 对连续属性采用标准化处理；
  - 分离标签和特征；
  - 构造超边所需的字典。

CSV 文件要求包含 15 列，顺序如下：
 1. age              (连续型)
 2. workclass        (类别型)
 3. fnlwgt           (连续型)
 4. education        (类别型)
 5. education-num    (连续型)
 6. marital-status   (类别型)
 7. occupation       (类别型)
 8. relationship     (类别型)
 9. race             (类别型)
10. sex              (类别型)
11. capital-gain     (连续型)
12. capital-loss     (连续型)
13. hours-per-week   (连续型)
14. native-country   (类别型)
15. income           (标签)

参数 is_test 为 True 时，默认跳过 CSV 文件的第一行（标题或注释）。
"""
import re
import time
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

import numpy as np
import torch
from scipy.sparse import coo_matrix
from HGCN.HyperGCN import laplacian

def build_incidence_matrix(hyperedges, num_nodes):
    """
    构建超图的邻接矩阵。

    参数：
    hyperedges (dict): 超边字典，其中键是 (属性, 值) 或 (属性, 值, 子簇编号)，
                       值是与该超边相关联的节点索引。
    num_nodes (int): 节点数量

    返回：
    H_sparse (scipy.sparse.coo_matrix): 超图的邻接矩阵
    """
    row_indices = []
    col_indices = []
    data = []

    # 构建超图的邻接矩阵
    for col_val, nodes in hyperedges.items():
        for node in nodes:
            row_indices.append(node)
            col_indices.append(list(hyperedges.keys()).index(col_val))  # 超边的索引
            data.append(1)

    # 使用 coo_matrix 创建稀疏矩阵
    H_sparse = coo_matrix((data, (row_indices, col_indices)), shape=(num_nodes, len(hyperedges)))

    return H_sparse


def preprocess_node_features(
    csv_path: str,
    is_test: bool = False,
    transformer=None,
    ignore_cols: list[str] = None
):
    """
    加载 CSV 并进行预处理；支持忽略 ignore_cols 中的列。
    如果 is_test=True，则跳过 CSV 文件第一行。
    如果传入 transformer，则直接 transform，否则 fit_transform。

    返回：
      X           (np.ndarray): 预处理后的特征矩阵。
      y           (list[int]): 标签列表（"<=50K"→0, ">50K"→1）。
      df           (pd.DataFrame): 原始读入的 DataFrame（未丢弃 ignore_cols）。
      transformer (ColumnTransformer): 用于转换的 transformer。
    """
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    # 1) 读 CSV
    col_names = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race",
        "sex", "capital-gain", "capital-loss", "hours-per-week",
        "native-country", "income"
    ]
    if is_test:
        df = pd.read_csv(csv_path, header=None, names=col_names,
                         skipinitialspace=True, skiprows=1)
    else:
        df = pd.read_csv(csv_path, header=None, names=col_names,
                         skipinitialspace=True)

    # 2) 填充类别缺失
    all_categorical = [
        "workclass", "education", "marital-status",
        "occupation", "relationship", "race", "sex", "native-country"
    ]
    df[all_categorical] = df[all_categorical].fillna("?")

    # 3) 分离标签
    labels = df["income"].values
    label_mapping = {"<=50K": 0, ">50K": 1}
    y = [label_mapping.get(val.strip().rstrip("."), 0) for val in labels]

    # 4) 丢弃 "income" 和 ignore_cols
    ignore_set = set(ignore_cols or [])
    feature_df = df.drop(columns=["income"] + list(ignore_set), errors="ignore")

    # 5) 动态选择离散/连续列（只保留实际在 feature_df 中的）
    default_continuous = ["age", "fnlwgt", "education-num",
                          "capital-gain", "capital-loss", "hours-per-week"]
    categorical_cols = [c for c in all_categorical if c in feature_df.columns]
    continuous_cols  = [c for c in default_continuous if c in feature_df.columns]

    # 6) 构造 ColumnTransformer
    if transformer is None:
        transformer = ColumnTransformer(transformers=[
            ("onehot",   OneHotEncoder(sparse_output=False, handle_unknown="ignore"), categorical_cols),
            ("standard", StandardScaler(),                                       continuous_cols)
        ], remainder="drop")
        X = transformer.fit_transform(feature_df)
    else:
        X = transformer.transform(feature_df)

    # 7) 打印信息
    num_pos = sum(y)
    num_neg = len(y) - num_pos
    print(f"标签分布 → <=50K: {num_neg}，>50K: {num_pos}")
    print(f"忽略列：{ignore_set}")
    # 打印特征维度
    print(f"Generated feature matrix shape: {X.shape}")
    return X, y, df, transformer


def cluster_nodes_by_similarity_gpu(indices, df, max_nodes, device, ignore_cols=None):
    """
    利用 GPU 上的向量化操作对给定节点（来自 df）的索引进行贪心聚类，
    每个簇最多包含 max_nodes 个节点。相似度定义为各列值（除 income 外）相同的数量。

    参数：
      indices     (list): 原始 df 中的节点索引列表。
      df          (DataFrame): 原始数据。
      max_nodes   (int): 单个簇最大节点数。
      device      (torch.device): 使用的设备（GPU）。
      ignore_cols (list or None): 调用方希望忽略的列名列表。

    返回：
      clusters (list of lists): 每个簇为一个列表，列表中元素为原始 df 的行索引。
    """
    ignore_set = set(ignore_cols or [])

    # 先去掉 income 列和要忽略的列
    sub_df = df.loc[indices].drop(columns=["income"] + list(ignore_set), errors="ignore")

    # 将每列 factorize，然后拼成 (n_samples, n_features) 的 tensor
    encoded = []
    for col in sub_df.columns:
        codes, _ = pd.factorize(sub_df[col])
        encoded.append(torch.tensor(codes, dtype=torch.long, device=device))
    X = torch.stack(encoded, dim=1)

    unassigned = torch.ones(X.size(0), dtype=torch.bool, device=device)
    orig_idx   = torch.tensor(indices, dtype=torch.long)
    clusters   = []

    while unassigned.any():
        rep = torch.nonzero(unassigned, as_tuple=False)[0].item()
        current = [orig_idx[rep].item()]
        unassigned[rep] = False

        rem = torch.nonzero(unassigned, as_tuple=False).squeeze(1)
        if rem.numel() > 0:
            rep_vec = X[rep].unsqueeze(0)          # (1, d)
            cand    = X[rem]                       # (r, d)
            sim     = (cand == rep_vec).sum(dim=1) # (r,)
            _, order = torch.sort(sim, descending=True)

            take = min(max_nodes - 1, order.numel())
            sel  = rem[order[:take]]
            for p in sel.tolist():
                current.append(orig_idx[p].item())
                unassigned[p] = False

        clusters.append(current)

    return clusters


import time
import numpy as np
import torch
from typing import List, Dict, Any



def generate_hyperedge_dict(df, categorical_cols, continuous_cols,
                            ignore_cols=None, max_nodes_per_hyperedge=None,
                            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    生成超边字典，支持跳过 ignore_cols 指定的列。
    如果某个超边的节点数超过 max_nodes_per_hyperedge，
    会调用 cluster_nodes_by_similarity_gpu 做聚类拆分。

    返回：
      hyperedges (dict): 键为 (col, val[, cluster_id])，值为节点索引列表。
    """
    categorical_cols = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country"
    ]
    continuous_cols = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]

    ignore_set = set(ignore_cols or [])
    hyperedges = {}
    col_edge_count = {}
    total_cluster_time = 0.0

    # 1) 离散列
    for col in categorical_cols:
        if col in ignore_set:
            continue
        start = len(hyperedges)
        # 在构造超边前，把 ignore_cols 从 df 中删除
        df_proc = df.drop(columns=list(ignore_set), errors="ignore")
        for val in df_proc[col].dropna().unique():
            idxs = df_proc.index[df_proc[col] == val].tolist()
            if len(idxs) <= max_nodes_per_hyperedge:
                hyperedges[(col, val)] = idxs
            else:
                t0 = time.time()
                clusters = cluster_nodes_by_similarity_gpu(
                    idxs, df_proc, max_nodes_per_hyperedge, device, ignore_cols
                )
                total_cluster_time += time.time() - t0
                for cid, cidx in enumerate(clusters):
                    hyperedges[(col, val, cid)] = cidx
        col_edge_count[col] = len(hyperedges) - start

    # 2) 连续列
    for col in continuous_cols:
        if col in ignore_set:
            continue
        start = len(hyperedges)
        vals = df[col].values.astype(float)

        # 不同列的分箱策略
        if col == "age":
            mn, mx = vals.min(), vals.max()
            bins = [mn + (mx - mn) * i / 10 for i in range(11)]
        elif col in ("capital-gain", "capital-loss"):
            nonz = vals[vals != 0]
            q = np.percentile(nonz, [0, 25, 50, 75, 100])
            bins = [0] + list(q)
        else:
            bins = np.percentile(vals[~np.isnan(vals)], [0, 25, 50, 75, 100]).tolist()

        df_proc = df.drop(columns=list(ignore_set), errors="ignore")
        for i in range(len(bins) - 1):
            lo, hi = bins[i], bins[i+1]
            idxs = df_proc.index[(df_proc[col] >= lo) & (df_proc[col] < hi)].tolist()
            if len(idxs) <= max_nodes_per_hyperedge:
                hyperedges[(col, f"{lo:.2f}-{hi:.2f}")] = idxs
            else:
                t0 = time.time()
                clusters = cluster_nodes_by_similarity_gpu(
                    idxs, df_proc, max_nodes_per_hyperedge, device, ignore_cols
                )
                total_cluster_time += time.time() - t0
                for cid, cidx in enumerate(clusters):
                    hyperedges[(col, f"{lo:.2f}-{hi:.2f}", cid)] = cidx
        col_edge_count[col] = len(hyperedges) - start

    # 打印统计信息
    print(f"Subgraph clustering total time: {total_cluster_time:.4f}s")
    # print("=== 每列生成的超边数量 ===")
    # for col, cnt in col_edge_count.items():
    #     print(f"  {col}: {cnt}")
    print("超边总数量：",len(hyperedges))
    return hyperedges


