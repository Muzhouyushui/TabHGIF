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
from typing import List, Dict, Tuple

import time
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import numpy as np

import torch
from config import get_args
def preprocess_node_features(csv_path, is_test=False, transformer=None):
    """
    加载 CSV 数据并进行预处理。

    如果 is_test=True，则跳过 CSV 文件的第一行（标题/注释）。
    如果传入 transformer，则直接调用 transformer.transform() 对数据转换，否则拟合新的 transformer。

    参数：
      csv_path     (str): CSV 文件路径。
      is_test      (bool): 是否为测试数据，默认为 False。
      transformer  (ColumnTransformer or None): 预处理对象，默认为 None。

    返回：
      X           (np.ndarray): 预处理后的节点特征矩阵。
      y           (list): 标签列表，对应 income 列（"<=50K" 映射为 0, ">50K" 映射为 1）。
      df          (DataFrame): 原始 CSV 数据构成的 DataFrame。
      transformer (ColumnTransformer): 拟合/使用过的预处理 transformer。
    """
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

    # 对类别型变量缺失值填充为 "?"
    categorical_cols = [
        "workclass", "education", "marital-status",
        "occupation", "relationship", "race", "sex", "native-country"
    ]
    df[categorical_cols] = df[categorical_cols].fillna("?")

    # 分离标签和特征
    labels = df["income"].values
    feature_df = df.drop(columns=["income"])

    continuous_cols = ["age", "fnlwgt", "education-num",
                       "capital-gain", "capital-loss", "hours-per-week"]
    discrete_cols = categorical_cols

    # 随机选取 70% 的数据用于训练
    df_train = df.sample(frac=1.0, random_state=42)  # 选择 70% 的数据
    df_remaining = df.drop(df_train.index)  # 剩下的 30% 数据不用于训练

    # 特征与标签分离
    labels_train = df_train["income"].values
    features_train = df_train.drop(columns=["income"])


    if transformer is None:
        transformer = ColumnTransformer(transformers=[
            ("discrete", OneHotEncoder(sparse_output=False), discrete_cols),
            ("continuous", StandardScaler(), continuous_cols)
        ])
        X = transformer.fit_transform(feature_df)
    else:
        X = transformer.transform(feature_df)

    # # 标签映射："<=50K" 映射为 0, ">50K" 映射为 1（去除两边可能的空格）
    # label_mapping = {"<=50K": 0, ">50K": 1}
    # y = [label_mapping.get(val.strip(), 0) for val in labels]
    # 在这之后，统一去掉标签里的尾部 '.'
    df["income"] = df["income"].str.strip().str.rstrip(".")

    # 然后再映射
    labels = df["income"].values
    label_mapping = {"<=50K": 0, ">50K": 1}
    y = [label_mapping[val] for val in labels]  # 不会再有找不到的情况了

    # **新增：统计正负标签数**
    num_pos = sum(y)  # 因为所有正标签是1，求和就是正样本数
    num_neg = len(y) - num_pos  # 总数减去正样本数就是负样本数
    print(f"标签分布 → <=50K: {num_neg}，>50K: {num_pos}")


    return X, y, df, transformer


def cluster_nodes_by_dissimilarity_gpu(indices: List[int],
                                       df: pd.DataFrame,
                                       max_nodes: int,
                                       device: torch.device) -> List[List[int]]:
    """Greedy *heterogeneous* clustering on GPU.

    Each cluster gathers **the most mutually dissimilar** nodes until the size
    reaches ``max_nodes``.  Dissimilarity is measured as *Hamming distance* over
    all feature columns (we exclude the label column ``income``).

    Compared with the previous *similarity‑based* routine, we simply sort the
    candidates **ascending** by per‑row similarity to the representative node –
    i.e. we preferentially absorb *least similar* samples first.

    Parameters
    ----------
    indices : list[int]
        Row indices (w.r.t. ``df``) to be clustered.
    df : pd.DataFrame
        Original dataframe that still contains the ``income`` column.
    max_nodes : int
        Maximum nodes allowed in a single cluster.
    device : torch.device
        CUDA / CPU device for vectorised ops.

    Returns
    -------
    clusters : list[list[int]]
        A list of clusters.  Each cluster is a list of *original* dataframe
        indices.
    """
    # 1. Encode categorical & continuous columns (drop the label).
    sub_df = df.loc[indices].drop(columns=["income"])

    encoded_cols = []
    for col in sub_df.columns:
        codes, _ = pd.factorize(sub_df[col])  # consistent integer encoding
        encoded_cols.append(torch.tensor(codes, dtype=torch.long, device=device))

    X = torch.stack(encoded_cols, dim=1)  # (n_samples, n_features)
    n = X.size(0)

    # 2. Greedy clustering – now using *ascending* similarity order.
    unassigned = torch.ones(n, dtype=torch.bool, device=device)
    orig_indices = torch.tensor(indices, dtype=torch.long)
    clusters: List[List[int]] = []

    while unassigned.any():
        # Representative = first unassigned node
        rep_idx = torch.nonzero(unassigned, as_tuple=False)[0].item()
        cluster = [orig_indices[rep_idx].item()]
        unassigned[rep_idx] = False

        remaining = torch.nonzero(unassigned, as_tuple=False).squeeze(1)
        if remaining.numel() > 0:
            rep_vec = X[rep_idx].unsqueeze(0)          # (1, d)
            cand_mat = X[remaining]                    # (r, d)
            # Similarity = #equal attributes
            sim = (cand_mat == rep_vec).sum(dim=1)     # (r,)
            # ⚠️  Sort **ascending** → least similar first
            _, order = torch.sort(sim, descending=False)
            num_take = min(max_nodes - 1, order.numel())
            take = remaining[order[:num_take]]
            for pos in take.tolist():
                cluster.append(orig_indices[pos].item())
                unassigned[pos] = False
        clusters.append(cluster)
    return clusters

# def generate_hyperedge_dict(df,categorical_cols,
#                             max_nodes_per_hyperedge: int,
#                             device=None):
#     """
#     随机构造超边：对所有节点随机打乱，然后按 max_nodes_per_hyperedge 切块。
#
#     参数：
#       df                      (pd.DataFrame): 输入数据，行索引代表节点 id。
#       max_nodes_per_hyperedge (int): 每个超边最大节点数。
#       device                  (torch.device): 占位，保持接口兼容（未使用）。
#
#     返回：
#       hyperedges (dict): {edge_id: [node_indices]}，edge_id 从 0 开始自动编号。
#     """
#     all_indices = df.index.tolist()
#     np.random.shuffle(all_indices)
#
#     hyperedges = {}
#     total = len(all_indices)
#     t0 = time.time()
#     # 切成若干块
#     for edge_id, start in enumerate(range(0, total, max_nodes_per_hyperedge)):
#         chunk = all_indices[start:start + max_nodes_per_hyperedge]
#         hyperedges[edge_id] = chunk
#     t1 = time.time()
#
#     print(f"Random hyperedge generation time: {t1 - t0:.4f} sec")
#     print(f"Total edges: {len(hyperedges)}, "
#           f"average size: {total/len(hyperedges):.1f}")
#     return hyperedges

def generate_hyperedge_dict(df,categorical_cols,
                                        max_nodes_per_hyperedge: int,
                                        device=None):
    """
    为每一列随机构造超边：对每个特征列，将所有节点随机打乱，然后按 max_nodes_per_hyperedge 切块。

    参数：
      df                      (pd.DataFrame): 输入数据，行索引代表节点 id。
      max_nodes_per_hyperedge (int): 每个超边最大节点数。
      device                  (torch.device): 占位，保持接口兼容（未使用）。

    返回：
      hyperedges (dict): {(col, edge_id): [node_indices]}，edge_id 从 0 开始自动编号。
    """
    max_nodes_per_hyperedge=30000000
    hyperedges = {}
    total_columns = len(df.columns)
    start_time = time.time()

    for col in df.columns:
        # 获取全体节点，并随机打乱
        all_indices = df.index.tolist()
        np.random.shuffle(all_indices)

        # 按块切分，生成超边
        for block_id, start in enumerate(range(0, len(all_indices), max_nodes_per_hyperedge)):
            chunk = all_indices[start:start + max_nodes_per_hyperedge]
            # 使用 (列名, 块编号) 作为键
            hyperedges[(col, block_id)] = chunk

    elapsed = time.time() - start_time
    print(f"Random hyperedge generation per column time: {elapsed:.4f} sec")
    total_edges = len(hyperedges)
    avg_size = sum(len(v) for v in hyperedges.values()) / total_edges if total_edges else 0
    print(f"Total edges: {total_edges}, average size: {avg_size:.1f}")

    return hyperedges
