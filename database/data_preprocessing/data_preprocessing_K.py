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


def cluster_nodes_by_similarity_gpu(indices, df, max_nodes, device):
    """
    利用 GPU 上的向量化操作对给定节点（来自 df）的索引进行贪心聚类，分成若干簇，
    每个簇最多包含 max_nodes 个节点。相似度定义为各列值（除 income 外）相同的数量。

    参数：
      indices  (list): 原始 df 中的节点索引列表。
      df       (DataFrame): 原始数据。
      max_nodes (int): 单个簇最大节点数。
      device   (torch.device): 使用的设备（GPU）。

    返回：
      clusters (list): 每个簇为一个列表，列表中元素为原始 df 的行索引。
    """
    # 取出相关子 DataFrame，并去掉 "income" 列
    sub_df = df.loc[indices].drop(columns=["income"])
    n_samples = sub_df.shape[0]
    # 对每一列进行 factorize，将其转换为整数编码；注意保持原顺序
    encoded_columns = []
    for col in sub_df.columns:
        codes, _ = pd.factorize(sub_df[col])
        # 转为 torch tensor 并发送到 device
        encoded_columns.append(torch.tensor(codes, dtype=torch.long, device=device))
    # 拼接后得到形状 (n_samples, n_features)
    X = torch.stack(encoded_columns, dim=1)
    n, d = X.shape

    # 使用布尔 mask 记录未分配的样本，初始全 True
    unassigned = torch.ones(n, dtype=torch.bool, device=device)
    clusters = []
    # 为了方便将来返回原始索引，转换输入的 indices 为 tensor（在 CPU 上即可）
    orig_indices = torch.tensor(indices, dtype=torch.long)

    while unassigned.any():
        # 选择第一个未分配的节点作为簇代表
        rep_idx = torch.nonzero(unassigned, as_tuple=False)[0].item()
        current_cluster = [orig_indices[rep_idx].item()]
        unassigned[rep_idx] = False

        # 找出所有未分配节点的索引（在 X 中的行号）
        remaining = torch.nonzero(unassigned, as_tuple=False).squeeze(1)
        if remaining.numel() > 0:
            # 取出代表节点向量：形状 (1, d)
            rep_vector = X[rep_idx].unsqueeze(0)
            # 取出剩余节点，形状 (r, d)
            candidates = X[remaining]
            # 利用向量化比较：计算每个候选与代表在各个特征上的相同（相等）情况，再求和
            sim = (candidates == rep_vector).sum(dim=1)  # 形状 (r,)
            # 获得按照相似度降序排列的索引
            sorted_sim, sorted_order = torch.sort(sim, descending=True)
            # 根据剩余节点数量和 max_nodes 限制，选择部分节点加入当前簇
            num_to_take = min(max_nodes - 1, sorted_order.numel())
            selected = remaining[sorted_order[:num_to_take]]
            # 将选中的节点标记为已分配，并加入当前簇（利用原始索引信息）
            for pos in selected.tolist():
                current_cluster.append(orig_indices[pos].item())
                unassigned[pos] = False
        clusters.append(current_cluster)
    return clusters

def generate_hyperedge_dict(df, categorical_cols, max_nodes_per_hyperedge=None,
                            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    扩展版本：处理离散列和连续列的超边构建。
    对于连续列，根据数据的分布选择合适的区间划分方法。

    参数：
      df                      (DataFrame): 输入数据
      categorical_cols        (list): 用于构造超边的类别属性名称列表。
      continuous_cols         (list): 用于构造超边的连续属性名称列表。
      max_nodes_per_hyperedge (int): 每个超边允许的最大节点数，超过则进行分割。
      device                  (torch.device): 用于聚类计算的设备。

    返回：
      hyperedges (dict): 超边字典。
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
    col_edge_count = {}
    hyperedges = {}
    total_cluster_time = 0.0  # 记录子图划分耗时
    continuous_cols = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
    # 1. 处理离散特征列
    for col in categorical_cols:
        start_count = len(hyperedges)
        unique_vals = df[col].unique()  # 获取该列的所有唯一值
        for val in unique_vals:
            # 获取当前类别值对应的节点索引
            indices = df.index[df[col] == val].tolist()  # 使用 df 中的有效索引

            if len(indices) <= max_nodes_per_hyperedge:
                key = (col, str(val))  # 使用 (特征名, 特征值) 作为键
                hyperedges[key] = indices
            else:
                # 如果节点数超过阈值，则进行聚类分割
                t0 = time.time()
                clusters = cluster_nodes_by_similarity_gpu(indices, df, max_nodes_per_hyperedge, device)
                t1 = time.time()
                total_cluster_time += (t1 - t0)

                # 为每个子簇生成子超边
                for cluster_id, cluster_indices in enumerate(clusters):
                    key = (col, str(val), cluster_id)  # 键中加入子簇编号
                    hyperedges[key] = cluster_indices
        col_edge_count[col] = len(hyperedges) - start_count

    # 2. 处理标准化后的连续特征列
    for col in continuous_cols:
        start_count = len(hyperedges)
        if col == "age":
            # 对 age 特征进行均匀划分区间
            min_val, max_val = df[col].min(), df[col].max()
            num_bins = 10  # 选择 10 个区间
            bins = [min_val + (max_val - min_val) * i / num_bins for i in range(num_bins + 1)]

        # elif col == "capital-gain" or col == "capital-loss":
        #     # 对于 capital-gain 和 capital-loss 使用分位数划分（例如：4个分位数）
        #     min_val, max_val = df[col].min(), df[col].max()
        #     bins = np.percentile(df[col], [0, 25, 50, 75, 100])  # 4个区间（分位数）

        elif col == "capital-gain" or col == "capital-loss":
            min_val, max_val = df[col].min(), df[col].max()
            bins = np.percentile(df[col][df[col] != 0], [0, 25, 50, 75, 100])  # 忽略 0，针对非 0 数据进行分位数划分
            bins = [0] + list(bins)  # 在最小值 0 前加上一个区间，以确保 0 被单独划分

        elif col == "fnlwgt":
            # 对于 fnlwgt 使用分位数划分（例如：4个分位数）
            min_val, max_val = df[col].min(), df[col].max()
            bins = np.percentile(df[col], [0, 25, 50, 75, 100])  # 4个区间（分位数）

        elif col == "hours-per-week":
            # 对于 hours-per-week 使用每 10 小时划分一个区间
            min_val, max_val = df[col].min(), df[col].max()
            bins = [min_val + (max_val - min_val) * i / 10 for i in range(11)]  # 每 10 小时划分一个区间
        elif col == "education-num":
            # 对于 education-num 使用分位数划分（4等分）
            min_val, max_val = df[col].min(), df[col].max()
            bins = np.percentile(df[col], [0, 25, 50, 75, 100]).tolist()

        # 遍历每个区间，生成超边
        for i in range(len(bins) - 1):
            lower, upper = bins[i], bins[i + 1]
            indices = df.index[(df[col] >= lower) & (df[col] < upper)].tolist()  # 获取当前区间内的节点索引

            if len(indices) <= max_nodes_per_hyperedge:
                key = (col, f"{lower:.2f}-{upper:.2f}")  # 区间的键
                hyperedges[key] = indices
            else:
                # 如果节点数超过阈值，则进行聚类分割
                t0 = time.time()
                clusters = cluster_nodes_by_similarity_gpu(indices, df, max_nodes_per_hyperedge, device)
                t1 = time.time()
                total_cluster_time += (t1 - t0)

                # 为每个子簇生成子超边
                for cluster_id, cluster_indices in enumerate(clusters):
                    key = (col, f"{lower:.2f}-{upper:.2f}", cluster_id)  # 子簇编号加入键
                    hyperedges[key] = cluster_indices
        col_edge_count[col] = len(hyperedges) - start_count
    # 打印聚类耗时和生成的超边数量
    print(f"Subgraph (clustering) division total time (GPU): {total_cluster_time:.4f} sec.")
    # print("=== 每条超边节点数 ===")
    # for key, nodes in hyperedges.items():
    #     print(f"Hyperedge {key} 连接了 {len(nodes)} 个节点")
    # 计算并打印平均节点数
    total_edges = len(hyperedges)
    if total_edges > 0:
        total_nodes = sum(len(nodes) for nodes in hyperedges.values())
        avg_nodes = total_nodes / total_edges
        print(f"Average nodes per hyperedge: {avg_nodes:.2f}")
    else:
        print("No hyperedges generated.")

    # print("=== 每列生成的超边数量 ===")
    # for col, cnt in col_edge_count.items():
    #     print(f"  {col}: {cnt}")
    return hyperedges