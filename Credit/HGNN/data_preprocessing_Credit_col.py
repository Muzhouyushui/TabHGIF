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
from scipy.sparse import coo_matrix
import time
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from typing import Union, Tuple
from Credit.HGNN.HGNN import compute_degree_vectors
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
    data: Union[str, pd.DataFrame],
    transformer: ColumnTransformer = None
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, ColumnTransformer]:
    """
    读取并预处理 UCI Credit-Approval 数据集。
    Args:
      data: 文件路径（.data/.csv，无 header，缺失用 '?' 表示）或已加载的 pd.DataFrame。
      transformer: 如果提供，则直接调用 transformer.transform；否则重新 fit 一个。
    Returns:
      X: (n_samples, n_feats) 数值型特征矩阵（numpy array）。
      y: (n_samples,) 0/1 标签，‘+’→1，‘-’→0。
      df: (n_samples, 原始列数) 用于超边构造的 DataFrame（index 已 reset）。
      transformer: 用于数值/类别列的 ColumnTransformer，便于测试集复用。
    """
    # 1. 读入 DataFrame 并设列名
    if isinstance(data, str):
        # UCI 官网 crx.data 无 header，缺失值用 '?'
        df = pd.read_csv(data, header=None, na_values='?')
    else:
        df = data.copy()
        if "y" in df.columns:
            df = df.drop(columns=["y"])
    df = df.reset_index(drop=True)
    # 原始共 16 列：A1~A15 输入 + class 输出
    df.columns = [
        "A1","A2","A3","A4","A5","A6","A7","A8","A9",
        "A10","A11","A12","A13","A14","A15","class"
    ]

    # 2. 划分特征／标签
    feature_cols = [f"A{i}" for i in range(1,16)]
    num_cols = ["A2","A3","A8","A11","A14","A15"]   # 6 个连续
    cat_cols = [c for c in feature_cols if c not in num_cols]  # 9 个类别

    # 3. 标签映射
    y = df["class"].map({"+": 1, "-": 0}).astype(int).values

    # 4. 特征预处理：缺失值、尺度化、独热编码
    X_df = df[feature_cols]
    if transformer is None:
        # 连续管线：中位数填补 + 标准化
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        # 类别管线：常数填补 + 独热
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False  # sklearn>=1.3
            ))
        ])
        transformer = ColumnTransformer([
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ])
        X = transformer.fit_transform(X_df)
    else:
        X = transformer.transform(X_df)

    print(f"[Preprocess] 样本={df.shape[0]} 正={y.sum()} 负={(y==0).sum()} → 特征维度={X.shape[1]}")
    return X, y, df, transformer


def cluster_nodes_by_similarity_gpu(
    indices: list,
    df: pd.DataFrame,
    max_nodes: int,
    device: torch.device
):
    """
    对同一超边中节点按属性相同度做贪心划分，确保每簇 <= max_nodes。
    相似度度量：两个样本在所有字段（除 class 列）上相同的总数。
    """
    # 取子表并去掉标签列
    sub = df.loc[indices].drop(columns=["class"])
    n, d = sub.shape

    # factorize 每列，转成长整型 tensor
    cols = []
    for c in sub.columns:
        codes, _ = pd.factorize(sub[c])
        cols.append(torch.tensor(codes, dtype=torch.long, device=device))
    X = torch.stack(cols, dim=1)  # (n, d)

    unassigned = torch.ones(n, dtype=torch.bool, device=device)
    orig_idx = torch.tensor(indices, dtype=torch.long)  # 对应原 df 索引
    clusters = []

    while unassigned.any():
        # 找到一个代表
        rep = torch.nonzero(unassigned, as_tuple=False)[0].item()
        unassigned[rep] = False
        current = [orig_idx[rep].item()]

        # 剩余候选
        rest = torch.nonzero(unassigned, as_tuple=False).squeeze(1)
        if rest.numel() > 0:
            rep_vec = X[rep].unsqueeze(0)            # (1, d)
            cand   = X[rest]                          # (r, d)
            sim    = (cand == rep_vec).sum(dim=1)     # (r,)
            _, order = torch.sort(sim, descending=True)
            take = min(max_nodes - 1, order.numel())
            chosen = rest[order[:take]]
            for i in chosen.tolist():
                current.append(orig_idx[i].item())
                unassigned[i] = False

        clusters.append(current)
    return clusters


def generate_hyperedge_dict(
    df: pd.DataFrame,
    feature_cols: list = None,
    max_nodes_per_hyperedge: int = 50,
    device: torch.device = torch.device("cpu")
) -> dict:
    """
    同之前版本，按列值/分箱生成超边，自动剔除全 NaN 和常数列。
    返回 {he_id: [node_idx, ...], ...}。
    """
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != "class"]
    hyperedges = {}
    he_id = 0

    for col in feature_cols:
        ser = df[col]
        # 剔除全 NaN
        if ser.isna().all():
            continue
        # 剔除常数列
        vals = ser.dropna().unique()
        if len(vals) < 2:
            continue

        # 类别列
        if ser.dtype == object or ser.dtype.name == "category":
            for v in vals:
                if pd.isna(v):
                    continue
                idxs = df.index[ser == v].tolist()
                if len(idxs) < 2:
                    continue
                for i in range(0, len(idxs), max_nodes_per_hyperedge):
                    hyperedges[he_id] = idxs[i:i+max_nodes_per_hyperedge]
                    he_id += 1

        # 数值列
        else:
            arr = pd.to_numeric(ser, errors="coerce").values
            mask = ~np.isnan(arr)
            if mask.sum() < 2:
                continue
            arr_valid = arr[mask]
            # 四分位分箱
            cuts = np.quantile(arr_valid, [0.0, 0.25, 0.5, 0.75, 1.0])
            bins = np.digitize(arr, bins=cuts[1:-1], right=True)
            for b in np.unique(bins):
                if b < 0:
                    continue
                idxs = df.index[bins == b].tolist()
                if len(idxs) < 2:
                    continue
                for i in range(0, len(idxs), max_nodes_per_hyperedge):
                    hyperedges[he_id] = idxs[i:i+max_nodes_per_hyperedge]
                    he_id += 1

    avg_size = np.mean([len(v) for v in hyperedges.values()]) if hyperedges else 0
    print(f"[Hyperedges] 共 {len(hyperedges)} 条，平均 {avg_size:.2f} 节点/超边")
    return hyperedges


def delete_feature_columns(
    X_tensor: torch.Tensor,
    transformer,
    column_names: list,           # 支持一次删除多列
    hyperedges: dict,             # {he_id: [node_idx, ...], ...}
    df_proc: pd.DataFrame,        # 预处理后 DataFrame，用于重现分组逻辑
    max_nodes_per_hyperedge: int = 50,
    device=None
):
    """
    • 将 column_names 对应的所有特征维度置 0
    • 删除 hyperedges 中那些由 column_names 生成的超边
    • 重建 HGNN 所需的 incidence 矩阵
    返回: (X_zeroed, new_hyperedges, H_tensor_new)
    """

    # —— 1) 零化特征维度 —— #
    try:
        feat_names = transformer.get_feature_names_out()
    except AttributeError:
        feat_names = transformer.get_feature_names()

    del_idx = []
    for col in column_names:
        # 在 feature names 中查找包含该列名的项
        idxs = [i for i, f in enumerate(feat_names)
                if col in re.split(r'__|_', str(f))]
        if idxs:
            print(f"[zero] Zeroing out {len(idxs)} features for '{col}' → {idxs}")
            del_idx.extend(idxs)
        else:
            print(f"[warning] Column '{col}' not found in feature names")

    if del_idx:
        nonzero_before = int((X_tensor[:, del_idx] != 0).sum().item())
        X_tensor[:, del_idx] = 0.0
        nonzero_after  = int((X_tensor[:, del_idx] != 0).sum().item())
        print(f"[verify] Features nonzero before: {nonzero_before}, after: {nonzero_after}")
    else:
        print("[zero] No features were zeroed")

    # —— 2) 在 df_proc 上重现要删除的超边集合 —— #
    to_remove = set()
    for col in column_names:
        if col not in df_proc.columns:
            continue
        ser = df_proc[col]
        # 跳过全 NaN 或者常数列
        if ser.isna().all() or ser.dropna().nunique() < 2:
            continue

        # 离散列
        if ser.dtype == object or ser.dtype.name == "category":
            for v in ser.dropna().unique():
                idxs = df_proc.index[ser == v].tolist()
                if len(idxs) < 2:
                    continue
                for i in range(0, len(idxs), max_nodes_per_hyperedge):
                    to_remove.add(tuple(idxs[i:i+max_nodes_per_hyperedge]))
        # 数值列（四分位分箱）
        else:
            arr = pd.to_numeric(ser, errors="coerce").values
            mask = ~np.isnan(arr)
            if mask.sum() < 2:
                continue
            cuts = np.quantile(arr[mask], [0.25, 0.5, 0.75])
            bins = np.digitize(arr, bins=cuts, right=True)
            for b in np.unique(bins):
                idxs = df_proc.index[bins == b].tolist()
                if len(idxs) < 2:
                    continue
                for i in range(0, len(idxs), max_nodes_per_hyperedge):
                    to_remove.add(tuple(idxs[i:i+max_nodes_per_hyperedge]))

    # —— 3) 过滤 hyperedges —— #
    total_before = len(hyperedges)
    new_hyper = {}
    for he_id, nodes in hyperedges.items():
        if tuple(nodes) in to_remove:
            continue
        new_hyper[he_id] = nodes
    total_after = len(new_hyper)
    removed = total_before - total_after
    print(f"Hyperedges before deletion: {total_before}")
    print(f"Hyperedges after deletion:  {total_after}")
    print(f"Total removed:              {removed} (for columns {column_names})")

    # —— 4) 重建 incidence 矩阵 —— #
    H_sp = build_incidence_matrix(new_hyper, X_tensor.shape[0])
    dv, de = compute_degree_vectors(H_sp)
    coo = H_sp.tocoo()
    idx = torch.LongTensor(np.vstack((coo.row, coo.col))).to(device or X_tensor.device)
    val = torch.FloatTensor(coo.data).to(device or X_tensor.device)
    H_tensor_new = torch.sparse_coo_tensor(
        idx, val, size=coo.shape, device=device or X_tensor.device
    ).coalesce()

    return X_tensor, new_hyper, H_tensor_new