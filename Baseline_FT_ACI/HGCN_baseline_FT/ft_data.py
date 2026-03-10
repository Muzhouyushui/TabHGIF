# ft_data.py
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ADULT_COLS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]

def _read_adult_file(path: str, is_test: bool) -> pd.DataFrame:
    # adult.data / adult.test 都是逗号分隔，且字段可能带空格
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.rstrip("\n") for ln in f]

    # 无论 train/test，只要第一行是说明行（以 | 开头），就跳过
    if len(lines) > 0 and lines[0].strip().startswith("|"):
        lines = lines[1:]

    # 去掉空行
    lines = [ln for ln in lines if ln.strip() != ""]

    # 用 pandas 读：逗号分隔；每列 strip
    # 注意：adult 文件行末可能有 '.'（test label）
    rows = []
    for ln in lines:
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) != 15:
            continue
        rows.append(parts)

    df = pd.DataFrame(rows, columns=ADULT_COLS)

    # 修正 test 标签带 '.' 的情况
    df["income"] = df["income"].astype(str).str.strip()
    df["income"] = df["income"].str.replace(".", "", regex=False)

    return df

def preprocess_node_features(path: str,
                             is_test: bool,
                             categ_cols,
                             label_col="income",
                             transformer=None,
                             filter_missing_q: bool=False):
    """
    返回：
      X: np.ndarray float [N, D]
      y: np.ndarray int   [N]
      df: 原始 DataFrame（未 one-hot）
      transformer: ColumnTransformer（train fit 后复用给 test）
    """
    df = _read_adult_file(path, is_test=is_test)

    # 过滤 '?' 缺失行（如果你原实验需要）
    if filter_missing_q:
        for c in categ_cols:
            if c in df.columns:
                df = df[df[c] != "?"].copy()

    # label:  >50K -> 1 else 0（Adult）
    y = (df[label_col].astype(str).str.strip() == ">50K").astype(int).to_numpy()

    # features
    df_x = df.drop(columns=[label_col])

    # 划分离散/连续列
    cat_cols = [c for c in categ_cols if c in df_x.columns]
    num_cols = [c for c in df_x.columns if c not in cat_cols]

    if transformer is None:
        pre = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), num_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ],
            remainder="drop"
        )
        X = pre.fit_transform(df_x)
        transformer = pre
    else:
        X = transformer.transform(df_x)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    return X, y, df, transformer

def generate_hyperedge_list(df_raw: pd.DataFrame,
                            categ_cols,
                            max_nodes_per_hyperedge: int = 1000,
                            seed: int = 1):
    """
    用“原始离散列取值”诱导超边：
      对每个离散列 c：
        对每个取值 v：
          edge = { i | df[i,c]==v }
    """
    rng = np.random.default_rng(seed)
    edges = []
    N = len(df_raw)

    for c in categ_cols:
        if c not in df_raw.columns:
            continue
        groups = df_raw.groupby(c).indices  # value -> row indices
        for _, idxs in groups.items():
            idxs = list(idxs)
            if len(idxs) < 2:
                continue
            if len(idxs) > max_nodes_per_hyperedge:
                idxs = rng.choice(idxs, size=max_nodes_per_hyperedge, replace=False).tolist()
            edges.append(idxs)

    # 去重（可选）
    # 这里简单保持即可
    return edges
