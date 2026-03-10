import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
from paths import CREDIT_DATA

def load_data_crx(path: str):
    """
    加载 UCI Credit-Approval (crx.data) 数据集：
    - 16 列：前 15 列是特征（数值+类别混合），第 16 列是 '+' / '-' 类标签
    - 缺失值用 '?' 表示，这里统一视为 np.nan 并直接 dropna
    - 类标映射：'+'->1, '-'->0
    Returns:
      X: pd.DataFrame, 未做归一化一步
      y: np.ndarray[int], 0/1
      cat_idxs: List[int], X 中哪些列是类别型
      cat_dims: List[int], 对应类别型列的基数
    """
    # crx.data 没有 header，自己给列名
    col_names = [
        "A1","A2","A3","A4","A5","A6","A7","A8",
        "A9","A10","A11","A12","A13","A14","A15","class"
    ]
    df = pd.read_csv(
        path, names=col_names, header=None, na_values="?"
    )
    # 丢弃任何含有缺失值的样本
    df.dropna(inplace=True)
    # 类标 + -> 1, - -> 0
    df["class"] = df["class"].map({"+":1, "-":0}).astype(int)

    y = df["class"].values
    X = df.drop(columns=["class"])

    # 找出哪些列是 object 类型（类别型）
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    cont_cols = [c for c in X.columns if c not in cat_cols]

    # LabelEncode 每个类别列，并记录下 cat_idxs/cat_dims
    cat_idxs = []
    cat_dims = []
    for idx, col in enumerate(X.columns):
        if col in cat_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            cat_idxs.append(idx)
            cat_dims.append(X[col].nunique())

    return X, y, cat_idxs, cat_dims

def run_tabnet_cv(X: pd.DataFrame,
                  y: np.ndarray,
                  cat_idxs: list,
                  cat_dims: list,
                  n_splits: int = 5,
                  seed: int = 2023):
    skf = StratifiedKFold(n_splits=n_splits,
                          shuffle=True,
                          random_state=seed)
    accs, f1s = [], []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_te = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_tr, y_te = y[train_idx], y[test_idx]

        # 从训练集再划出 small-val 用于早停
        X_train, X_val, y_train, y_val = train_test_split(
            X_tr, y_tr,
            test_size=0.2,
            stratify=y_tr,
            random_state=seed+fold
        )

        # 连续特征标准化 —— 先按列名操作，并显式转 float
        cont_cols = [i for i in range(X.shape[1]) if i not in cat_idxs]

        if cont_cols:
            num_cols = [X.columns[i] for i in cont_cols]
            scaler = StandardScaler()
          # 先把这些列全转成 float, 再 fit/transform
            X_train[num_cols] = scaler.fit_transform(
                X_train[num_cols].astype(float)
            )
            X_val[num_cols] = scaler.transform(
                            X_val[num_cols].astype(float)
            )
            X_te[num_cols] = scaler.transform(
                            X_te[num_cols].astype(float)
            )
        # TabNet 要求输入 numpy
        X_train_np = X_train.values
        X_val_np   = X_val.values
        X_test_np  = X_te.values

        # 构建 TabNet
        clf = TabNetClassifier(
            cat_idxs=cat_idxs,
            cat_dims=cat_dims,
            cat_emb_dim=1,        # 类别特征 embedding dim
            optimizer_fn=torch.optim.Adam,
            optimizer_params={"lr":2e-2},
            scheduler_params={"step_size":10, "gamma":0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            mask_type="entmax",
            n_d=16, n_a=16,       # 可根据数据量调整
            n_steps=5,
            lambda_sparse=1e-3,
            verbose=0,
            seed=seed+fold
        )

        # 训练
        clf.fit(
            X_train_np, y_train,
            eval_set=[(X_val_np, y_val)],
            eval_name=["val"],
            eval_metric=["accuracy"],
            max_epochs=100,
            patience=10,
            batch_size=64,
            virtual_batch_size=32
        )

        # 测试
        preds = clf.predict(X_test_np)
        acc = accuracy_score(y_te, preds)
        f1  = f1_score(y_te, preds)
        print(f"Fold {fold:>2d} — Acc: {acc:.4f}, F1: {f1:.4f}")

        accs.append(acc)
        f1s.append(f1)

    print(f"\n>> Mean Acc: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f">> Mean F1 : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

if __name__ == "__main__":
    # 1) 改成你本机的 crx.data 路径
    data_path = CREDIT_DATA
    X, y, cat_idxs, cat_dims = load_data_crx(data_path)

    print(f"[Data] samples={X.shape[0]}, features={X.shape[1]}, "
          f"cats={len(cat_idxs)}, cont={X.shape[1]-len(cat_idxs)}")
    run_tabnet_cv(X, y, cat_idxs, cat_dims, n_splits=5, seed=2025)