# tabnet_baseline_bank.py

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
from paths import BANK_DATA

def load_and_preprocess_bank(path):
    """
    读取 bank-full.csv（sep=';'），做简单的缺失/unknown 处理、
    连续特征标准化、类别特征 LabelEncode。
    Returns:
      X: np.ndarray, shape=(n_samples, n_features)
      y: np.ndarray, shape=(n_samples,)
      cont_cols, cat_cols: 列名列表
    """
    # 1) 读取原始数据
    df = pd.read_csv(path, sep=';')

    # 2) 目标列二值化
    df['y'] = df['y'].map({'no': 0, 'yes': 1})

    # 3) 缺失值/Unknown 处理
    for c in df.columns:
        if df[c].dtype == object:
            # bank 数据集里的 unknown，全部当成一个类别
            df[c] = df[c].fillna('unknown').astype(str)
        else:
            df[c] = df[c].fillna(df[c].median())

    # 4) 指定连续 & 类别特征
    cont_cols = [
        'age', 'balance', 'day', 'duration',
        'campaign', 'pdays', 'previous'
    ]
    cat_cols = [c for c in df.columns if c not in cont_cols + ['y']]

    # 5) 连续特征标准化
    scaler = StandardScaler()
    df[cont_cols] = scaler.fit_transform(df[cont_cols])

    # 6) 类别特征 Label Encoding
    encoders = {}
    for c in cat_cols:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c])
        encoders[c] = le

    # 7) 构造最终数组
    X = df[cont_cols + cat_cols].values
    y = df['y'].values.astype(int)
    return X, y, cont_cols, cat_cols

def main():
    # 0) 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # 1) 加载 & 预处理
    data_path = BANK_DATA
    X, y, cont_cols, cat_cols = load_and_preprocess_bank(data_path)
    print(f"[Data] samples={X.shape[0]}, features={X.shape[1]}, "
          f"cont={len(cont_cols)}, cat={len(cat_cols)}")

    # 2) 5-折交叉验证
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, f1s = [], []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        # 3) 定义 TabNet 模型
        clf = TabNetClassifier(
            n_d=8, n_a=8, n_steps=3,
            gamma=1.3, lambda_sparse=1e-3,
            optimizer_fn=torch.optim.Adam,
            optimizer_params={"lr": 2e-2},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            scheduler_params={"step_size": 50, "gamma": 0.9},
            mask_type="sparsemax",
            verbose=0
        )

        # 4) 训练（只用 accuracy 做 early‐stop）
        clf.fit(
            X_tr, y_tr,
            eval_set=[(X_te, y_te)],
            eval_name=["val"],
            eval_metric=["accuracy"],
            max_epochs=100,
            patience=20,
            batch_size=1024,
            virtual_batch_size=128,
            num_workers=0,
            drop_last=False
        )

        # 5) 测试集上计算 Acc & F1
        preds = clf.predict(X_te)
        acc = accuracy_score(y_te, preds)
        f1 = f1_score(y_te, preds)
        print(f"Fold {fold}: Acc={acc:.4f}, F1={f1:.4f}")
        accs.append(acc)
        f1s.append(f1)

    # 6) 输出平均与标准差
    print(f">> Mean Acc: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f">> Mean F1 : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

if __name__ == "__main__":
    main()
