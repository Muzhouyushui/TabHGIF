import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
from paths import ACI_TRAIN

def load_and_preprocess(path):
    # Adult 列名
    cols = [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
    ]
    df = pd.read_csv(path, header=None, names=cols, na_values="?", skipinitialspace=True)
    # 删除含过多缺失的列(如 fnlwgt) 可选
    # df = df.drop(columns=["fnlwgt"])

    # 处理缺失：类别用 "Missing"，连续用中位数
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].fillna("Missing")
        else:
            df[c] = df[c].fillna(df[c].median())

    # 标签二值化
    df["income"] = df["income"].map({"<=50K": 0, ">50K": 1})

    # 划分特征列
    cont_cols = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
    cat_cols = [c for c in cols if c not in cont_cols + ["income"]]

    # 标准化连续
    scaler = StandardScaler()
    df[cont_cols] = scaler.fit_transform(df[cont_cols])

    # LabelEncode 类别
    encoders = {}
    for c in cat_cols:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c])
        encoders[c] = le

    X = df[cont_cols + cat_cols].values
    y = df["income"].values.astype(int)
    return X, y, cont_cols, cat_cols

def main():
    # 1. 加载预处理
    data_path = ACI_TRAIN
    X, y, cont_cols, cat_cols = load_and_preprocess(data_path)
    print(f"[Data] samples={X.shape[0]}, features={X.shape[1]}, cont={len(cont_cols)}, cat={len(cat_cols)}")

    # 2. 5-折 CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, f1s = [], []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

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

        # 1) 训练时只用 accuracy 作为 early‐stop 指标
        clf.fit(
            X_tr, y_tr,
            eval_set=[(X_te, y_te)],
            eval_name=["val"],
            eval_metric=["accuracy"],  # <-- 这里删除了 'f1'
            max_epochs=100,
            patience=20,
            batch_size=10240,
            virtual_batch_size=1280,
            num_workers=0,
            drop_last=False
        )

        # 2) 训练后再手动算一次 f1
        preds = clf.predict(X_te)
        acc = accuracy_score(y_te, preds)
        f1 = f1_score(y_te, preds)

        print(f"Fold {fold}: Acc={acc:.4f}, F1={f1:.4f}")
        accs.append(acc)
        f1s.append(f1)

    # 最后再把 mean±std 打印出来
    print(f">> Mean Acc: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f">> Mean F1 : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

if __name__ == "__main__":
    main()