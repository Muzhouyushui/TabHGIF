import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from types import SimpleNamespace
from copy import deepcopy
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import TensorDataset, DataLoader
from pytorch_tabnet.tab_model import TabNetClassifier

from reload_unlearning import reload_unlearn_tabnet
from baseline_Tabnet.MIA_Tabnet import membership_inference
from paths import ACI_TEST, ACI_TRAIN
def load_and_preprocess_train(path):
    cols = [
        "age","workclass","fnlwgt","education","education_num",
        "marital_status","occupation","relationship","race","sex",
        "capital_gain","capital_loss","hours_per_week","native_country","income"
    ]
    df = pd.read_csv(path, header=None, names=cols,
                     na_values="?", skipinitialspace=True)

    cont_cols = ["age","fnlwgt","education_num","capital_gain","capital_loss","hours_per_week"]
    cat_cols  = [c for c in cols if c not in cont_cols + ["income"]]

    med_map = {c: df[c].median() for c in cont_cols}
    df.fillna(med_map, inplace=True)
    df.fillna({c: "Missing" for c in cat_cols}, inplace=True)

    df["income"] = df["income"].map({"<=50K":0,">50K":1})

    scaler = StandardScaler()
    df[cont_cols] = scaler.fit_transform(df[cont_cols])

    encoders = {}
    for c in cat_cols:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c])
        encoders[c] = le

    X = df[cont_cols + cat_cols].values
    y = df["income"].values.astype(int)
    return X, y, cont_cols, cat_cols, scaler, encoders

def preprocess_test(path, cont_cols, cat_cols, scaler, encoders):
    cols = [
        "age","workclass","fnlwgt","education","education_num",
        "marital_status","occupation","relationship","race","sex",
        "capital_gain","capital_loss","hours_per_week","native_country","income"
    ]
    df = pd.read_csv(
        path,
        header=None,
        names=cols,
        na_values="?",
        skipinitialspace=True,
        comment="|"
    )

    med_map = {c: df[c].median() for c in cont_cols}
    df.fillna(med_map, inplace=True)
    df.fillna({c: "Missing" for c in cat_cols}, inplace=True)

    df["income"] = df["income"].str.rstrip(".")
    df["income"] = df["income"].map({"<=50K":0,">50K":1})

    df[cont_cols] = scaler.transform(df[cont_cols])
    for c in cat_cols:
        df[c] = encoders[c].transform(df[c])

    X = df[cont_cols + cat_cols].values
    y = df["income"].values.astype(int)
    return X, y

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_path = ACI_TRAIN
    test_path = ACI_TEST

    # 1) 预处理，加载原始数据及特征信息
    X_tr, y_tr, cont_cols, cat_cols, scaler, encoders = load_and_preprocess_train(train_path)
    X_te, y_te = preprocess_test(test_path, cont_cols, cat_cols, scaler, encoders)

    print(f"[Data] train={X_tr.shape[0]} samples, test={X_te.shape[0]} samples")

    # 找到待遗忘列 'age' 在连续特征中的索引
    forget_col = 'age'
    if forget_col in cont_cols:
        col_idx = cont_cols.index(forget_col)
    else:
        raise ValueError(f"Column '{forget_col}' not found in cont_cols")

    runs = 3
    for run in range(1, runs + 1):
        print(f"=== Run {run}/{runs} ===")

        # train/val split for early stopping
        X_train2, X_val, y_train2, y_val = train_test_split(
            X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=run
        )

        # 2) TabNet 全量训练（含 age）
        clf = TabNetClassifier(
            n_d=8, n_a=8, n_steps=3, gamma=1.3, lambda_sparse=1e-3,
            optimizer_fn=torch.optim.Adam, optimizer_params={"lr": 2e-2},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            scheduler_params={"step_size": 50, "gamma": 0.9},
            mask_type="sparsemax", verbose=50, device_name='cuda'
        )
        clf.fit(
            X_train2, y_train2,
            eval_set=[(X_val, y_val)], eval_name=["val"], eval_metric=["accuracy"],
            max_epochs=200, patience=40,
            batch_size=16384, virtual_batch_size=128,
            num_workers=8, drop_last=False
        )

        # 全量测试
        preds_base = clf.predict(X_te)
        acc_base = accuracy_score(y_te, preds_base)
        f1_base = f1_score(y_te, preds_base)
        print(f"[Baseline] Test Acc: {acc_base:.4f}, F1: {f1_base:.4f}")

        # 3) 构造 DataLoader: full_loader 用于缓存 grads_all（原始数据）
        full_ds = TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).long())
        full_loader = DataLoader(full_ds, batch_size=16384, shuffle=True)

        # 4) 构造 retain_loader: 将 age 列置零（不访问原始 age 数据）
        X_tr_masked = X_tr.copy()
        X_tr_masked[:, col_idx] = 0.0
        retain_ds = TensorDataset(torch.from_numpy(X_tr_masked).float(),
                                  torch.from_numpy(y_tr).long())
        retain_loader = DataLoader(retain_ds, batch_size=16384, shuffle=False)

        # 5) RELOAD 列级盲遗忘
        def loss_fn(o, t):
            logits = o[0] if isinstance(o, tuple) else o
            return nn.CrossEntropyLoss()(logits, t)

        model = clf.network.to(device).eval()
        model_un = reload_unlearn_tabnet(
            model, full_loader, retain_loader,
            loss_fn, eta_p=0.1, reset_frac=0.1,
            ft_epochs=10, ft_lr=1e-3
        ).to(device).eval()

        # 6) 构造测试集：将 age 列同样置零
        X_te_masked = X_te.copy()
        X_te_masked[:, col_idx] = 0.0
        Xte_t = torch.from_numpy(X_te_masked).float().to(device)
        with torch.no_grad():
            out = model_un(Xte_t)
            logits = out[0] if isinstance(out, tuple) else out
            preds_un = logits.argmax(dim=1).cpu().numpy()
        acc_un = accuracy_score(y_te, preds_un)
        f1_un = f1_score(y_te, preds_un)
        print(f"[RELOAD Unlearned] Test Acc (age=0): {acc_un:.4f}, F1: {f1_un:.4f}")

if __name__ == "__main__":
    main()