import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

from baseline_Tabnet.reload_unlearning import reload_unlearn_tabnet
from paths import CREDIT_DATA

def load_and_preprocess_credit(path):
    """
    读取 Credit 数据集，自动区分数值/类别特征，
    对数值特征做缺失值填充（中位数）+标准化，
    对类别特征做缺失填充('unknown') + LabelEncode。
    返回：
      X: np.ndarray, shape=(n_samples, n_features)
      y: np.ndarray, shape=(n_samples,)
      feature_names: list[str]
    """
    df = pd.read_csv(path)
    label_col = df.columns[-1]

    cont_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if label_col in cont_cols:
        cont_cols.remove(label_col)
    cat_cols = [c for c in df.columns if c not in cont_cols + [label_col]]

    # 填充缺失
    for c in cont_cols:
        df[c] = df[c].fillna(df[c].median())
    for c in cat_cols:
        df[c] = df[c].fillna('unknown').astype(str)

    # 标准化连续特征
    scaler = StandardScaler()
    df[cont_cols] = scaler.fit_transform(df[cont_cols])

    # 编码离散特征
    for c in cat_cols:
        df[c] = LabelEncoder().fit_transform(df[c])

    feature_names = cont_cols + cat_cols
    X = df[feature_names].values.astype(np.float32)

    # 处理标签
    y_raw = df[label_col]
    if y_raw.dtype == object:
        y = LabelEncoder().fit_transform(y_raw).astype(int)
    else:
        y = y_raw.astype(int).values

    return X, y, feature_names

def main():
    # 1) 设备 & 数据加载
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = CREDIT_DATA
    X, y, feature_names = load_and_preprocess_credit(data_path)

    # —— 调试用：打印所有特征名，确认 A5 是哪一个 ——
    print("All feature names:", feature_names)

    # 2) 硬编码删除第 5 列（Python index=4）
    drop_idx = 4
    print(f"Dropping feature at index {drop_idx}: {feature_names[drop_idx]}")

    # 3) 5-fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, f1s = [], []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), start=1):
        # 分割训练/测试
        X_tr, X_te = X[tr_idx].copy(), X[te_idx].copy()
        y_tr, y_te = y[tr_idx], y[te_idx]

        # 4) 构造原始训练 DataLoader（用于训练 base_model）
        train_ds_full = TensorDataset(
            torch.from_numpy(X_tr).float(),
            torch.from_numpy(y_tr).long()
        )
        train_loader_full = DataLoader(
            train_ds_full, batch_size=16384, shuffle=True,
            num_workers=4, pin_memory=True
        )

        # 5) 构造“删除 A5”后的 retain DataLoader（用于 RELOAD）
        X_tr_zero = X_tr.copy()
        X_tr_zero[:, drop_idx] = 0
        retain_ds_zero = TensorDataset(
            torch.from_numpy(X_tr_zero).float(),
            torch.from_numpy(y_tr).long()
        )
        retain_loader_zero = DataLoader(
            retain_ds_zero, batch_size=16384, shuffle=False,
            num_workers=4, pin_memory=True
        )

        # 6) 训练 TabNet（原始特征）
        clf = TabNetClassifier(
            n_d=8, n_a=8, n_steps=3,
            gamma=1.3, lambda_sparse=1e-3,
            optimizer_fn=torch.optim.Adam,
            optimizer_params={"lr": 2e-2},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            scheduler_params={"step_size": 50, "gamma": 0.9},
            mask_type="sparsemax",
            verbose=0,
            device_name=device.type
        )
        clf.fit(
            X_tr, y_tr,
            eval_set=[(X_te, y_te)],
            eval_name=["val"],
            eval_metric=["accuracy"],
            max_epochs=200,
            patience=150,
            batch_size=16384,
            virtual_batch_size=2048,
            num_workers=4,
            drop_last=False
        )

        # 7) base_model
        base_model = clf.network.to(device).eval()

        # 8) RELOAD Unlearning（传入原始 train_loader 和置零后的 retain_loader）
        def loss_fn_wrapper(output, target):
            logits = output[0] if isinstance(output, tuple) else output
            return nn.CrossEntropyLoss()(logits, target)

        unlearned_nn = reload_unlearn_tabnet(
            model=base_model,
            train_loader=train_loader_full,
            retain_loader=retain_loader_zero,
            loss_fn=loss_fn_wrapper,
            eta_p=0.1,
            reset_frac=0.1,
            ft_epochs=10,
            ft_lr=1e-3,
            eps=1e-8
        ).to(device).eval()

        # 9) wrap into TabNetClassifier for unlearned
        clf_unlearn = TabNetClassifier(
            n_d=clf.network.n_d,
            n_a=clf.network.n_a,
            n_steps=clf.network.n_steps,
            gamma=clf.gamma,
            lambda_sparse=clf.lambda_sparse,
            optimizer_fn=clf.optimizer_fn,
            optimizer_params=clf.optimizer_params,
            scheduler_fn=clf.scheduler_fn,
            scheduler_params=clf.scheduler_params,
            mask_type=clf.mask_type,
            verbose=0,
            device_name=device.type
        )
        clf_unlearn.network = unlearned_nn

        # 10) Unlearned 模型评估（test 也置零 A5）
        X_te_zero = X_te.copy()
        X_te_zero[:, drop_idx] = 0
        X_te_t = torch.from_numpy(X_te_zero).float().to(device)
        with torch.no_grad():
            out = unlearned_nn(X_te_t)
            logits = out[0] if isinstance(out, tuple) else out
            preds = logits.argmax(dim=1).cpu().numpy()

        accs.append(accuracy_score(y_te, preds))
        f1s.append(f1_score(y_te, preds))
        print(f"Fold {fold} UNLEARNED: Acc={accs[-1]:.4f}, F1={f1s[-1]:.4f}")

    # 11) 总结
    print(f">> Mean UN Acc: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f">> Mean UN F1 : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

if __name__ == "__main__":
    main()
