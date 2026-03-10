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
from types import SimpleNamespace

from baseline_Tabnet.MIA_Tabnet import membership_inference
from baseline_Tabnet.reload_unlearning import reload_unlearn_tabnet
from paths import BANK_DATA

def load_and_preprocess_bank(path):
    df = pd.read_csv(path, sep=';')
    df['y'] = df['y'].map({'no': 0, 'yes': 1})

    cont_cols = ['age', 'balance', 'day', 'duration',
                 'campaign', 'pdays', 'previous']
    cat_cols  = [c for c in df.columns if c not in cont_cols + ['y']]

    # 缺失 & 标准化 & 编码
    for c in cont_cols:
        df[c] = df[c].fillna(df[c].median())
    for c in cat_cols:
        df[c] = df[c].fillna('unknown').astype(str)

    scaler = StandardScaler()
    df[cont_cols] = scaler.fit_transform(df[cont_cols])
    for c in cat_cols:
        df[c] = LabelEncoder().fit_transform(df[c])

    X = df[cont_cols + cat_cols].values.astype(np.float32)
    y = df['y'].values.astype(int)
    return X, y, cont_cols, cat_cols

def main():
    mia_args = SimpleNamespace(
        num_shadows=3,
        shadow_n_d=8,
        shadow_n_a=8,
        shadow_n_steps=3,
        shadow_gamma=1.3,
        shadow_lambda=1e-3,
        shadow_lr=1e-2,
        shadow_epochs=150,
        attack_test_split=0.3,
        attack_lr=1e-2,
        attack_epochs=200
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = BANK_DATA

    # 1) 加载并预处理
    X, y, cont_cols, cat_cols = load_and_preprocess_bank(data_path)
    print(f"[Data] samples={X.shape[0]}, features={X.shape[1]}")

    # 2) 确定要遗忘的列索引
    forget_col = 'age'
    if forget_col not in cont_cols:
        raise ValueError(f"Column '{forget_col}' not found in continuous cols")
    col_idx = cont_cols.index(forget_col)

    # 3) 5-fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, f1s = [], []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        # 4) TabNet 基线训练（使用完整特征）
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
            eval_set=[(X_te, y_te)], eval_name=["val"], eval_metric=["accuracy"],
            max_epochs=100, patience=100,
            batch_size=16384, virtual_batch_size=2048,
            num_workers=4, drop_last=False
        )

        preds_base = clf.predict(X_te)
        print(f"Fold {fold} BASELINE: Acc={accuracy_score(y_te, preds_base):.4f},"
              f" F1={f1_score(y_te, preds_base):.4f}")

        # 5) 构造 full_loader 和 retain_loader（age 列置零）
        X_tr_masked = X_tr.copy()
        X_te_masked = X_te.copy()
        X_te_masked[:, col_idx] = 0.0

        full_ds   = TensorDataset(torch.from_numpy(X_tr).float(),
                                   torch.from_numpy(y_tr).long())
        retain_ds = TensorDataset(torch.from_numpy(X_tr_masked).float(),
                                   torch.from_numpy(y_tr).long())
        full_loader   = DataLoader(full_ds,   batch_size=16384, shuffle=True)
        retain_loader = DataLoader(retain_ds, batch_size=16384, shuffle=False)

        # 6) 执行 RELOAD 列级盲遗忘
        def loss_fn(output, target):
            logits = output[0] if isinstance(output, tuple) else output
            return nn.CrossEntropyLoss()(logits, target)

        base_model = clf.network.to(device).eval()
        unlearned_nn = reload_unlearn_tabnet(
            model=base_model,
            train_loader=full_loader,
            retain_loader=retain_loader,
            loss_fn=loss_fn,
            eta_p=0.1,
            reset_frac=0.1,
            ft_epochs=10,
            ft_lr=1e-3,
            eps=1e-8
        ).to(device).eval()

        # 7) 评估卸载后模型
        X_te_t = torch.from_numpy(X_te_masked).float().to(device)
        with torch.no_grad():
            out = unlearned_nn(X_te_t)
            logits = out[0] if isinstance(out, tuple) else out
            preds_un = logits.argmax(dim=1).cpu().numpy()
        accs.append(accuracy_score(y_te, preds_un))
        f1s.append(f1_score(y_te, preds_un))
        print(f"Fold {fold} COLUMN UNLEARNED: Acc={accs[-1]:.4f}, F1={f1s[-1]:.4f}")

if __name__ == "__main__":
    main()
