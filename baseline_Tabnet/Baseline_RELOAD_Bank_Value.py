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
    """
    读取 bank-full.csv（sep=';'），做简单的缺失/unknown 处理、
    连续特征标准化、类别特征 LabelEncode。
    Returns:
      X: np.ndarray, shape=(n_samples, n_features)
      y: np.ndarray, shape=(n_samples,)
    """
    df = pd.read_csv(path, sep=';')
    df['y'] = df['y'].map({'no': 0, 'yes': 1})

    # 连续 vs. 类别特征拆分
    cont_cols = ['age', 'balance', 'day', 'duration',
                 'campaign', 'pdays', 'previous']
    cat_cols  = [c for c in df.columns if c not in cont_cols + ['y']]

    # 缺失处理
    for c in cont_cols:
        df[c] = df[c].fillna(df[c].median())
    for c in cat_cols:
        df[c] = df[c].fillna('unknown').astype(str)

    # 标准化连续
    scaler = StandardScaler()
    df[cont_cols] = scaler.fit_transform(df[cont_cols])

    # LabelEncode 类别
    for c in cat_cols:
        df[c] = LabelEncoder().fit_transform(df[c])

    X = df[cont_cols + cat_cols].values.astype(np.float32)
    y = df['y'].values.astype(int)
    return X, y
def main():
    # 0) MIA 参数
    mia_args = SimpleNamespace(
        num_shadows      = 3,
        shadow_n_d       = 8,
        shadow_n_a       = 8,
        shadow_n_steps   = 3,
        shadow_gamma     = 1.3,
        shadow_lambda    = 1e-3,
        shadow_lr        = 1e-2,
        shadow_epochs    = 150,
        attack_test_split= 0.3,
        attack_lr        = 1e-2,
        attack_epochs    = 200
    )

    # 1) 设备 & 读原始 CSV
    device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = BANK_DATA
    raw_df = pd.read_csv(data_path, sep=';', skipinitialspace=True)

    # 2) 预处理得到特征矩阵 X, 标签 y
    X, y = load_and_preprocess_bank(data_path)
    print(f"[Data] samples={X.shape[0]}, features={X.shape[1]}")

    # 3) 5‐折 CV
    skf  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, f1s = [], []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), start=1):
        print(f"\n--- Fold {fold} ---")
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        df_tr_raw  = raw_df.iloc[tr_idx].reset_index(drop=True)

        # 4) 按值删除：marital=='married' & housing=='yes'
        mask = (df_tr_raw['marital'] == 'married') & (df_tr_raw['housing'] == 'yes')
        forget_idx = np.where(mask.values)[0]
        retain_idx = np.where(~mask.values)[0]
        print(f"按值删除，共 {len(forget_idx)} 样本，占比 {len(forget_idx)/len(df_tr_raw):.2%}")

        X_retain = X_tr[retain_idx]
        y_retain = y_tr[retain_idx]

        # 5) DataLoader
        train_ds      = TensorDataset(torch.from_numpy(X_tr).float(),  torch.from_numpy(y_tr).long())
        train_loader  = DataLoader(train_ds, batch_size=16384, shuffle=True,
                                   num_workers=4, pin_memory=True)
        retain_ds     = TensorDataset(torch.from_numpy(X_retain).float(), torch.from_numpy(y_retain).long())
        retain_loader = DataLoader(retain_ds, batch_size=16384, shuffle=False,
                                   num_workers=4, pin_memory=True)

        # 6) 训练原模型 TabNet
        clf = TabNetClassifier(
            n_d=8, n_a=8, n_steps=3, gamma=1.3, lambda_sparse=1e-3,
            optimizer_fn=torch.optim.Adam, optimizer_params={"lr":2e-2},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            scheduler_params={"step_size":50,"gamma":0.9},
            mask_type="sparsemax", verbose=0, device_name=device.type
        )
        clf.fit(
            X_tr, y_tr,
            eval_set=[(X_te,y_te)], eval_name=["val"], eval_metric=["accuracy"],
            max_epochs=100, patience=100,
            batch_size=16384, virtual_batch_size=2048,
            num_workers=4, drop_last=False
        )

        # 7) RELOAD Unlearning
        def loss_fn(output, target):
            logits = output[0] if isinstance(output, tuple) else output
            return nn.CrossEntropyLoss()(logits, target)

        base_model   = clf.network.to(device).eval()
        unlearned_nn = reload_unlearn_tabnet(
            model=base_model,
            train_loader=train_loader,
            retain_loader=retain_loader,
            loss_fn=loss_fn,
            eta_p=0.1, reset_frac=0.1,
            ft_epochs=10, ft_lr=1e-3,
            eps=1e-8
        ).to(device).eval()

        # 8) Unlearned 模型在测试集上评估
        X_te_t = torch.from_numpy(X_te).float().to(device)
        with torch.no_grad():
            out    = unlearned_nn(X_te_t)
            logits = out[0] if isinstance(out, tuple) else out
            preds  = logits.argmax(dim=1).cpu().numpy()
        accs.append(accuracy_score(y_te, preds))
        f1s.append(f1_score(y_te, preds))
        print(f"Fold {fold} UNLEARNED: Acc={accs[-1]:.4f}, F1={f1s[-1]:.4f}")

        # 9) 构造 MIA 正负例：保留(True) vs 删除(False)
        member_mask = np.zeros(len(X_tr), dtype=bool)
        member_mask[retain_idx] = True

        # 9.1) 原模型 MIA，训练攻击器
        attack_model, (_, _), (auc_orig, _) = membership_inference(
            X_train      = X_tr,
            y_train      = y_tr,
            hyperedges   = None,
            target_model = clf,
            args         = mia_args,
            device       = device,
            member_mask  = member_mask,
            attack_model = None
        )
        print(f"Fold {fold} MIA Original AUC={auc_orig:.4f}")

        # 9.2) Unlearned 模型 MIA，复用攻击器
        # 9.2) Unlearned 模型 MIA，复用攻击器
        # —— 先 new 一个和原模型超参一模一样的 TabNetClassifier —— #
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
            device_name=device.type,
        )
        # —— 把 fine-tuned 的子网络塞进去 —— #
        clf_unlearn.network = unlearned_nn

        # 然后再调用 MIA
        _, (_, _), (auc_un, f1_un) = membership_inference(
            X_train=X_tr,
            y_train=y_tr,
            hyperedges=None,
            target_model=clf_unlearn,  # 传入我们刚刚组装好的 classifier
            args=mia_args,
            device=device,
            member_mask=member_mask,  # True 表示保留样本（正例），False 表示删除样本（负例）
            attack_model=attack_model  # 复用原来的攻击器
        )
        print(f"Fold {fold} MIA Unlearned AUC={auc_un:.4f}, F1={f1_un:.4f}")
    # 10) 汇总
    print(f">> Mean UN Acc: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f">> Mean UN F1 : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
if __name__ == "__main__":
    main()
