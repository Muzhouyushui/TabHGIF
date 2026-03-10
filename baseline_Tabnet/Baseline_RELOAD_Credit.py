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
from paths import CREDIT_DATA

def load_and_preprocess_credit(path):
    """
    读取 Credit 数据集，自动区分数值/类别特征，
    对数值特征做缺失值填充（中位数）+标准化，
    对类别特征做缺失填充('unknown') + LabelEncode。
    返回：
      X: np.ndarray, shape=(n_samples, n_features)
      y: np.ndarray, shape=(n_samples,)
    """
    df = pd.read_csv(path)
    label_col = df.columns[-1]

    cont_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if label_col in cont_cols:
        cont_cols.remove(label_col)
    cat_cols = [c for c in df.columns if c not in cont_cols + [label_col]]

    for c in cont_cols:
        df[c] = df[c].fillna(df[c].median())
    for c in cat_cols:
        df[c] = df[c].fillna('unknown').astype(str)

    scaler = StandardScaler()
    df[cont_cols] = scaler.fit_transform(df[cont_cols])

    for c in cat_cols:
        df[c] = LabelEncoder().fit_transform(df[c])

    X = df[cont_cols + cat_cols].values.astype(np.float32)
    y_raw = df[label_col]
    if y_raw.dtype == object:
        y = LabelEncoder().fit_transform(y_raw).astype(int)
    else:
        y = y_raw.astype(int).values
    return X, y

def main():
    # 0) MIA 参数
    mia_args = SimpleNamespace(
        num_shadows=3,
        shadow_n_d=8,
        shadow_n_a=8,
        shadow_n_steps=3,
        shadow_gamma=1.3,
        shadow_lambda=1e-3,
        shadow_lr=1e-2,
        shadow_epochs=200,
        attack_test_split=0.3,
        attack_lr=1e-2,
        attack_epochs=200
    )

    # 1) 设备 & 数据加载
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = CREDIT_DATA
    X, y = load_and_preprocess_credit(data_path)
    print(f"[Data] samples={X.shape[0]}, features={X.shape[1]}")

    # 2) 5-fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, f1s = [], []
    mia_orig_aucs, mia_un_aucs = [], []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        # 3) 随机 Forget 30% 样本
        n_tr = X_tr.shape[0]
        n_forget = int(n_tr * 0.3)
        perm = np.random.RandomState(fold).permutation(n_tr)
        forget_idx = perm[:n_forget]
        retain_idx = perm[n_forget:]
        X_retain, y_retain = X_tr[retain_idx], y_tr[retain_idx]

        # 4) DataLoader
        train_ds = TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).long())
        train_loader = DataLoader(train_ds, batch_size=16384, shuffle=True,
                                   num_workers=4, pin_memory=True)
        retain_ds = TensorDataset(torch.from_numpy(X_retain).float(), torch.from_numpy(y_retain).long())
        retain_loader = DataLoader(retain_ds, batch_size=16384, shuffle=False,
                                   num_workers=4, pin_memory=True)

        # 5) 训练 TabNet
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

        # 6) RELOAD Unlearning
        def loss_fn_wrapper(output, target):
            logits = output[0] if isinstance(output, tuple) else output
            return nn.CrossEntropyLoss()(logits, target)

        base_model = clf.network.to(device).eval()
        unlearned_nn = reload_unlearn_tabnet(
            model=base_model,
            train_loader=train_loader,
            retain_loader=retain_loader,
            loss_fn=loss_fn_wrapper,
            eta_p=0.1,
            reset_frac=0.1,
            ft_epochs=10,
            ft_lr=1e-3,
            eps=1e-8
        ).to(device).eval()

        # wrap into a new TabNetClassifier for MIA on unlearned model
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

        # 7) Unlearned 模型评估
        X_te_t = torch.from_numpy(X_te).float().to(device)
        with torch.no_grad():
            out = unlearned_nn(X_te_t)
            logits = out[0] if isinstance(out, tuple) else out
            preds = logits.argmax(dim=1).cpu().numpy()
        accs.append(accuracy_score(y_te, preds))
        f1s.append(f1_score(y_te, preds))
        print(f"Fold {fold} UNLEARNED: Acc={accs[-1]:.4f}, F1={f1s[-1]:.4f}")

        # 8) MIA: 构造 member_mask
        member_mask = np.zeros(len(X_tr), dtype=bool)
        member_mask[retain_idx] = True

        # 8.1) 原模型 MIA —— 训练并返回攻击器
        attack_model, (auc_shadow, f1_shadow), (auc_orig, f1_orig) = membership_inference(
            X_train      = X_tr,
            y_train      = y_tr,
            hyperedges   = None,
            target_model = clf,
            args         = mia_args,
            device       = device,
            member_mask  = member_mask,
            attack_model = None
        )
        mia_orig_aucs.append(auc_orig)
        print(f"Fold {fold} MIA Original: ShadowAUC={auc_shadow:.4f}, OrigAUC={auc_orig:.4f}")

        # 8.2) Unlearned 模型 MIA —— 复用同一攻击器
        _, (_, _), (auc_un, f1_un) = membership_inference(
            X_train      = X_tr,
            y_train      = y_tr,
            hyperedges   = None,
            target_model = clf_unlearn,
            args         = mia_args,
            device       = device,
            member_mask  = member_mask,
            attack_model = attack_model
        )
        mia_un_aucs.append(auc_un)
        print(f"Fold {fold} MIA Unlearned: UnlearnAUC={auc_un:.4f}, F1={f1_un:.4f}")

    # 9) 总结
    print(f">> Mean UN Acc : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f">> Mean UN F1  : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f">> Mean Orig AUC: {np.mean(mia_orig_aucs):.4f} ± {np.std(mia_orig_aucs):.4f}")
    print(f">> Mean Unlearn AUC: {np.mean(mia_un_aucs):.4f} ± {np.std(mia_un_aucs):.4f}")

if __name__ == "__main__":
    main()

# import os
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import StratifiedKFold
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.metrics import accuracy_score, f1_score
# from pytorch_tabnet.tab_model import TabNetClassifier
# import torch
# from torch.utils.data import TensorDataset, DataLoader
# import torch.nn as nn
# from types import SimpleNamespace
#
# from baseline_Tabnet.MIA_Tabnet import membership_inference
# from baseline_Tabnet.reload_unlearning import reload_unlearn_tabnet
#
#
# def load_and_preprocess_credit(path):
#     """
#     读取 Credit 数据集，自动区分数值/类别特征，
#     对数值特征做缺失值填充（中位数）+标准化，
#     对类别特征做缺失填充('unknown') + LabelEncode。
#     返回：
#       X: np.ndarray, shape=(n_samples, n_features)
#       y: np.ndarray, shape=(n_samples,)
#     """
#     df = pd.read_csv(path)
#     # 假设最后一列是 label
#     label_col = df.columns[-1]
#
#     # 区分数值 vs. 类别
#     cont_cols = df.select_dtypes(include=[np.number]).columns.tolist()
#     if label_col in cont_cols:
#         cont_cols.remove(label_col)
#     cat_cols = [c for c in df.columns if c not in cont_cols + [label_col]]
#
#     # 填充缺失
#     for c in cont_cols:
#         df[c] = df[c].fillna(df[c].median())
#     for c in cat_cols:
#         df[c] = df[c].fillna('unknown').astype(str)
#
#     # 标准化数值
#     scaler = StandardScaler()
#     df[cont_cols] = scaler.fit_transform(df[cont_cols])
#
#     # LabelEncode 类别
#     for c in cat_cols:
#         df[c] = LabelEncoder().fit_transform(df[c])
#
#     # 构造 X, y
#     X = df[cont_cols + cat_cols].values.astype(np.float32)
#     # 如果 label 已经是 0/1 或整数，直接 astype(int)；否则自动映射
#     y_raw = df[label_col]
#     if y_raw.dtype == object:
#         y = LabelEncoder().fit_transform(y_raw).astype(int)
#     else:
#         y = y_raw.astype(int).values
#     return X, y
#
#
# def main():
#     # 0) MIA 参数
#     mia_args = SimpleNamespace(
#         num_shadows=5,
#         shadow_n_d=8,
#         shadow_n_a=8,
#         shadow_n_steps=3,
#         shadow_gamma=1.3,
#         shadow_lambda=1e-3,
#         shadow_lr=1e-2,
#         shadow_epochs=50,
#         attack_test_split=0.3,
#         attack_lr=1e-2,
#         attack_epochs=50
#     )
#
#     # 1) 设备 & 数据加载
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     X, y = load_and_preprocess_credit(data_path)
#     print(f"[Data] samples={X.shape[0]}, features={X.shape[1]}")
#
#     # 2) 5-fold CV
#     skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#     accs, f1s = [], []
#
#     for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), start=1):
#         X_tr, X_te = X[tr_idx], X[te_idx]
#         y_tr, y_te = y[tr_idx], y[te_idx]
#
#         # 3) 随机 Forget 30% 样本
#         n_tr = X_tr.shape[0]
#         n_forget = int(n_tr * 0.3)
#         perm = np.random.RandomState(fold).permutation(n_tr)
#         forget_idx = perm[:n_forget]
#         retain_idx = perm[n_forget:]
#         X_retain = X_tr[retain_idx]
#         y_retain = y_tr[retain_idx]
#
#         # 4) DataLoader
#         train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
#         train_loader = DataLoader(train_ds, batch_size=16384, shuffle=True,
#                                   num_workers=4, pin_memory=True)
#         retain_ds = TensorDataset(torch.from_numpy(X_retain), torch.from_numpy(y_retain))
#         retain_loader = DataLoader(retain_ds, batch_size=16384, shuffle=False,
#                                    num_workers=4, pin_memory=True)
#
#         # 5) 训练 TabNet
#         clf = TabNetClassifier(
#             n_d=8, n_a=8, n_steps=3,
#             gamma=1.3, lambda_sparse=1e-3,
#             optimizer_fn=torch.optim.Adam,
#             optimizer_params={"lr": 2e-2},
#             scheduler_fn=torch.optim.lr_scheduler.StepLR,
#             scheduler_params={"step_size": 50, "gamma": 0.9},
#             mask_type="sparsemax",
#             verbose=0,
#             device_name=device.type
#         )
#         clf.fit(
#             X_tr, y_tr,
#             eval_set=[(X_te, y_te)],
#             eval_name=["val"],
#             eval_metric=["accuracy"],
#             max_epochs=200,
#             patience=150,
#             batch_size=16384,
#             virtual_batch_size=2048,
#             num_workers=4,
#             drop_last=False
#         )
#
#         # 6) RELOAD Unlearning
#         def loss_fn_wrapper(output, target):
#             logits = output[0] if isinstance(output, tuple) else output
#             return nn.CrossEntropyLoss()(logits, target)
#
#         base_model = clf.network.to(device).eval()
#         unlearned_nn = reload_unlearn_tabnet(
#             model=base_model,
#             train_loader=train_loader,
#             retain_loader=retain_loader,
#             loss_fn=loss_fn_wrapper,
#             eta_p=0.1,
#             reset_frac=0.1,
#             ft_epochs=10,
#             ft_lr=1e-3,
#             eps=1e-8
#         ).to(device).eval()
#
#         # wrap into a new TabNetClassifier for MIA
#         clf_unlearn = TabNetClassifier(
#             n_d=clf.network.n_d,
#             n_a=clf.network.n_a,
#             n_steps=clf.network.n_steps,
#             gamma=clf.gamma,
#             lambda_sparse=clf.lambda_sparse,
#             optimizer_fn=clf.optimizer_fn,
#             optimizer_params=clf.optimizer_params,
#             scheduler_fn=clf.scheduler_fn,
#             scheduler_params=clf.scheduler_params,
#             mask_type=clf.mask_type,
#             verbose=0,
#             device_name=device.type
#         )
#         clf_unlearn.network = unlearned_nn
#
#         # 7) Unlearned 模型评估
#         X_te_t = torch.from_numpy(X_te).float().to(device)
#         with torch.no_grad():
#             out = unlearned_nn(X_te_t)
#             logits = out[0] if isinstance(out, tuple) else out
#             preds = logits.argmax(dim=1).cpu().numpy()
#         accs.append(accuracy_score(y_te, preds))
#         f1s.append(f1_score(y_te, preds))
#         print(f"Fold {fold} UNLEARNED: Acc={accs[-1]:.4f}, F1={f1s[-1]:.4f}")
#
#     # 8) 总结
#     print(f">> Mean UN Acc: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
#     print(f">> Mean UN F1 : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
#
#
# if __name__ == "__main__":
#     main()
