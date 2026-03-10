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
    cat_cols = [c for c in df.columns if c not in cont_cols + ['y']]

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

def eval_tabnet_on_subset(clf_or_model, X_sub, y_sub, device=None, is_tabnet_classifier=True):
    """
    评估 TabNetClassifier 或已unlearn的 nn.Module 在给定子集上的 Acc/F1（orig-feat）
    """
    if len(X_sub) == 0:
        return np.nan, np.nan

    if is_tabnet_classifier:
        preds = clf_or_model.predict(X_sub)
    else:
        X_t = torch.from_numpy(X_sub).float().to(device)
        with torch.no_grad():
            out = clf_or_model(X_t)
            logits = out[0] if isinstance(out, tuple) else out
            preds = logits.argmax(dim=1).cpu().numpy()

    acc = accuracy_score(y_sub, preds)
    f1 = f1_score(y_sub, preds)
    return acc, f1

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
        shadow_epochs=150,
        attack_test_split=0.3,
        attack_lr=1e-2,
        attack_epochs=200
    )

    # 1) 设备 & 数据加载
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = BANK_DATA

    X, y = load_and_preprocess_bank(data_path)
    print(f"[Data] samples={X.shape[0]}, features={X.shape[1]}")

    # 2) 5-fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    accs, f1s = [], []
    orig_aucs, unl_aucs = [], []
    orig_mia_f1s, unl_mia_f1s = [], []

    # 新增：删除集（forget set）前后指标
    del_acc_befores, del_f1_befores = [], []
    del_acc_afters, del_f1_afters = [], []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), start=1):
        print(f"\n=== Fold {fold}/5 ===")
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        # 3) 随机 Forget 30% 样本（在训练折里）
        n_tr = X_tr.shape[0]
        n_forget = int(n_tr * 0.3)
        perm = np.random.RandomState(fold).permutation(n_tr)
        forget_idx = perm[:n_forget]
        retain_idx = perm[n_forget:]

        X_forget = X_tr[forget_idx]
        y_forget = y_tr[forget_idx]
        X_retain = X_tr[retain_idx]
        y_retain = y_tr[retain_idx]

        print(f"[Fold {fold}] train={n_tr}, forget={len(forget_idx)}, retain={len(retain_idx)}")

        # 4) DataLoader（RELOAD 用）
        train_ds = TensorDataset(
            torch.from_numpy(X_tr).float(),
            torch.from_numpy(y_tr).long()
        )
        train_loader = DataLoader(
            train_ds, batch_size=16384, shuffle=True,
            num_workers=4, pin_memory=True
        )

        retain_ds = TensorDataset(
            torch.from_numpy(X_retain).float(),
            torch.from_numpy(y_retain).long()
        )
        retain_loader = DataLoader(
            retain_ds, batch_size=16384, shuffle=False,
            num_workers=4, pin_memory=True
        )

        # 5) 训练 TabNet（原模型）
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
            max_epochs=100,
            patience=100,
            batch_size=16384,
            virtual_batch_size=2048,
            num_workers=4,
            drop_last=False
        )

        # 原模型测试集表现（可选打印）
        preds_base = clf.predict(X_te)
        acc_base = accuracy_score(y_te, preds_base)
        f1_base = f1_score(y_te, preds_base)
        print(f"Fold {fold} BASE: Acc={acc_base:.4f}, F1={f1_base:.4f}")

        # ===== 新增：删除集 Forget Accuracy（orig-feat）- BEFORE =====
        del_acc_before, del_f1_before = eval_tabnet_on_subset(
            clf, X_forget, y_forget, device=device, is_tabnet_classifier=True
        )
        print(f"[ForgetMetric|orig-feat|RELOAD] Fold {fold} deleted-set BEFORE Acc={del_acc_before:.4f}, F1={del_f1_before:.4f}")

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

        # wrap into a new TabNetClassifier for MIA
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

        # 7) Unlearned 模型测试集评估
        X_te_t = torch.from_numpy(X_te).float().to(device)
        with torch.no_grad():
            out = unlearned_nn(X_te_t)
            logits = out[0] if isinstance(out, tuple) else out
            preds = logits.argmax(dim=1).cpu().numpy()

        acc_un = accuracy_score(y_te, preds)
        f1_un = f1_score(y_te, preds)
        accs.append(acc_un)
        f1s.append(f1_un)
        print(f"Fold {fold} UNLEARNED: Acc={acc_un:.4f}, F1={f1_un:.4f}")

        # ===== 新增：删除集 Forget Accuracy（orig-feat）- AFTER =====
        del_acc_after, del_f1_after = eval_tabnet_on_subset(
            unlearned_nn, X_forget, y_forget, device=device, is_tabnet_classifier=False
        )
        print(f"[ForgetMetric|orig-feat|RELOAD] Fold {fold} deleted-set AFTER  Acc={del_acc_after:.4f}, F1={del_f1_after:.4f}")
        print(f"[ForgetMetric|orig-feat|RELOAD] Fold {fold} ΔAcc={del_acc_after - del_acc_before:+.4f}, ΔF1={del_f1_after - del_f1_before:+.4f}")

        del_acc_befores.append(del_acc_before)
        del_f1_befores.append(del_f1_before)
        del_acc_afters.append(del_acc_after)
        del_f1_afters.append(del_f1_after)

        # 8) MIA: member_mask（retain 为 member，forget 为 non-member）
        member_mask = np.zeros(len(X_tr), dtype=bool)
        member_mask[retain_idx] = True

        # —— 8.1) 原模型 MIA：训练并返回攻击器 ——
        attack_model, (auc_shadow, f1_shadow), (auc_orig, f1_orig) = membership_inference(
            X_train=X_tr,
            y_train=y_tr,
            hyperedges=None,
            target_model=clf,
            args=mia_args,
            device=device,
            member_mask=member_mask,
            attack_model=None
        )
        orig_aucs.append(auc_orig)
        orig_mia_f1s.append(f1_orig)
        print(f"Fold {fold} MIA Original: ShadowAUC={auc_shadow:.4f}, OrigAUC={auc_orig:.4f}, OrigF1={f1_orig:.4f}")

        # —— 8.2) Unlearned 模型 MIA：复用同一攻击器 ——
        _, (_, _), (auc_un, f1_un_mia) = membership_inference(
            X_train=X_tr,
            y_train=y_tr,
            hyperedges=None,
            target_model=clf_unlearn,
            args=mia_args,
            device=device,
            member_mask=member_mask,
            attack_model=attack_model
        )
        unl_aucs.append(auc_un)
        unl_mia_f1s.append(f1_un_mia)
        print(f"Fold {fold} MIA Unlearned: AUC={auc_un:.4f}, F1={f1_un_mia:.4f}")

        # 本 fold 汇总（便于审稿回复摘数）
        print(
            f"[Fold {fold} Summary] "
            f"TestAcc {acc_base:.4f}->{acc_un:.4f} | "
            f"DelAcc {del_acc_before:.4f}->{del_acc_after:.4f} | "
            f"MIA AUC {auc_orig:.4f}->{auc_un:.4f}"
        )

    # 9) 总结
    print("\n================ Overall Summary (5-fold) ================")
    print(f">> Mean UNLEARNED Acc : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f">> Mean UNLEARNED F1  : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

    print(f">> Mean MIA Orig AUC  : {np.mean(orig_aucs):.4f} ± {np.std(orig_aucs):.4f}")
    print(f">> Mean MIA Unl  AUC  : {np.mean(unl_aucs):.4f} ± {np.std(unl_aucs):.4f}")
    print(f">> Mean MIA Orig F1   : {np.mean(orig_mia_f1s):.4f} ± {np.std(orig_mia_f1s):.4f}")
    print(f">> Mean MIA Unl  F1   : {np.mean(unl_mia_f1s):.4f} ± {np.std(unl_mia_f1s):.4f}")

    # 新增：审稿人关心的 Forget Accuracy（deleted-set/orig-feat）
    print("\n===== Forget Accuracy on Deleted Set (orig-feat) =====")
    print(f">> Mean Deleted Acc BEFORE: {np.mean(del_acc_befores):.4f} ± {np.std(del_acc_befores):.4f}")
    print(f">> Mean Deleted Acc AFTER : {np.mean(del_acc_afters):.4f} ± {np.std(del_acc_afters):.4f}")
    print(f">> Mean Deleted F1  BEFORE: {np.mean(del_f1_befores):.4f} ± {np.std(del_f1_befores):.4f}")
    print(f">> Mean Deleted F1  AFTER : {np.mean(del_f1_afters):.4f} ± {np.std(del_f1_afters):.4f}")
    print(f">> Mean ΔDeleted Acc      : {(np.mean(del_acc_afters) - np.mean(del_acc_befores)):+.4f}")
    print(f">> Mean ΔDeleted F1       : {(np.mean(del_f1_afters) - np.mean(del_f1_befores)):+.4f}")

if __name__ == "__main__":
    main()