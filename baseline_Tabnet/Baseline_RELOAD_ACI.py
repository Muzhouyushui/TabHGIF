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
    mia_args = SimpleNamespace(
        num_shadows=3, shadow_n_d=8, shadow_n_a=8, shadow_n_steps=3,
        shadow_gamma=1.3, shadow_lambda=1e-3,
        shadow_lr=1e-2, shadow_epochs=200, shadow_patience=50,
        attack_test_split=0.3, attack_lr=1e-2, attack_epochs=200
    )

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_path = ACI_TRAIN
    test_path  = ACI_TEST
    # parser.add_argument("--train_csv", type=str,
    #                     help="训练数据 CSV 路径")
    # parser.add_argument("--test_csv", type=str,
    #                     help="测试数据 CSV 路径")

    # 1) 预处理
    X_tr, y_tr, cont_cols, cat_cols, scaler, encoders = load_and_preprocess_train(train_path)
    X_te, y_te = preprocess_test(test_path, cont_cols, cat_cols, scaler, encoders)

    print(f"[Data] train={X_tr.shape[0]} samples, test={X_te.shape[0]} samples")

    runs = 5
    unlearn_accs, unlearn_f1s = [], []
    orig_aucs, unl_aucs       = [], []

    for run in range(1, runs+1):
        print(f"=== Run {run}/{runs} ===")

        # train/val split for early stopping
        X_train2, X_val, y_train2, y_val = train_test_split(
            X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=run
        )

        # full 和 retain 的 DataLoader
        full_ds     = TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).long())
        full_loader = DataLoader(full_ds, batch_size=16384, shuffle=True)

        perm        = np.random.RandomState(seed=run).permutation(len(X_tr))
        n_forget    = int(0.3 * len(X_tr))
        retain_idx  = perm[n_forget:]
        retain_ds   = TensorDataset(
            torch.from_numpy(X_tr[retain_idx]).float(),
            torch.from_numpy(y_tr[retain_idx]).long()
        )
        retain_loader = DataLoader(retain_ds, batch_size=16384, shuffle=False)

        # 1) Baseline TabNet
        clf = TabNetClassifier(
            n_d=8, n_a=8, n_steps=3, gamma=1.3, lambda_sparse=1e-3,
            optimizer_fn=torch.optim.Adam, optimizer_params={"lr":2e-2},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            scheduler_params={"step_size":50,"gamma":0.9},
            mask_type="sparsemax", verbose=50, device_name='cuda'
        )
        clf.fit(
            X_train2, y_train2,
            eval_set=[(X_val,y_val)], eval_name=["val"], eval_metric=["accuracy"],
            max_epochs=200, patience=40,
            batch_size=16384, virtual_batch_size=128,
            num_workers=8, drop_last=False
        )

        preds_base = clf.predict(X_te)
        acc_base   = accuracy_score(y_te, preds_base)
        f1_base    = f1_score(y_te, preds_base)

        # 2) RELOAD Unlearning
        def loss_fn(o,t):
            logits = o[0] if isinstance(o,tuple) else o
            return nn.CrossEntropyLoss()(logits, t)

        model    = clf.network.to(device).eval()
        model_un = reload_unlearn_tabnet(
            model, full_loader, retain_loader,
            loss_fn, eta_p=0.1, reset_frac=0.1,
            ft_epochs=10, ft_lr=1e-3
        ).to(device).eval()

        Xte_t = torch.from_numpy(X_te).float().to(device)
        with torch.no_grad():
            out      = model_un(Xte_t)
            logits   = out[0] if isinstance(out,tuple) else out
            preds_un = logits.argmax(dim=1).cpu().numpy()
        acc_un = accuracy_score(y_te, preds_un)
        f1_un  = f1_score(y_te, preds_un)
        print(f"[RELOAD Unlearning] Test Acc: {acc_un:.4f}, Test F1: {f1_un:.4f}")

        # 3) 构造 keep / del 攻击集
        #    X_tr, y_tr 原始训练集； retain_idx 是保留下来的索引
        all_idx = np.arange(len(X_tr))
        del_idx = np.setdiff1d(all_idx, retain_idx)  # 这就是被删样本
        X_keep = X_tr[retain_idx]
        y_keep = y_tr[retain_idx]
        X_del = X_tr[del_idx]
        y_del = y_tr[del_idx]

        X_attack = np.vstack([X_keep, X_del])
        y_attack = np.hstack([y_keep, y_del])

        member_mask = np.array([True] * len(X_keep) + [False] * len(X_del))

        # 4) MIA on original
        # 4) 原模型 MIA：训练并返回 attack_model
        attack_model, _,  (auc_orig, _) = membership_inference(
            X_attack, y_attack, None,
            target_model=clf,
            args=mia_args,
            device=device,
            member_mask=member_mask,
            attack_model=None
        )

        # 5) MIA on unlearned
        clf_un = TabNetClassifier(  # 重新包装 network
            n_d=clf.network.n_d, n_a=clf.network.n_a, n_steps=clf.network.n_steps,
            gamma=clf.gamma, lambda_sparse=clf.lambda_sparse,
            optimizer_fn=clf.optimizer_fn, optimizer_params=clf.optimizer_params,
            scheduler_fn=clf.scheduler_fn, scheduler_params=clf.scheduler_params,
            mask_type=clf.mask_type, verbose=0, device_name='cuda'
        )
        clf_un.network = model_un

        # 5) 对 unlearned 模型 MIA —— 复用同一个 attack_model
        _, (_, _), (auc_unl, f1_unl) = membership_inference(
            X_attack,
            y_attack, None,
            # target_model=model_un,  # 直接传入已经 unlearn 好的 nn.Module
            target_model=clf_un,
            args=mia_args,
            device=device,
            member_mask=member_mask,
            attack_model=attack_model,  # 复用原先训练好的攻击器
        )

        print(f"MIA Original AUC={auc_orig:.4f}, After Unlearning AUC={auc_unl:.4f}")

        # 6) 打印本次 Run 的结果
        print(
            f"Run {run:>2} | BASE Acc={acc_base:.4f} F1={f1_base:.4f}"
            f" | UNL Acc={acc_un:.4f} F1={f1_un:.4f}"
            f" | MIA Orig AUC={auc_orig:.4f} Unl AUC={auc_unl:.4f}"
        )

        unlearn_accs.append(acc_un)
        unlearn_f1s.append(f1_un)
        orig_aucs.append(auc_orig)
        unl_aucs.append(auc_unl)

    # 7) 汇总
    print("\n===== Summary Across Runs =====")
    print(f"Mean UNLEARNED Acc: {np.mean(unlearn_accs):.4f} ± {np.std(unlearn_accs):.4f}")
    print(f"Mean UNLEARNED F1 : {np.mean(unlearn_f1s):.4f} ± {np.std(unlearn_f1s):.4f}")
    print(f"Mean MIA Orig AUC: {np.mean(orig_aucs):.4f} ± {np.std(orig_aucs):.4f}")
    print(f"Mean MIA Unl  AUC: {np.mean(unl_aucs):.4f} ± {np.std(unl_aucs):.4f}")

if __name__ == "__main__":
    main()
