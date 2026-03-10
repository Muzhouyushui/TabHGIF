# -*- coding: utf-8 -*-
"""
run_ft_hgcn_col_zero.py
列级删除（Column Unlearning）下的 HGCN 微调基线实验（FT-K / FT-head）
- 仿照 HGCN_Unlearning_col.py 的训练与删列流程
- 在 Edited Hypergraph 上进行 practical fine-tuning baseline 对比
"""

import time
import copy
import csv
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ====== 你的项目依赖（按你现有工程路径）======
from config_HGCN import get_args as get_base_args
from database.data_preprocessing.data_preprocessing_column import (
    preprocess_node_features,
    generate_hyperedge_dict,
    delete_feature_columns_hgcn,
)
from utils.common_utils import evaluate_hgcn_acc, evaluate_hgcn_f1
from GIF.GIF_HGCN_COL import train_model
from HGCN.HyperGCN import HyperGCN, laplacian


# =========================================================
#                     参数补齐（不改原 config）
# =========================================================
def _ensure_attr(args, name, value):
    if not hasattr(args, name):
        setattr(args, name, value)


def get_args():
    """
    复用你原 config_HGCN.get_args()，并补充 FT 实验需要的参数。
    """
    args = get_base_args()

    # ===== FT baseline args =====
    _ensure_attr(args, "runs", 1)
    _ensure_attr(args, "ft_steps", [50, 100, 200])   # list[int]
    _ensure_attr(args, "ft_lr", 1e-3)
    _ensure_attr(args, "ft_wd", 0.0)
    _ensure_attr(args, "ft_milestones", [])          # 可选，如 [100]
    _ensure_attr(args, "ft_gamma", 0.1)
    _ensure_attr(args, "run_mia", False)
    _ensure_attr(args, "out_csv", "ft_hgcn_col_results.csv")

    # 列删除设置（兼容 string / list）
    _ensure_attr(args, "columns_to_unlearn", "age")

    # 日志频率（兼容你的 train_model）
    _ensure_attr(args, "log_every", 10)

    # 训练轮数（若原 config 无）
    _ensure_attr(args, "epochs", 100)

    return args


# =========================================================
#                        工具函数
# =========================================================
def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _set_model_structure(model, A):
    """同步结构矩阵，并关闭 layer 内部重近似。"""
    model.structure = A
    if hasattr(model, "layers"):
        for layer in model.layers:
            if hasattr(layer, "reapproximate"):
                layer.reapproximate = False


def _forward_hgcn(model, x):
    """
    尝试兼容 HyperGCN 的前向返回形式。
    常见情况：model(x) -> log_probs
    """
    out = model(x)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out


def _clone_model(model: nn.Module) -> nn.Module:
    return copy.deepcopy(model)


def _get_last_trainable_module(model: nn.Module):
    """
    尝试找“最后一层”用于 FT-head。
    优先：classifier / final / out_layer / 最后一个 Linear
    """
    for name in ["classifier", "final", "out_layer", "fc", "linear"]:
        if hasattr(model, name):
            mod = getattr(model, name)
            if isinstance(mod, nn.Module):
                return mod

    # fallback: 找最后一个 nn.Linear
    last_linear = None
    for _, m in model.named_modules():
        if isinstance(m, nn.Linear):
            last_linear = m
    if last_linear is not None:
        return last_linear

    # fallback: 最后一个子模块（不太理想，但至少可跑）
    children = [m for m in model.children()]
    if len(children) > 0:
        return children[-1]

    return None


def _freeze_all_then_unfreeze(module_model: nn.Module, target_module: nn.Module):
    for p in module_model.parameters():
        p.requires_grad = False
    if target_module is not None:
        for p in target_module.parameters():
            p.requires_grad = True


def _build_hgcn_model(X_tr_np, lbls_tensor, hyperedges_tr_list, args, device):
    """
    仿照你 HGCN_Unlearning_col.py 的模型构建方式。
    """
    cfg = lambda: None
    cfg.d = X_tr_np.shape[1]
    cfg.depth = args.depth
    cfg.c = int(lbls_tensor.max().item() + 1)
    cfg.dropout = args.dropout
    cfg.fast = args.fast
    cfg.mediators = args.mediators
    cfg.cuda = args.cuda
    cfg.dataset = getattr(args, "dataset", "ACI")

    model = HyperGCN(
        num_nodes=X_tr_np.shape[0],
        edge_list=hyperedges_tr_list,
        X_init=X_tr_np,
        args=cfg
    ).to(device)
    return model


def _train_full_model_hgcn(model, fts_tr, lbls_tr, args):
    """
    复用你 GIF.GIF_HGCN_COL 里的 train_model（保持训练口径一致）。
    """
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.milestones, gamma=args.gamma
    )
    criterion = nn.NLLLoss()

    model = train_model(
        model,
        criterion,
        optimizer,
        scheduler,
        fts_tr,
        lbls_tr,
        num_epochs=args.epochs,
        print_freq=args.log_every,
    )
    return model


def _finetune_steps_hgcn(
    model,
    x_train,
    y_train,
    K: int,
    lr: float,
    weight_decay: float,
    milestones=None,
    gamma=0.1,
    print_prefix="[FT]",
):
    """
    在编辑后的超图上进行 K 步微调（全参数或已冻结后的 head-only 都复用此函数）
    """
    model.train()
    criterion = nn.NLLLoss()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = None
    if milestones is not None and len(milestones) > 0:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    t0 = time.time()
    for step in range(1, K + 1):
        optimizer.zero_grad()
        out = _forward_hgcn(model, x_train)   # model.structure 已提前设置为 A_after
        loss = criterion(out, y_train)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

    update_time = time.time() - t0
    return model, update_time


def _safe_eval_acc(model, x, y):
    return evaluate_hgcn_acc(model, {"x": x, "y": y})


def _safe_eval_f1(model, x, y):
    return evaluate_hgcn_f1(model, {"x": x, "y": y})


def _maybe_mia_hgcn(model, args, x_train, y_train, x_test, y_test):
    """
    占位 / 兼容函数：
    - 若你项目里已有 MIA_HGCN.py 且接口稳定，可在这里接入
    - 目前默认返回 None，避免因为接口差异导致脚本崩
    """
    if not getattr(args, "run_mia", False):
        return None, None

    # 你可以在这里替换为自己的实际 MIA 调用，例如：
    # from MIA_HGCN import membership_inference
    # ...
    # return float(auc_target), float(f1_target)
    try:
        # ===== 示例占位（请按你本地接口替换）=====
        # from MIA_HGCN import membership_inference
        # _, _, (auc_target, f1_target) = membership_inference(...)
        # return float(auc_target), float(f1_target)
        print("[MIA] run_mia=True, but _maybe_mia_hgcn() is placeholder. Returning NA.")
        return None, None
    except Exception as e:
        print(f"[MIA] failed: {e}")
        return None, None


def _fmt_num(x):
    if x is None:
        return "NA"
    return f"{x:.4f}"


def _save_rows_csv(rows, out_csv):
    if len(rows) == 0:
        return
    keys = list(rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(rows):
    """
    仅做 mean±std 汇总（按 method + K 分组）
    """
    groups = defaultdict(list)
    for r in rows:
        groups[(r["method"], int(r["K"]))].append(r)

    print("\n== Summary (mean±std) ==")
    ordered = sorted(groups.keys(), key=lambda x: (x[0], x[1]))
    for key in ordered:
        method, K = key
        vals = groups[key]

        def mstd(field):
            arr = np.array([float(v[field]) for v in vals if v[field] is not None], dtype=float)
            if len(arr) == 0:
                return None, None
            return arr.mean(), arr.std(ddof=0)

        e_m, e_s = mstd("edit")
        u_m, u_s = mstd("update")
        t_m, t_s = mstd("total")
        acc_m, acc_s = mstd("test_acc")
        mia_m, mia_s = mstd("mia_overall")

        if "FT-head" in method:
            pad = ""  # 你前面的格式喜欢对齐，我这里简单点
        else:
            pad = " "

        line = (
            f"{method}{pad} K={K:4d} | "
            f"edit={_fmt_num(e_m)}" + (f"±{e_s:.4f}" if e_s is not None else "") + " | "
            f"update={_fmt_num(u_m)}" + (f"±{u_s:.4f}" if u_s is not None else "") + " | "
            f"total={_fmt_num(t_m)}" + (f"±{t_s:.4f}" if t_s is not None else "") + " | "
            f"test_acc={_fmt_num(acc_m)}" + (f"±{acc_s:.4f}" if acc_s is not None else "") + " | "
            f"mia_overall={_fmt_num(mia_m)}" + (f"±{mia_s:.4f}" if mia_s is not None else "")
        )
        print(line)


# =========================================================
#                  单次运行（一个 seed / run）
# =========================================================
def run_one(args, run_id: int):
    seed = int(getattr(args, "seed", 1)) + run_id
    seed_everything(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n================= RUN {run_id + 1}/{args.runs} =================")
    print(f"[Device] {device} | seed={seed}")

    # -------------------------------
    # 1) 训练集预处理（原始）
    # -------------------------------
    X_tr, y_tr, df_tr, transformer = preprocess_node_features(
        args.train_csv, is_test=False
    )
    hyperedges_tr_dict = generate_hyperedge_dict(
        df_tr,
        args.categate_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        device=device,
    )
    hyperedges_tr_list = list(hyperedges_tr_dict.values())

    fts_tr = torch.from_numpy(X_tr).float().to(device)
    lbls_tr = torch.tensor(y_tr, dtype=torch.long).to(device)

    # -------------------------------
    # 2) 构建模型 + 原图 Laplacian
    # -------------------------------
    model = _build_hgcn_model(X_tr, lbls_tr, hyperedges_tr_list, args, device)

    A_before = laplacian(hyperedges_tr_list, X_tr, args.mediators).to(device)
    _set_model_structure(model, A_before)

    # -------------------------------
    # 3) 训练 full model（原始列）
    # -------------------------------
    print("== Train Full Model ==")
    t_train0 = time.time()
    model = _train_full_model_hgcn(model, fts_tr, lbls_tr, args)
    train_time = time.time() - t_train0
    print(f"[Full] train_time={train_time:.2f}s")

    # -------------------------------
    # 4) 测试集（原始列）
    # -------------------------------
    X_te, y_te, df_te, _ = preprocess_node_features(
        args.test_csv, is_test=True, transformer=transformer
    )
    fts_te = torch.from_numpy(X_te).float().to(device)
    lbls_te = torch.tensor(y_te, dtype=torch.long).to(device)

    hyperedges_te_dict = generate_hyperedge_dict(
        df_te,
        args.categate_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        device=device,
    )
    hyperedges_te_list = list(hyperedges_te_dict.values())
    A_te_before = laplacian(hyperedges_te_list, X_te, args.mediators).to(device)

    _set_model_structure(model, A_te_before)
    test_acc_before = _safe_eval_acc(model, fts_te, lbls_te)
    test_f1_before = _safe_eval_f1(model, fts_te, lbls_te)
    print(f"[Before Col-Unlearning] Test ACC={test_acc_before:.4f} | Test F1={test_f1_before:.4f}")

    # -------------------------------
    # 5) 训练集列删除（构造 EditedHG）
    # -------------------------------
    column_to_delete = args.columns_to_unlearn
    if isinstance(column_to_delete, str):
        # 兼容 "education" 或 "education,occupation" 形式
        if "," in column_to_delete:
            column_to_delete = [c.strip() for c in column_to_delete.split(",") if c.strip()]
        else:
            column_to_delete = [column_to_delete]

    print(f"[Column Unlearning] columns_to_unlearn={column_to_delete}")

    t_edit0 = time.time()
    x_tr_after, hyperedges_tr_after_dict, A_tr_after = delete_feature_columns_hgcn(
        X_tensor=fts_tr,
        transformer=transformer,
        column_names=column_to_delete,
        hyperedges=hyperedges_tr_dict,   # dict（与你原脚本一致）
        mediators=args.mediators,
        use_cuda=(device.type == "cuda"),
    )
    edit_time = time.time() - t_edit0

    # -------------------------------
    # 6) 测试集列删除（用于评估）
    # -------------------------------
    x_te_after, hyperedges_te_after_dict, A_te_after = delete_feature_columns_hgcn(
        X_tensor=fts_te,
        transformer=transformer,
        column_names=column_to_delete,
        hyperedges=hyperedges_te_dict,
        mediators=args.mediators,
        use_cuda=(device.type == "cuda"),
    )

    # -------------------------------
    # 7) Full@EditedHG（不更新参数）
    # -------------------------------
    rows = []

    full_edit_model = _clone_model(model)
    _set_model_structure(full_edit_model, A_te_after)
    full_test_acc = _safe_eval_acc(full_edit_model, x_te_after, lbls_te)
    full_test_f1 = _safe_eval_f1(full_edit_model, x_te_after, lbls_te)
    mia_auc, mia_f1 = _maybe_mia_hgcn(full_edit_model, args, x_tr_after, lbls_tr, x_te_after, lbls_te)

    print(f"[Full@EditedHG] Test ACC={full_test_acc:.4f} | Test F1={full_test_f1:.4f} | edit={edit_time:.4f}s")

    rows.append({
        "run_id": run_id + 1,
        "method": "Full@EditedHG",
        "K": 0,
        "edit": float(edit_time),
        "update": 0.0,
        "total": float(edit_time),
        "test_acc": float(full_test_acc),
        "test_f1": float(full_test_f1),
        "mia_overall": None if mia_auc is None else float(mia_auc),
        "mia_aux": None if mia_f1 is None else float(mia_f1),  # 注意：默认是F1占位，不是deleted-AUC
    })

    # -------------------------------
    # 8) FT-K（全参数微调）
    # -------------------------------
    print("\n== FT-K (warm-start on EditedHG) ==")
    for K in args.ft_steps:
        m_ft = _clone_model(model)

        # 训练时结构用编辑后的训练图
        _set_model_structure(m_ft, A_tr_after)
        # 保证全部参数可训练
        for p in m_ft.parameters():
            p.requires_grad = True

        m_ft, update_time = _finetune_steps_hgcn(
            m_ft,
            x_train=x_tr_after,
            y_train=lbls_tr,
            K=int(K),
            lr=float(args.ft_lr),
            weight_decay=float(args.ft_wd),
            milestones=list(getattr(args, "ft_milestones", [])),
            gamma=float(getattr(args, "ft_gamma", 0.1)),
            print_prefix="[FT-K]",
        )

        # 测试时结构切到编辑后的测试图
        _set_model_structure(m_ft, A_te_after)
        test_acc = _safe_eval_acc(m_ft, x_te_after, lbls_te)
        test_f1 = _safe_eval_f1(m_ft, x_te_after, lbls_te)
        mia_auc, mia_f1 = _maybe_mia_hgcn(m_ft, args, x_tr_after, lbls_tr, x_te_after, lbls_te)

        total_time = edit_time + update_time
        print(
            f"[FT-K] K={K:4d} | Test ACC={test_acc:.4f} | Test F1={test_f1:.4f} | "
            f"edit={edit_time:.4f}s | update={update_time:.4f}s | total={total_time:.4f}s"
        )

        rows.append({
            "run_id": run_id + 1,
            "method": "FT-K@EditedHG",
            "K": int(K),
            "edit": float(edit_time),
            "update": float(update_time),
            "total": float(total_time),
            "test_acc": float(test_acc),
            "test_f1": float(test_f1),
            "mia_overall": None if mia_auc is None else float(mia_auc),
            "mia_aux": None if mia_f1 is None else float(mia_f1),
        })

    # -------------------------------
    # 9) FT-head（只训练最后层）
    # -------------------------------
    print("\n== FT-head (only train last layer on EditedHG) ==")
    for K in args.ft_steps:
        m_head = _clone_model(model)

        # 冻结除最后层外所有参数
        head_module = _get_last_trainable_module(m_head)
        _freeze_all_then_unfreeze(m_head, head_module)

        _set_model_structure(m_head, A_tr_after)

        m_head, update_time = _finetune_steps_hgcn(
            m_head,
            x_train=x_tr_after,
            y_train=lbls_tr,
            K=int(K),
            lr=float(args.ft_lr),
            weight_decay=float(args.ft_wd),
            milestones=list(getattr(args, "ft_milestones", [])),
            gamma=float(getattr(args, "ft_gamma", 0.1)),
            print_prefix="[FT-head]",
        )

        _set_model_structure(m_head, A_te_after)
        test_acc = _safe_eval_acc(m_head, x_te_after, lbls_te)
        test_f1 = _safe_eval_f1(m_head, x_te_after, lbls_te)
        mia_auc, mia_f1 = _maybe_mia_hgcn(m_head, args, x_tr_after, lbls_tr, x_te_after, lbls_te)

        total_time = edit_time + update_time
        print(
            f"[FT-head] K={K:4d} | Test ACC={test_acc:.4f} | Test F1={test_f1:.4f} | "
            f"edit={edit_time:.4f}s | update={update_time:.4f}s | total={total_time:.4f}s"
        )

        rows.append({
            "run_id": run_id + 1,
            "method": "FT-head@EditedHG",
            "K": int(K),
            "edit": float(edit_time),
            "update": float(update_time),
            "total": float(total_time),
            "test_acc": float(test_acc),
            "test_f1": float(test_f1),
            "mia_overall": None if mia_auc is None else float(mia_auc),
            "mia_aux": None if mia_f1 is None else float(mia_f1),
        })

    return rows


# =========================================================
#                           主函数
# =========================================================
def main():
    args = get_args()

    # 打印关键配置（方便你排错）
    print("==== FT HGCN Column Baseline Config ====")
    for k in [
        "train_csv", "test_csv", "columns_to_unlearn", "categate_cols",
        "max_nodes_per_hyperedge", "epochs", "lr", "weight_decay",
        "ft_steps", "ft_lr", "ft_wd", "runs", "run_mia", "out_csv"
    ]:
        if hasattr(args, k):
            print(f"{k}: {getattr(args, k)}")
    print("=======================================")

    all_rows = []
    for run_id in range(int(args.runs)):
        rows = run_one(args, run_id)
        all_rows.extend(rows)

    # 保存逐次结果
    _save_rows_csv(all_rows, args.out_csv)
    print(f"\n[Saved] {args.out_csv}")

    # 打印汇总
    _print_summary(all_rows)


if __name__ == "__main__":
    main()