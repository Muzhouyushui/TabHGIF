# run_ft_row_zero.py
import os
import time
import copy
import numpy as np
import torch

from ft_config import get_args
from ft_data import preprocess_node_features, generate_hyperedge_list
from ft_train import set_seed, train_full, finetune_steps, freeze_all_but_head
from ft_eval import eval_split, build_masks, member_mask_from_retain, member_mask_from_deleted

# 你已有：HyperGCN.py（同目录）
from HyperGCN import HyperGCN, laplacian

# 你提供的 MIA 实现：MIA_HGCN.py（同目录）
from MIA_HGCN import membership_inference_hgcn


def _device(args):
    if args.cuda and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _new_model(X_np, hyperedges, args, device):
    # 按你 MIA_HGCN.py 的 cfg 习惯构造一个对象
    cfg = lambda: None
    cfg.d = X_np.shape[1]
    cfg.c = 2  # Adult binary
    cfg.depth = args.depth
    cfg.dropout = args.dropout
    cfg.fast = args.fast
    cfg.mediators = args.mediators
    cfg.cuda = args.cuda
    cfg.dataset = "adult"

    model = HyperGCN(
        num_nodes=X_np.shape[0],
        edge_list=hyperedges,
        X_init=X_np,
        args=cfg
    ).to(device)

    # 结构（Laplacian）放 GPU
    A = laplacian(hyperedges, X_np, args.mediators).to(device)
    model.structure = A
    for l in model.layers:
        l.reapproximate = False
    return model, A


def _clone_model(model):
    return copy.deepcopy(model)


def _remove_nodes_from_hyperedges(hyperedges, deleted_nodes):
    """
    Edited hypergraph（行删除后）：
    - 保持节点编号不变（仍是 0..N-1）
    - 仅在“结构上”把 deleted 节点从每条超边里剔除
    - 超边长度 < 2 的直接丢弃（否则对 Laplacian/GCN 没意义）
    """
    del_set = set(int(x) for x in deleted_nodes)
    new_edges = []
    for e in hyperedges:
        # e 通常是 list[int]
        e2 = [v for v in e if int(v) not in del_set]
        if len(e2) >= 2:
            new_edges.append(e2)
    return new_edges


def run_one(args, run_id: int):
    device = _device(args)
    print(f"[Device] {device}")
    seed = args.seed + run_id
    set_seed(seed)

    # ===== Load train/test =====
    X_tr, y_tr_np, df_tr, transformer = preprocess_node_features(
        args.train_csv, is_test=False,
        categ_cols=args.categate_cols,
        label_col=args.label_col,
        transformer=None,
        filter_missing_q=args.filter_missing_q
    )
    X_te, y_te_np, df_te, _ = preprocess_node_features(
        args.test_csv, is_test=True,
        categ_cols=args.categate_cols,
        label_col=args.label_col,
        transformer=transformer,
        filter_missing_q=args.filter_missing_q
    )

    # ===== Hyperedges (from raw df) =====
    hedges_tr = generate_hyperedge_list(
        df_tr, args.categate_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        seed=seed
    )
    hedges_te = generate_hyperedge_list(
        df_te, args.categate_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge,
        seed=seed
    )

    # ===== Torch tensors =====
    fts_tr = torch.from_numpy(X_tr).float().to(device)
    lbls_tr = torch.from_numpy(y_tr_np).long().to(device)

    fts_te = torch.from_numpy(X_te).float().to(device)
    lbls_te = torch.from_numpy(y_te_np).long().to(device)

    # ===== Build model + structures (ORIGINAL train graph) =====
    model_full, A_tr = _new_model(X_tr, hedges_tr, args, device)
    A_te = laplacian(hedges_te, X_te, args.mediators).to(device)

    # ===== Train full =====
    print("== Train Full Model ==")
    t0 = time.time()
    model_full = train_full(
        model_full, fts_tr, lbls_tr,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        milestones=args.milestones,
        gamma=args.gamma
    )
    full_train_time = time.time() - t0

    f1_test, acc_test = eval_split(model_full, fts_te, lbls_te, A_te, mask=None)
    print(f"[Full] Test ACC={acc_test:.4f}, micro-F1={f1_test:.4f} | train_time={full_train_time:.2f}s")

    # ===== Sample deleted rows =====
    N = X_tr.shape[0]
    n_del = int(N * args.remove_ratio)
    rng = np.random.default_rng(seed)
    deleted_idx = rng.choice(np.arange(N), size=n_del, replace=False)
    deleted = torch.tensor(deleted_idx, dtype=torch.long, device=device)
    print(f"[Delete] remove_ratio={args.remove_ratio}, deleted={len(deleted)}")

    retain_mask, del_mask = build_masks(N, deleted, device)

    # ===== Feature-zero edited training features =====
    fts_tr_after = fts_tr.clone()
    fts_tr_after[deleted] = 0.0

    # ============================================================
    # ✅ 新增统计：构造 edited hypergraph + edited Laplacian 的时间
    # ============================================================
    t_edit0 = time.time()
    deleted_cpu = deleted.detach().cpu().tolist()
    hedges_tr_edit = _remove_nodes_from_hyperedges(hedges_tr, deleted_cpu)
    # 注意：这里 laplacian 需要 X_np（numpy），与原脚本一致
    A_tr_edit = laplacian(hedges_tr_edit, X_tr, args.mediators).to(device)
    edit_time_sec = time.time() - t_edit0

    print(f"[Edited HG] #edges(original)={len(hedges_tr)} -> #edges(edited)={len(hedges_tr_edit)} "
          f"| edit_time={edit_time_sec:.4f}s")

    # ===== Evaluate forget/retain BEFORE FT =====
    # Full 模型本身没更新，但你可以用 edited graph 来观测“结构变化下”的 retain/forget
    f1_ret, acc_ret = eval_split(model_full, fts_tr, lbls_tr, A_tr_edit, mask=retain_mask)
    f1_for, acc_for = eval_split(model_full, fts_tr, lbls_tr, A_tr_edit, mask=del_mask)
    print(f"[Full@EditedHG] Retain ACC={acc_ret:.4f} F1={f1_ret:.4f} | Forget ACC={acc_for:.4f} F1={f1_for:.4f}")

    results = []

    def _maybe_mia(tag, model, A_used, hedges_used):
        mia_overall_auc = None
        mia_deleted_auc = None

        if not args.run_mia:
            return mia_overall_auc, mia_deleted_auc

        # 在跑 MIA 前，把 target_model 的 structure 设为“对应的训练图结构”
        model.structure = A_used
        for l in model.layers:
            l.reapproximate = False

        # 1) overall（retained=1, deleted=0）
        mm_retain = member_mask_from_retain(retain_mask)  # int 0/1, len=N
        print(f"— MIA on {tag} (overall/retain-membership) —")
        _, _, tgt = membership_inference_hgcn(
            X_train=X_tr,
            y_train=y_tr_np,
            hyperedges=hedges_used,
            target_model=model,
            args=args,
            device=device,
            member_mask=mm_retain
        )
        if tgt is not None:
            mia_overall_auc = float(tgt[0])

        # 2) deleted-only（deleted=1, others=0）
        mm_deleted = member_mask_from_deleted(deleted, N)
        print(f"— MIA on {tag} (deleted-only membership) —")
        _, _, tgt2 = membership_inference_hgcn(
            X_train=X_tr,
            y_train=y_tr_np,
            hyperedges=hedges_used,
            target_model=model,
            args=args,
            device=device,
            member_mask=mm_deleted
        )
        if tgt2 is not None:
            mia_deleted_auc = float(tgt2[0])

        return mia_overall_auc, mia_deleted_auc

    # ===== Record Full (still original training, but evaluate under edited HG) =====
    mia_o, mia_d = _maybe_mia("Full@EditedHG", model_full, A_tr_edit, hedges_tr_edit)
    results.append({
        "run": run_id,
        "method": "Full@EditedHG",
        "K": 8,
        "time_sec": full_train_time,                 # 原定义：full training time
        "edit_time_sec": edit_time_sec,              # 新增：edited HG 构造时间
        "total_time_sec": full_train_time + edit_time_sec,
        "test_acc": acc_test,
        "retain_acc": acc_ret,
        "forget_acc": acc_for,
        "mia_overall_auc": mia_o,
        "mia_deleted_auc": mia_d,
        "seed": seed
    })

    # ===== FT-K =====
    print("\n== FT-K (warm-start) on Edited Hypergraph ==")
    for K in args.ft_steps:
        m = _clone_model(model_full)

        # ✅ 关键：训练时结构用 edited graph
        m.structure = A_tr_edit
        for l in m.layers:
            l.reapproximate = False

        t1 = time.time()
        m = finetune_steps(
            m,
            x_after=fts_tr_after,
            y=lbls_tr,
            retain_mask=retain_mask,
            steps=K,
            lr=args.ft_lr,
            weight_decay=args.ft_wd
        )
        ft_time = time.time() - t1

        f1_test_k, acc_test_k = eval_split(m, fts_te, lbls_te, A_te, mask=None)
        f1_ret_k, acc_ret_k = eval_split(m, fts_tr, lbls_tr, A_tr_edit, mask=retain_mask)
        f1_for_k, acc_for_k = eval_split(m, fts_tr, lbls_tr, A_tr_edit, mask=del_mask)

        print(f"[FT-K@EditedHG] K={K:4d} | Test ACC={acc_test_k:.4f} F1={f1_test_k:.4f} "
              f"| Retain ACC={acc_ret_k:.4f} | Forget ACC={acc_for_k:.4f} | "
              f"ft_time={ft_time:.4f}s | total={ft_time + edit_time_sec:.4f}s")

        mia_o, mia_d = _maybe_mia(f"FT-K@EditedHG(K={K})", m, A_tr_edit, hedges_tr_edit)
        results.append({
            "run": run_id,
            "method": "FT-K@EditedHG",
            "K": K,
            "time_sec": ft_time,                      # 原定义：finetune time
            "edit_time_sec": edit_time_sec,           # 新增：同一次 run 的 edited HG 构造时间
            "total_time_sec": ft_time + edit_time_sec,
            "test_acc": acc_test_k,
            "retain_acc": acc_ret_k,
            "forget_acc": acc_for_k,
            "mia_overall_auc": mia_o,
            "mia_deleted_auc": mia_d,
            "seed": seed
        })

    # ===== FT-head =====
    print("\n== FT-head (only train last layer) on Edited Hypergraph ==")
    for K in args.ft_steps:
        m = _clone_model(model_full)
        freeze_all_but_head(m)

        # ✅ 关键：训练时结构用 edited graph
        m.structure = A_tr_edit
        for l in m.layers:
            l.reapproximate = False

        t1 = time.time()
        m = finetune_steps(
            m,
            x_after=fts_tr_after,
            y=lbls_tr,
            retain_mask=retain_mask,
            steps=K,
            lr=args.ft_lr,
            weight_decay=args.ft_wd
        )
        ft_time = time.time() - t1

        f1_test_k, acc_test_k = eval_split(m, fts_te, lbls_te, A_te, mask=None)
        f1_ret_k, acc_ret_k = eval_split(m, fts_tr, lbls_tr, A_tr_edit, mask=retain_mask)
        f1_for_k, acc_for_k = eval_split(m, fts_tr, lbls_tr, A_tr_edit, mask=del_mask)

        print(f"[FT-head@EditedHG] K={K:4d} | Test ACC={acc_test_k:.4f} F1={f1_test_k:.4f} "
              f"| Retain ACC={acc_ret_k:.4f} | Forget ACC={acc_for_k:.4f} | "
              f"ft_time={ft_time:.4f}s | total={ft_time + edit_time_sec:.4f}s")

        mia_o, mia_d = _maybe_mia(f"FT-head@EditedHG(K={K})", m, A_tr_edit, hedges_tr_edit)
        results.append({
            "run": run_id,
            "method": "FT-head@EditedHG",
            "K": K,
            "time_sec": ft_time,                      # 原定义：finetune time
            "edit_time_sec": edit_time_sec,
            "total_time_sec": ft_time + edit_time_sec,
            "test_acc": acc_test_k,
            "retain_acc": acc_ret_k,
            "forget_acc": acc_for_k,
            "mia_overall_auc": mia_o,
            "mia_deleted_auc": mia_d,
            "seed": seed
        })

    print("\nDone.\n")
    return results


def summarize_and_save(all_rows, out_csv):
    import pandas as pd

    df = pd.DataFrame(all_rows)
    df.to_csv(out_csv, index=False)

    # 汇总 mean±std（按 method,K）
    agg_cols = [
        "edit_time_sec", "time_sec", "total_time_sec",
        "test_acc", "retain_acc", "forget_acc",
        "mia_overall_auc", "mia_deleted_auc"
    ]
    g = df.groupby(["method", "K"], dropna=False)[agg_cols].agg(["mean", "std"]).reset_index()

    # 打印简表
    print("== Summary (mean±std) ==")
    for _, r in g.iterrows():
        method = r[("method", "")]
        K = r[("K", "")]

        def fmt(col):
            m = r[(col, "mean")]
            s = r[(col, "std")]
            if pd.isna(m):
                return "NA"
            if pd.isna(s):
                return f"{m:.4f}"
            return f"{m:.4f}±{s:.4f}"

        print(
            f"{method:16s} K={int(K):4d} | "
            f"edit={fmt('edit_time_sec')} | "
            f"update={fmt('time_sec')} | "
            f"total={fmt('total_time_sec')} | "
            f"test_acc={fmt('test_acc')} | retain_acc={fmt('retain_acc')} | forget_acc={fmt('forget_acc')} | "
            f"mia_overall={fmt('mia_overall_auc')} | mia_deleted={fmt('mia_deleted_auc')}"
        )

    print(f"\n[Saved] {out_csv}")


def main():
    args = get_args()

    if not os.path.exists(args.train_csv):
        print(f"[WARN] train_csv not found: {args.train_csv}")
    if not os.path.exists(args.test_csv):
        print(f"[WARN] test_csv not found: {args.test_csv}")

    all_rows = []
    for run_id in range(args.runs):
        print(f"\n================= RUN {run_id + 1}/{args.runs} =================")
        rows = run_one(args, run_id)
        all_rows.extend(rows)

    summarize_and_save(all_rows, args.out_csv)


if __name__ == "__main__":
    main()
