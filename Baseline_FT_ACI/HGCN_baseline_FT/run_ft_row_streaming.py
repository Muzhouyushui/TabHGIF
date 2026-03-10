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


# -------------------------
# Helpers
# -------------------------
def _device(args):
    if getattr(args, "cuda", False) and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _new_model(X_np, hyperedges, args, device):
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
    - 超边长度 < 2 的直接丢弃
    """
    del_set = set(int(x) for x in deleted_nodes)
    new_edges = []
    for e in hyperedges:
        e2 = [v for v in e if int(v) not in del_set]
        if len(e2) >= 2:
            new_edges.append(e2)
    return new_edges


def _maybe_mia(args, tag, model, A_used, hedges_used, device, X_tr, y_tr_np, retain_mask, deleted_tensor, N):
    """
    你说 overall=target；这里保留：
    - mia_overall: retain-membership（retained=1, deleted=0）
    - mia_deleted: deleted-only membership（deleted=1, others=0）
    """
    mia_overall_auc = None
    mia_deleted_auc = None

    if not getattr(args, "run_mia", False):
        return mia_overall_auc, mia_deleted_auc

    model.structure = A_used
    for l in model.layers:
        l.reapproximate = False

    # overall
    mm_retain = member_mask_from_retain(retain_mask)
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

    # deleted-only
    mm_deleted = member_mask_from_deleted(deleted_tensor, N)
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


# -------------------------
# Streaming Core
# -------------------------
def run_streaming_one(args, run_id: int, method: str):
    """
    method:
      - "FT-head@EditedHG"
      - "FT-K@EditedHG"
    """
    assert method in ["FT-head@EditedHG", "FT-K@EditedHG"]

    device = _device(args)
    print(f"[Device] {device}")
    seed = args.seed + run_id
    set_seed(seed)

    # streaming params (给默认值，避免你必须改 ft_config.py)
    T = int(getattr(args, "stream_rounds", 50))
    per_round_ratio = float(getattr(args, "per_round_ratio", 0.01))   # 1%
    per_round_n = int(getattr(args, "per_round_n", 0))                # 若>0 则覆盖 ratio
    ft_steps = getattr(args, "stream_ft_steps", None)
    if ft_steps is None:
        # 默认：每轮固定 50 steps（最强对手）
        ft_steps = [int(getattr(args, "stream_K", 50))]

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

    N = X_tr.shape[0]

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

    # ===== Streaming init =====
    rng = np.random.default_rng(seed)
    all_idx = np.arange(N)
    deleted_set = set()
    results = []

    # streaming model starts from full
    stream_model = _clone_model(model_full)
    if method == "FT-head@EditedHG":
        freeze_all_but_head(stream_model)

    cum_time = 0.0

    # ----- precompute a stable pool (exclude deleted) each round -----
    def sample_batch():
        remaining = np.array([i for i in all_idx if i not in deleted_set], dtype=np.int64)
        if len(remaining) == 0:
            return np.array([], dtype=np.int64)

        if per_round_n > 0:
            k = min(per_round_n, len(remaining))
        else:
            k = int(np.ceil(N * per_round_ratio))
            k = max(1, min(k, len(remaining)))

        return rng.choice(remaining, size=k, replace=False)

    print(f"== Streaming FT ({method}) == rounds={T}, per_round_ratio={per_round_ratio}, per_round_n={per_round_n}, ft_steps={ft_steps}")

    # ===== Streaming loop =====
    for t in range(1, T + 1):
        batch = sample_batch()
        if batch.size == 0:
            print("[Stream] No remaining nodes to delete. Stop.")
            break

        for i in batch.tolist():
            deleted_set.add(int(i))

        deleted_idx = np.array(sorted(list(deleted_set)), dtype=np.int64)
        deleted = torch.tensor(deleted_idx, dtype=torch.long, device=device)

        retain_mask, del_mask = build_masks(N, deleted, device)

        # features after deletion (zero out ALL deleted so far)
        fts_tr_after = fts_tr.clone()
        fts_tr_after[deleted] = 0.0

        # build edited hypergraph from cumulative deleted
        hedges_tr_edit = _remove_nodes_from_hyperedges(hedges_tr, deleted_idx.tolist())
        A_tr_edit = laplacian(hedges_tr_edit, X_tr, args.mediators).to(device)

        # Evaluate BEFORE this round update (under edited graph)
        f1_b_te, acc_b_te = eval_split(stream_model, fts_te, lbls_te, A_te, mask=None)
        f1_b_ret, acc_b_ret = eval_split(stream_model, fts_tr, lbls_tr, A_tr_edit, mask=retain_mask)
        f1_b_for, acc_b_for = eval_split(stream_model, fts_tr, lbls_tr, A_tr_edit, mask=del_mask)

        # apply FT with fixed budget(s); 这里默认每轮只取一个 budget（比如 50）
        # 如果你给了多个 ft_steps，会把每个 budget 都跑一遍（会更慢），一般 streaming 只用一个 budget
        K = int(ft_steps[0])

        # update model structure to edited graph
        stream_model.structure = A_tr_edit
        for l in stream_model.layers:
            l.reapproximate = False

        t1 = time.time()
        stream_model = finetune_steps(
            stream_model,
            x_after=fts_tr_after,
            y=lbls_tr,
            retain_mask=retain_mask,
            steps=K,
            lr=args.ft_lr,
            weight_decay=args.ft_wd
        )
        un_time = time.time() - t1
        cum_time += un_time

        # Evaluate AFTER this round update
        f1_a_te, acc_a_te = eval_split(stream_model, fts_te, lbls_te, A_te, mask=None)
        f1_a_ret, acc_a_ret = eval_split(stream_model, fts_tr, lbls_tr, A_tr_edit, mask=retain_mask)
        f1_a_for, acc_a_for = eval_split(stream_model, fts_tr, lbls_tr, A_tr_edit, mask=del_mask)

        mia_o, mia_d = _maybe_mia(
            args=args,
            tag=f"{method}(round={t},K={K})",
            model=stream_model,
            A_used=A_tr_edit,
            hedges_used=hedges_tr_edit,
            device=device,
            X_tr=X_tr,
            y_tr_np=y_tr_np,
            retain_mask=retain_mask,
            deleted_tensor=deleted,
            N=N
        )

        print(
            f"[Round {t:02d}] cum_deleted={len(deleted_idx)}/{N} ({len(deleted_idx)/N:.2%}) "
            f"| un_time={un_time:.3f}s cum_time={cum_time:.3f}s "
            f"| TestAcc {acc_b_te:.4f}->{acc_a_te:.4f} "
            f"| RetAcc {acc_b_ret:.4f}->{acc_a_ret:.4f} "
            f"| ForAcc {acc_b_for:.4f}->{acc_a_for:.4f} "
            f"| MIA {mia_o if mia_o is not None else 'NA'}"
        )

        results.append({
            "run": run_id,
            "seed": seed,
            "method": method,
            "round": t,
            "K": K,
            "deleted_num": int(len(deleted_idx)),
            "deleted_ratio": float(len(deleted_idx) / N),
            "unlearn_time_sec": float(un_time),
            "cum_time_sec": float(cum_time),

            "test_acc_before": float(acc_b_te),
            "test_acc_after": float(acc_a_te),
            "retain_acc_before": float(acc_b_ret),
            "retain_acc_after": float(acc_a_ret),
            "forget_acc_before": float(acc_b_for),
            "forget_acc_after": float(acc_a_for),

            "mia_overall_auc": mia_o,
            "mia_deleted_auc": mia_d,
            "edited_hyperedges": int(len(hedges_tr_edit)),
        })

    return results


def save_streaming(all_rows, out_csv):
    import pandas as pd
    df = pd.DataFrame(all_rows)
    df.to_csv(out_csv, index=False)
    print(f"\n[Saved] {out_csv}")

    # 打印最后一轮的 mean±std（每个方法、每个 K）
    if len(df) == 0:
        print("[WARN] empty results")
        return

    # 取每个 run 的最后一轮（round 最大）用于汇总
    last = df.sort_values(["run", "round"]).groupby(["run", "method", "K"], as_index=False).tail(1)

    agg_cols = [
        "cum_time_sec",
        "test_acc_after",
        "retain_acc_after",
        "forget_acc_after",
        "mia_overall_auc",
        "mia_deleted_auc"
    ]
    g = last.groupby(["method", "K"])[agg_cols].agg(["mean", "std"]).reset_index()

    print("== Streaming Summary (last-round mean±std) ==")
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
            f"{method:16s} K={int(K):4d} | cum_time={fmt('cum_time_sec')} "
            f"| test_acc={fmt('test_acc_after')} | retain_acc={fmt('retain_acc_after')} | forget_acc={fmt('forget_acc_after')} "
            f"| mia_overall={fmt('mia_overall_auc')} | mia_deleted={fmt('mia_deleted_auc')}"
        )


def main():
    args = get_args()

    # 默认输出文件名（避免你必须改 args）
    out_csv = getattr(args, "stream_out_csv", None)
    if out_csv is None:
        out_csv = "ft_row_streaming_results.csv"

    runs = int(getattr(args, "runs", 3))

    # 你可以用 args.stream_method 指定：
    #   FT-head@EditedHG / FT-K@EditedHG / both
    stream_method = getattr(args, "stream_method", "FT-head@EditedHG")
    methods = []
    if stream_method == "both":
        methods = ["FT-head@EditedHG", "FT-K@EditedHG"]
    else:
        methods = [stream_method]

    all_rows = []
    for m in methods:
        for run_id in range(runs):
            print(f"\n================= {m} RUN {run_id + 1}/{runs} =================")
            rows = run_streaming_one(args, run_id, m)
            all_rows.extend(rows)

    save_streaming(all_rows, out_csv)


if __name__ == "__main__":
    main()
