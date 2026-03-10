
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from collections import Counter

from utils.common_utils import evaluate_test_acc, evaluate_test_f1
from Credit.HGNN.data_preprocessing_Credit_col import (
    generate_hyperedge_dict,
    preprocess_node_features,
    delete_feature_columns,
)
from bank.HGNN.HGNN import HGNN_implicit, build_incidence_matrix, compute_degree_vectors
from GIF.GIF_HGNN_COL import approx_gif_col, train_model
from paths import CREDIT_DATA

def main():
    parser = argparse.ArgumentParser(description="HGNN on Credit Approval 数据集 - 列级 Unlearning 版")
    # parser.add_argument(
    #     "--data-csv", type=str,
    #     help="完整 Credit Approval 数据集路径（no header，缺失用 '?' 表示）"
    # )
    parser.add_argument("--data-csv", type=str,
                        default=CREDIT_DATA,
                        help="Credit Approval data CSV file path")
    parser.add_argument("--split-ratio", type=float, default=0.2,
                        help="测试集比例")
    parser.add_argument("--max-nodes-per-hyperedge-train", type=int, default=50,
                        help="训练集超边最大节点数")
    parser.add_argument("--max-nodes-per-hyperedge-test", type=int, default=50,
                        help="测试集超边最大节点数")
    parser.add_argument("--hidden-dim", type=int, default=128, help="HGNN 隐藏维度")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout 比例")
    parser.add_argument("--lr", type=float, default=0.01, help="学习率")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="Adam 权重衰减")
    parser.add_argument("--epochs", type=int, default=200, help="训练轮数")
    parser.add_argument("--milestones", nargs='+', type=int, default=[100,150], help="学习率里程碑")
    parser.add_argument("--gamma", type=float, default=0.1, help="学习率衰减系数")
    parser.add_argument("--gif-iters", type=int, default=20, help="GIF 迭代次数")
    parser.add_argument("--gif-damp", type=float, default=0.01, help="GIF 阻尼")
    parser.add_argument("--gif-scale", type=float, default=1e7, help="GIF scale")
    parser.add_argument(
        "--columns-to-unlearn", nargs=1, default=["A5"],
        help="要进行列级 Unlearning 的列名列表（单列），如 A5"
    )
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {device}")

    # —— 1) 读取 Credit Approval 数据并拆分 ——
    df = pd.read_csv(
        args.data_csv,
        header=None,
        na_values='?',
        skipinitialspace=True
    )
    # 原始 16 列：A1–A15 + class
    df.columns = [f"A{i}" for i in range(1,16)] + ["class"]
    # 映射标签
    df["y"] = df["class"].map({"+": 1, "-": 0})
    df_train, df_test = train_test_split(
        df,
        test_size=args.split_ratio,
        random_state=42,
        stratify=df["y"]
    )
    df_train = df_train.reset_index(drop=True)
    df_test  = df_test .reset_index(drop=True)
    print(f"训练集样本数: {len(df_train)}, 测试集样本数: {len(df_test)}")
    print("– TRAIN label dist:", Counter(df_train["y"]))
    print("– TEST  label dist:", Counter(df_test["y"]))

    # —— 2) 预处理训练集 ——
    X_train, y_train, df_train_proc, transformer = preprocess_node_features(
        df_train, transformer=None
    )
    print(f"预处理后 TRAIN shape: {X_train.shape}, labels: {Counter(y_train)}")

    # —— 3) 构建训练超图 ——
    # Credit Approval 的连续/类别列定义
    cont_cols = ["A2","A3","A8","A11","A14","A15"]
    cat_cols  = [f"A{i}" for i in range(1,16) if f"A{i}" not in cont_cols]
    hyperedges_train = generate_hyperedge_dict(
        df_train_proc,
        feature_cols=cat_cols+cont_cols,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge_train,
        device=device
    )
    H_sp_tr = build_incidence_matrix(hyperedges_train, X_train.shape[0])
    dv_tr, de_tr = compute_degree_vectors(H_sp_tr)
    Hc_tr = H_sp_tr.tocoo()
    idx_tr = np.vstack((Hc_tr.row, Hc_tr.col)).astype(np.int64)
    H_tensor_tr = torch.sparse_coo_tensor(
        torch.from_numpy(idx_tr),
        torch.from_numpy(Hc_tr.data).float(),
        size=Hc_tr.shape
    ).to(device)
    fts_tr  = torch.from_numpy(X_train).float().to(device)
    lbls_tr = torch.LongTensor(y_train).to(device)

    # —— 4) 初始化模型 & 训练 ——
    model = HGNN_implicit(
        in_ch=fts_tr.shape[1],
        n_class=int(max(y_train)) + 1,
        n_hid=args.hidden_dim,
        dropout=args.dropout
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    criterion = nn.CrossEntropyLoss()

    # 兼容你的 reasoning 接口
    def reason_once(data):
        return model(data["x"], data["H"], data["dv_inv"], data["de_inv"])
    model.reason_once = reason_once
    model.reason_once_unlearn = reason_once

    print("—— 开始训练 ——")
    model = train_model(
        model, criterion, optimizer, scheduler,
        fts_tr, lbls_tr, H_tensor_tr,
        torch.from_numpy(dv_tr).float().to(device),
        torch.from_numpy(de_tr).float().to(device),
        num_epochs=args.epochs, print_freq=10
    )

    # —— 5) 预处理测试集 & 第一次评估 ——
    X_test, y_test, df_test_proc, _ = preprocess_node_features(
        df_test, transformer=transformer
    )
    print(f"预处理后 TEST shape: {X_test.shape}, labels: {Counter(y_test)}")

    hyperedges_test = generate_hyperedge_dict(
        df_test_proc,
        feature_cols=cat_cols+cont_cols,
        max_nodes_per_hyperedge=50,
        device=device
    )
    H_sp_te = build_incidence_matrix(hyperedges_test, X_test.shape[0])
    dv_te, de_te = compute_degree_vectors(H_sp_te)
    Hc_te = H_sp_te.tocoo()
    idx_te = np.vstack((Hc_te.row, Hc_te.col)).astype(np.int64)
    H_tensor_te = torch.sparse_coo_tensor(
        torch.from_numpy(idx_te),
        torch.from_numpy(Hc_te.data).float(),
        size=Hc_te.shape
    ).to(device)
    fts_te  = torch.from_numpy(X_test).float().to(device)
    lbls_te = torch.LongTensor(y_test).to(device)

    test_obj = {
        "x": fts_te, "y": lbls_te,
        "H": H_tensor_te,
        "dv_inv": torch.from_numpy(dv_te).float().to(device),
        "de_inv": torch.from_numpy(de_te).float().to(device)
    }
    f1_before, acc_before = evaluate_test_f1(model, test_obj), evaluate_test_acc(model, test_obj)
    print(f"Before Unlearning — F1: {f1_before:.4f}, Acc: {acc_before:.4f}")

    # —— 6) 列级 Unlearning ——
    col_to_remove = args.columns_to_unlearn[0]

    # 6a) 备份训练时结构
    fts_b, H_b, dv_b, de_b = (
        fts_tr.clone(),
        H_tensor_tr.coalesce(),
        torch.from_numpy(dv_tr).float().to(device),
        torch.from_numpy(de_tr).float().to(device)
    )

    # —— 6b) 删除训练集列并通过 GIF 更新 —— #
    # 注意：delete_feature_columns 返回顺序为 (X_zeroed, new_hyperedges, H_tensor_new)
    fts_a, hyperedges_a, H_tensor_a = delete_feature_columns(
        fts_tr.clone(),
        transformer,
        [col_to_remove],
        hyperedges_train,
        df_train_proc,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge_train,
        device=device
    )
    # 现在 hyperedges_a 是 dict，H_tensor_a 是张量
    H_sp_a = build_incidence_matrix(hyperedges_a, fts_a.shape[0])
    dv_a, de_a = compute_degree_vectors(H_sp_a)
    # … 后续代码保持不变 …
    Hc_a = H_sp_a.tocoo()
    idx_a = np.vstack((Hc_a.row, Hc_a.col)).astype(np.int64)
    H_tensor_a = torch.sparse_coo_tensor(
        torch.from_numpy(idx_a),
        torch.from_numpy(Hc_a.data).float(),
        size=Hc_a.shape
    ).to(device)

    batch_before = (fts_b, H_b, dv_b, de_b, lbls_tr)
    batch_after  = (fts_a, H_tensor_a, torch.from_numpy(dv_a).float().to(device),
                    torch.from_numpy(de_a).float().to(device), lbls_tr)
    v_updates    = approx_gif_col(
        model, criterion, batch_before, batch_after,
        cg_iters=args.gif_iters, damping=args.gif_damp, scale=args.gif_scale
    )
    with torch.no_grad():
        for p, v in zip(model.parameters(), v_updates):
            p.sub_(v)

    # —— 7) 删除后的测试评估 —— #
    fts_te_new, hyperedges_te_new, H_tensor_te_new = delete_feature_columns(
        fts_te.clone(),
        transformer,
        [col_to_remove],
        hyperedges_test,
        df_test_proc,
        max_nodes_per_hyperedge=args.max_nodes_per_hyperedge_test,
        device=device
    )
    H_sp_te_new = build_incidence_matrix(hyperedges_te_new, fts_te_new.shape[0])
    dv_te_new, de_te_new = compute_degree_vectors(H_sp_te_new)
    # … 后续代码保持不变 …
    Hc_te_new = H_sp_te_new.tocoo()
    idx_te_new = np.vstack((Hc_te_new.row, Hc_te_new.col)).astype(np.int64)
    H_tensor_te_new = torch.sparse_coo_tensor(
        torch.from_numpy(idx_te_new),
        torch.from_numpy(Hc_te_new.data).float(),
        size=Hc_te_new.shape
    ).to(device)

    test_obj_after = {
        "x": fts_te_new, "y": lbls_te,
        "H": H_tensor_te_new,
        "dv_inv": torch.from_numpy(dv_te_new).float().to(device),
        "de_inv": torch.from_numpy(de_te_new).float().to(device)
    }
    f1_after, acc_after = evaluate_test_f1(model, test_obj_after), evaluate_test_acc(model, test_obj_after)
    print(f"After Unlearning  — F1: {f1_after:.4f}, Acc: {acc_after:.4f}")

if __name__ == "__main__":
    main()
    for run in range(1,4):
        print("=== Run",run,"===")
        main()