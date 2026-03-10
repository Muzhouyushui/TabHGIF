from sklearn.metrics import f1_score
import torch
#####################
# ========== 评估函数（用F1代替Acc） ==========
#####################
def evaluate_test_f1(model, data_obj):
    """
    在 data_obj (测试集) 上评估 F1 分数
    data_obj 包含:
      "x", "y", "H", "dv_inv", "de_inv"
    """
    model.eval()
    with torch.no_grad():
        logits = model(data_obj["x"], data_obj["H"], data_obj["dv_inv"], data_obj["de_inv"])
        preds = logits.argmax(dim=-1).cpu().numpy()


        labels = data_obj["y"].cpu().numpy()

        # 计算正标签、负标签以及分类错误的标签数量
        true_pos = ((preds == 1) & (labels == 1)).sum()  # 预测为正且实际为正
        true_neg = ((preds == 0) & (labels == 0)).sum()  # 预测为负且实际为负
        false_pos = ((preds == 1) & (labels == 0)).sum()  # 预测为正但实际为负
        false_neg = ((preds == 0) & (labels == 1)).sum()  # 预测为负但实际为正

        # # 输出相关信息
        # print(f"Evaluation set:")
        # print(f"  - True Positive: {true_pos}")
        # print(f"  - True Negative: {true_neg}")
        # print(f"  - False Positive: {false_pos}")
        # print(f"  - False Negative: {false_neg}")
        # print(f"  - Total Positive Labels: {true_pos + false_neg}")
        # print(f"  - Total Negative Labels: {true_neg + false_pos}")

        return f1_score(labels, preds, average="macro")


def evaluate_test_acc(model, data_obj):
    """
    在 data_obj (测试集) 上评估 Accuracy
    用于衡量 Retain-Set Accuracy (↑)
    """
    model.eval()
    with torch.no_grad():
        logits = model(data_obj["x"], data_obj["H"], data_obj["dv_inv"], data_obj["de_inv"])
        preds = logits.argmax(dim=-1).cpu()
        labels = data_obj["y"].cpu()
        acc = (preds == labels).float().mean().item()
        return acc


def evaluate_hgcn_f1(model, data_obj):
    """
    在 HGCN 模型上评估 micro-F1。

    参数
    ────
    model : HyperGCN
        已加载好结构（model.structure）的模型实例。
    data_obj : dict
        测试集数据，需包含
          - "x": 特征张量，shape (N, D)
          - "y": 标签张量，shape (N,)

    返回
    ────
    f1 : float
        micro-F1 分数
    """
    model.eval()
    with torch.no_grad():
        logits = model(data_obj["x"])  # 隐式用 model.structure
        preds = logits.argmax(dim=1).cpu().numpy()
        labels = data_obj["y"].cpu().numpy()

    # 计算正标签、负标签以及分类错误的标签数量
    true_pos = ((preds == 1) & (labels == 1)).sum()  # 预测为正且实际为正
    true_neg = ((preds == 0) & (labels == 0)).sum()  # 预测为负且实际为负
    false_pos = ((preds == 1) & (labels == 0)).sum()  # 预测为正但实际为负
    false_neg = ((preds == 0) & (labels == 1)).sum()  # 预测为负但实际为正

    # # 输出相关信息
    # print(f"Evaluation set:")
    # print(f"  - True Positive: {true_pos}")
    # print(f"  - True Negative: {true_neg}")
    # print(f"  - False Positive: {false_pos}")
    # print(f"  - False Negative: {false_neg}")
    # print(f"  - Total Positive Labels: {true_pos + false_neg}")
    # print(f"  - Total Negative Labels: {true_neg + false_pos}")

     # 计算 micro-F1 分数

    return f1_score(labels, preds, average="micro")


def evaluate_hgcn_acc(model, data_obj):
    """
    在 HGCN 模型上评估 Accuracy（Retain-Set Accuracy）。

    参数同上。

    返回
    ────
    acc : float
        (preds == labels) 的平均值
    """
    model.eval()
    with torch.no_grad():
        logits = model(data_obj["x"])
        preds = logits.argmax(dim=1)
        labels = data_obj["y"]
        acc = (preds == labels).float().mean().item()
    return acc


import random


def spot_check_samples(fts_te, lbls_te, model, k=15):
    """
    对整个测试集做一次前向，然后随机抽 k 个样本打印：
      特征向量 → 真实标签 vs. 预测标签 (概率)
    fts_te: [N, feature_dim] FloatTensor
    lbls_te: [N] LongTensor
    model: 已训练好的 HyperGCN，输出 LogSoftmax
    """
    model.eval()
    N = fts_te.size(0)
    # 随机选 k 个样本索引
    indices = random.sample(range(N), k)

    with torch.no_grad():
        # 对全图做一次前向
        log_probs_all = model(fts_te)    # [N, n_class]
        probs_all     = log_probs_all.exp()
        preds_all     = probs_all.argmax(dim=1)

    # 按索引打印
    for idx in indices:
        feat     = fts_te[idx]            # [feature_dim]
        true_lbl = lbls_te[idx].item()    # int
        pred_lbl = preds_all[idx].item()  # int
        pred_prob= probs_all[idx, pred_lbl].item()
        # print(f"样本索引: {idx}")
        # print(f"特征向量: {feat.cpu().tolist()}")
        # print(f"真实标签: {true_lbl}  →  预测标签: {pred_lbl}  (概率: {pred_prob:.4f})")
        # print("-" * 50)


def spot_check_samples_HGNN(test_data_obj, model, k=15):
    """
    随机抽 k 条测试样本，打印特征向量、真实标签 → 预测标签 (概率)。
    test_data_obj 包含：
      "x":      [N, D] FloatTensor
      "H":      sparse_coo_tensor for incidence
      "dv_inv": [N] FloatTensor
      "de_inv": [#edges] FloatTensor
      "y":      [N] LongTensor
    model.forward(fts, H, dv_inv, de_inv) -> log_probs [N, n_class]
    """
    model.eval()

    fts_te = test_data_obj["x"]
    H       = test_data_obj["H"]
    dv_inv  = test_data_obj["dv_inv"]
    de_inv  = test_data_obj["de_inv"]
    lbls_te = test_data_obj["y"]

    N = fts_te.size(0)
    indices = random.sample(range(N), k)

    with torch.no_grad():
        # 全量前向
        log_probs_all = model(fts_te, H, dv_inv, de_inv)  # [N, n_class]
        probs_all     = log_probs_all.exp()
        preds_all     = probs_all.argmax(dim=1)

    for idx in indices:
        feat      = fts_te[idx]            # [feature_dim]
        true_lbl  = lbls_te[idx].item()
        pred_lbl  = preds_all[idx].item()
        pred_prob = probs_all[idx, pred_lbl].item()

        # print(f"样本索引: {idx}")
        # print(f"特征向量: {feat.cpu().tolist()}")
        # print(f"真实标签: {true_lbl}  →  预测标签: {pred_lbl}  (概率: {pred_prob:.4f})")
        # print("-" * 50)
