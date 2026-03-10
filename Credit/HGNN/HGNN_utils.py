from sklearn.metrics import f1_score
import torch
#####################
# ========== 评估函数（用F1代替Acc） ==========
from sklearn.metrics import f1_score

def evaluate_test_f1(model, data_obj):
    model.eval()
    with torch.no_grad():
        logits = model(
            data_obj["x"],
            data_obj["H"],
            data_obj["dv_inv"],
            data_obj["de_inv"]
        )
        preds  = logits.argmax(dim=-1).cpu().numpy()
        labels = data_obj["y"].cpu().numpy()

    # 只返回 macro F1
    f1_macro = f1_score(labels, preds, average="macro")
    return f1_macro

def evaluate_test_acc(model, data_obj):
    """
    在 data_obj (测试集) 上评估 Accuracy
    用于衡量 Retain-Set Accuracy (↑)
    """
    model.eval()
    with torch.no_grad():
        logits = model(
            data_obj["x"],
            data_obj["H"],
            data_obj["dv_inv"],
            data_obj["de_inv"]
        )
        preds  = logits.argmax(dim=-1).cpu()
        labels = data_obj["y"].cpu()
        acc    = (preds == labels).float().mean().item()
    return acc
