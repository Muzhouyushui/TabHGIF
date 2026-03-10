
from sklearn.metrics import f1_score
import torch
def evaluate_hgcn_f1(model, data_obj):
    model.eval()
    with torch.no_grad():
        # 只传 laplacian
        logits = model(data_obj["lap"])
        preds  = logits.argmax(dim=-1).cpu().numpy()
        labels = data_obj["y"].cpu().numpy()
    return f1_score(labels, preds, average="macro")

def evaluate_hgcn_acc(model, data_obj):
    model.eval()
    with torch.no_grad():
        logits = model(data_obj["lap"])
        preds  = logits.argmax(dim=-1).cpu()
        labels = data_obj["y"].cpu()
        return (preds == labels).float().mean().item()