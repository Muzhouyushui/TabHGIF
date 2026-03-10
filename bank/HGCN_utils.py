from sklearn.metrics import f1_score
import torch

def evaluate_hgcn_f1(model, data_obj):
    """
    Evaluate micro-F1 on the HGCN model.

    Parameters
    ────
    model : HyperGCN
        A model instance with structure already loaded (model.structure).
    data_obj : dict
        Test-set data, which must contain:
          - "x": feature tensor, shape (N, D)
          - "y": label tensor, shape (N,)

    Returns
    ────
    f1 : float
        micro-F1 score
    """
    model.eval()
    with torch.no_grad():
        logits = model(data_obj["x"])  # implicitly uses model.structure
        preds = logits.argmax(dim=1).cpu().numpy()
        labels = data_obj["y"].cpu().numpy()

    # Compute the numbers of positive labels, negative labels, and classification errors
    true_pos = ((preds == 1) & (labels == 1)).sum()  # predicted positive and actually positive
    true_neg = ((preds == 0) & (labels == 0)).sum()  # predicted negative and actually negative
    false_pos = ((preds == 1) & (labels == 0)).sum()  # predicted positive but actually negative
    false_neg = ((preds == 0) & (labels == 1)).sum()  # predicted negative but actually positive

    # Output related information
    print(f"Evaluation set:")
    print(f"  - True Positive: {true_pos}")
    print(f"  - True Negative: {true_neg}")
    print(f"  - False Positive: {false_pos}")
    print(f"  - False Negative: {false_neg}")
    print(f"  - Total Positive Labels: {true_pos + false_neg}")
    print(f"  - Total Negative Labels: {true_neg + false_pos}")

    # Compute the micro-F1 score
    return f1_score(labels, preds, average="micro")


def evaluate_hgcn_acc(model, data_obj):
    """
    Evaluate Accuracy (Retain-Set Accuracy) on the HGCN model.

    Parameters are the same as above.

    Returns
    ────
    acc : float
        Mean value of (preds == labels)
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
    Perform one forward pass on the whole test set, then randomly print k samples:
      feature vector → true label vs. predicted label (probability)
    fts_te: [N, feature_dim] FloatTensor
    lbls_te: [N] LongTensor
    model: a trained HyperGCN that outputs LogSoftmax
    """
    model.eval()
    N = fts_te.size(0)
    # Randomly select k sample indices
    indices = random.sample(range(N), k)

    with torch.no_grad():
        # Perform one forward pass on the whole graph
        log_probs_all = model(fts_te)    # [N, n_class]
        probs_all     = log_probs_all.exp()
        preds_all     = probs_all.argmax(dim=1)

    # Print by index
    for idx in indices:
        feat      = fts_te[idx]            # [feature_dim]
        true_lbl  = lbls_te[idx].item()    # int
        pred_lbl  = preds_all[idx].item()  # int
        pred_prob = probs_all[idx, pred_lbl].item()
        # print(f"Sample index: {idx}")
        # print(f"Feature vector: {feat.cpu().tolist()}")
        # print(f"True label: {true_lbl}  →  Predicted label: {pred_lbl}  (Probability: {pred_prob:.4f})")
        # print("-" * 50)


def spot_check_samples_HGNN(test_data_obj, model, k=15):
    """
    Randomly select k test samples and print feature vectors, true labels → predicted labels (probabilities).
    test_data_obj contains:
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
        # Full forward pass
        log_probs_all = model(fts_te, H, dv_inv, de_inv)  # [N, n_class]
        probs_all     = log_probs_all.exp()
        preds_all     = probs_all.argmax(dim=1)

    for idx in indices:
        feat      = fts_te[idx]            # [feature_dim]
        true_lbl  = lbls_te[idx].item()
        pred_lbl  = preds_all[idx].item()
        pred_prob = probs_all[idx, pred_lbl].item()

        # print(f"Sample index: {idx}")
        # print(f"Feature vector: {feat.cpu().tolist()}")
        # print(f"True label: {true_lbl}  →  Predicted label: {pred_lbl}  (Probability: {pred_prob:.4f})")
        # print("-" * 50)