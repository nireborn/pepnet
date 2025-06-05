import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, roc_curve, auc, precision_recall_curve


def evaluate_thresholds_and_plot(yt, y_pred, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)

    thresholds = np.arange(0.1, 1.0, 0.1)
    accs = []
    pr_aucs = []
    losses = []

    for thresh in thresholds:
        preds = (y_pred >= thresh).astype(int)
        acc = accuracy_score(yt, preds)
        ap = average_precision_score(yt, y_pred)
        loss = -np.mean(yt * np.log(y_pred + 1e-7) + (1 - yt) * np.log(1 - y_pred + 1e-7))

        accs.append(acc)
        pr_aucs.append(ap)
        losses.append(loss)

    # 1. 损失图
    plt.figure()
    plt.plot(thresholds, losses, marker='o')
    plt.title("Binary Crossentropy vs Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(f"{output_dir}/loss_vs_threshold.png")

    # 2. ACC vs PR-AUC
    plt.figure()
    plt.plot(thresholds, accs, marker='o', label="Accuracy")
    plt.plot(thresholds, pr_aucs, marker='s', label="PR-AUC")
    plt.title("Accuracy & PR-AUC vs Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/acc_prauc_vs_threshold.png")

    # 3. ROC 曲线（使用 PR-AUC 最佳阈值）
    best_idx = np.argmax(pr_aucs)
    best_thresh = thresholds[best_idx]
    best_preds = (y_pred >= best_thresh).astype(int)

    fpr, tpr, _ = roc_curve(yt, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.title(f"ROC Curve (Best PR-AUC @ Threshold={best_thresh:.2f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(f"{output_dir}/roc_best_prauc.png")

    # 4. PR 曲线
    precision, recall, _ = precision_recall_curve(yt, y_pred)
    pr_auc_score = average_precision_score(yt, y_pred)

    plt.figure()
    plt.plot(recall, precision, color='purple', lw=2, label=f'PR curve (AUC = {pr_auc_score:.4f})')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.savefig(f"{output_dir}/pr_curve.png")

    return best_thresh, accs, pr_aucs, losses
