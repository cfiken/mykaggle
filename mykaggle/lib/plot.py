from typing import Optional, List
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(preds: np.ndarray, labels: np.ndarray, class_labels: Optional[List[str]] = None):
    '''
    Confusion Matrix をプロットする
    Args:
      preds: 予測, [num_data]
      labels: 正解データ, [num_data]
      class_labels: クラスの名前のリスト, [num_classes]
    '''
    if class_labels is None:
        class_labels = np.unique(labels)
    cm = confusion_matrix(labels, preds, class_labels)
    cm = cm.astype(np.float) / cm.sum(axis=1)[:, np.newaxis]
    annot = np.around(cm, 2)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, xticklabels=class_labels, yticklabels=class_labels, cmap='Blues', annot=annot, linewidths=0.5)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_aspect('equal')
    plt.tight_layout()


def plot_regression_prediction(
    y_true: np.ndarray, y_pred: np.ndarray, ckptdir: Path
) -> None:
    plt.figure(figsize=(5, 5))
    plt.plot([-3, 2], [-3, 2], color='black')
    plt.scatter(y_true, y_pred, alpha=0.2)
    plt.xlim(-3, 2)
    plt.ylim(-3, 2)
    plt.xlabel('True')
    plt.ylabel('Pred')
    plt.tight_layout()
    plt.savefig(ckptdir / 'prediction.png')
