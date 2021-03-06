from typing import Optional, List
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
