# define metrics for evaluation

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import numpy as np
import json
import config


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='macro', zero_division=0),
        "recall": recall_score(y_true, y_pred, average='macro', zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average='macro', zero_division=0)
    }

def save_metrics(metrics, vocab_size = config.DEFAULT_VOCAB_SIZE):
    os.makedirs(config.METRICS_DIR, exist_ok=True)
    path = os.path.join(config.METRICS_DIR, f"metrics_{vocab_size}.json")
    with open(path,'w') as f:
        json.dump(metrics, f, indent=2)

def plot_confusion_matrix(y_true, y_pred, class_names, vocab_size = config.DEFAULT_VOCAB_SIZE):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix (Vocab Size: {vocab_size})')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    os.makedirs(config.CONFUSION_DIR, exist_ok=True)
    path = os.path.join(config.CONFUSION_DIR, f"confusion_matrix_{vocab_size}.png")
    plt.savefig(path)
    print(f"Confusion matrix saved to {path}")
