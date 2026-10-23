import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)


def classification_metrics(labels, probabilities):
    predictions = probabilities.argmax(axis=1)
    one_hot = np.eye(probabilities.shape[1])[labels]
    valid = (one_hot.sum(0) > 0) & (one_hot.sum(0) < len(labels))
    return {
        "Accuracy": float(accuracy_score(labels, predictions)),
        "Precision": float(precision_score(labels, predictions, average="macro", zero_division=0)),
        "Recall": float(recall_score(labels, predictions, average="macro", zero_division=0)),
        "F1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "AUC": float(np.mean([roc_auc_score(one_hot[:, c], probabilities[:, c])
                              for c in np.where(valid)[0]])) if valid.any() else None,
        "AUPR": float(np.mean([average_precision_score(one_hot[:, c], probabilities[:, c])
                               for c in np.where(valid)[0]])) if valid.any() else None,
    }
