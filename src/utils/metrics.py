import numpy as np
from sklearn.metrics import precision_score, recall_score

def compute_precision_recall(predictions, labels):
    """
    Compute Precision and Recall for multi-label classification
    :param predictions: Model predictions (binary matrix for multi-labels)
    :param labels: Ground truth labels (binary matrix)
    """
    precision = precision_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')
    return precision, recall
