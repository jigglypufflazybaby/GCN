import matplotlib.pyplot as plt
import numpy as np

def visualize_feature_distribution(features, labels=None):
    """
    Visualize the distribution of the features
    :param features: The features to be visualized (e.g., face normals, areas, etc.)
    :param labels: (optional) Labels for classification, to color the points differently
    """
    if labels is not None:
        plt.scatter(features[:, 0], features[:, 1], c=labels)
    else:
        plt.scatter(features[:, 0], features[:, 1])
    plt.title("Feature Distribution")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

def visualize_predictions(predictions, ground_truth):
    """
    Visualize predictions vs. ground truth
    :param predictions: Model predictions
    :param ground_truth: Ground truth labels
    """
    plt.plot(predictions, label='Predictions')
    plt.plot(ground_truth, label='Ground Truth')
    plt.legend()
    plt.show()
