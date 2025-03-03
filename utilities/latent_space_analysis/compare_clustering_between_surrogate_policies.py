import torch
import numpy as np
from itertools import combinations

from sklearn.metrics import (
    rand_score,
    adjusted_rand_score,
    mutual_info_score,
    adjusted_mutual_info_score,
    normalized_mutual_info_score
)

def compare_clustering_between_surrogate_policies(surrogate_policies_cluster_labels):
    """
    Compare the clustering results between different surrogate policies using various metrics.

    Parameters:
    surrogate_policies_cluster_labels (list[torch.Tensor]): List of cluster labels for each policy.

    Returns:
    dict: Dictionary containing the statistics (mean and standard deviation) for different similarity metrics.
    """

    # Input validation
    if len(surrogate_policies_cluster_labels) < 2:
        raise ValueError("At least two sets of labels are required for comparison.")

    # Initialization of result lists
    metrics = {
        'rand': [],
        'adjusted_rand': [],
        'mutual_info': [],
        'adjusted_mutual_info': [],
        'normalized_mutual_info': []
    }

    # Calculation of metrics for each unique pair
    for labels_1, labels_2 in combinations(surrogate_policies_cluster_labels, 2):
        # Conversion of PyTorch tensors to numpy arrays
        arr1 = labels_1.cpu().numpy()
        arr2 = labels_2.cpu().numpy()

        # Calculation of different metrics
        metrics['rand'].append(rand_score(arr1, arr2))
        metrics['adjusted_rand'].append(adjusted_rand_score(arr1, arr2))
        metrics['mutual_info'].append(mutual_info_score(arr1, arr2))
        metrics['adjusted_mutual_info'].append(adjusted_mutual_info_score(arr1, arr2))
        metrics['normalized_mutual_info'].append(normalized_mutual_info_score(arr1, arr2))

    # Calculation of summary statistics
    results = {}
    for name, values in metrics.items():
        arr = np.array(values)
        results[f'{name}_mean'] = arr.mean()
        results[f'{name}_std'] = arr.std()

    return results
