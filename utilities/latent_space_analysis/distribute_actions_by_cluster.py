import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from typing import List


def distribution_actions_by_cluster(
    observations: torch.Tensor,
    cluster_labels: torch.Tensor,
    actions: torch.Tensor,
    save_path: Path,
    class_names: List[str] = None,
):
    save_path = save_path / 'distribution_actions_by_cluster'
    os.makedirs(save_path, exist_ok=True)

    # Check that the inputs have compatible dimensions
    if not (observations.shape[0] == cluster_labels.shape[0] == actions.shape[0]):
        raise ValueError("The dimensions of observations, cluster_labels, and actions must be compatible.")

    is_convertible_to_int = torch.all(actions == actions.to(torch.int))

    if not is_convertible_to_int:
        return

    # Get the number of clusters
    num_clusters = len(np.unique(cluster_labels.cpu().numpy()))

    # If action_names is not provided, use default names
    if class_names is None:
        class_names = [f'Action {i}' for i in np.unique(actions.cpu().numpy())]

    # Iterate over each cluster and create a histogram
    for cluster in range(num_clusters):
        # Select actions for the current cluster
        cluster_actions = actions[cluster_labels == cluster]

        # Count the number of each type of action in the cluster
        action_counts = np.bincount(cluster_actions.cpu().numpy(), minlength=len(class_names))

        # Create the histogram
        plt.figure(figsize=(10, 6))
        plt.bar(class_names, action_counts, color='skyblue')
        plt.xlabel('Actions')
        plt.ylabel('Count')
        plt.title(f'Action distribution for cluster {cluster}')
        plt.xticks(rotation=45)

        # Save the histogram as an image
        plt.savefig(save_path / f'cluster_{cluster}_action_distribution.png')
        plt.close()


