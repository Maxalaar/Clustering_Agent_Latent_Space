import torch
import torch.nn as nn
import warnings

try:
    from cuml.cluster import DBSCAN
    from cuml.metrics.cluster import silhouette_score
except Exception as e:
    warnings.warn('Error: Unable to import cuml.')


class Dbscan(nn.Module):
    def __init__(
            self,
            epsilon: float = 0.5,
            min_samples: int = 1,
            memory_size: int = 0,
            logger=None,
            number_points_for_silhouette_score: int = 1000,
    ):
        super(Dbscan, self).__init__()
        self.epsilon = epsilon
        self.min_samples = min_samples
        self.max_memory_size = memory_size
        self.logger = logger
        self.number_points_for_silhouette_score: int = number_points_for_silhouette_score
        self.dbscan = DBSCAN(eps=self.epsilon, min_samples=self.min_samples)
        self.memory = None

    def forward(self, embeddings):
        device = embeddings.device

        # Initialize or update memory with new embeddings
        if self.memory is None:
            self.memory = embeddings.detach()
        else:
            self.memory = torch.cat((self.memory, embeddings.detach()), dim=0)
            max_data_size = self.max_memory_size + embeddings.size(0)

            # Limit memory size to the maximum allowed
            if self.memory.size(0) > max_data_size:
                self.memory = self.memory[-max_data_size:]

        all_embeddings = self.memory
        # Perform DBSCAN clustering on all embeddings
        all_cluster_labels = torch.tensor(
            self.dbscan.fit_predict(all_embeddings), device=device
        )
        unique_labels = torch.unique(all_cluster_labels)
        # Log the number of clusters
        self.logger('number_cluster', unique_labels.size(0), on_epoch=True)

        centroids = []
        for label in unique_labels:
            if label != -1:  # Exclude noise points (label -1)
                cluster_points = all_embeddings[all_cluster_labels == label]
                centroid = torch.mean(cluster_points, dim=1)
                centroids.append(centroid)

        centroids = torch.tensor(centroids, device=device)

        # Identify cluster labels for new embeddings only
        new_labels = all_cluster_labels[-embeddings.size(0):]

        # Compute the silhouette score if needed
        if self.number_points_for_silhouette_score is not None:
            indices = torch.randperm(all_embeddings.size(0))[:self.number_points_for_silhouette_score]
            silhouette_score = silhouette_score(
                X=all_embeddings[indices].detach(),
                labels=all_cluster_labels[indices].detach()
            )
            if self.logger is not None:
                self.logger('silhouette_score', silhouette_score, on_epoch=True)

        return {'cluster_labels': new_labels, 'centroids': centroids}
