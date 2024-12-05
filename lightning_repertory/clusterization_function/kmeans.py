import warnings

import torch
import torch.nn as nn

try:
    from cuml import KMeans
    from cuml.metrics.cluster import silhouette_score
except Exception as e:
    warnings.warn('Error: Unable to import cuml.')



class Kmeans(nn.Module):
    def __init__(
            self,
            number_cluster: int,
            memory_size: int = 0,
            logger=None,
            number_points_for_silhouette_score: int = 1000,
    ):
        super(Kmeans, self).__init__()
        self.number_cluster = number_cluster
        self.max_memory_size = memory_size
        self.logger = logger
        self.number_points_for_silhouette_score: int = number_points_for_silhouette_score
        self.kmeans = KMeans(n_clusters=self.number_cluster)
        self.memory = None  # Memory for previous embeddings

    def forward(self, embeddings):
        device = embeddings.device

        # Add new embeddings to memory
        if self.memory is None:
            self.memory = embeddings.detach()
        else:
            self.memory = torch.cat((self.memory, embeddings.detach()), dim=0)
            max_data_size = self.max_memory_size + embeddings.size(0)

            # Limit memory size to the maximum allowed
            if self.memory.size(0) > max_data_size:
                self.memory = self.memory[-max_data_size:]

        # Perform clustering on the combined memory and new embeddings
        all_embeddings = self.memory
        cluster_labels_all = torch.tensor(
            self.kmeans.fit_predict(all_embeddings), device=device
        )
        centroids = torch.tensor(self.kmeans.cluster_centers_, device=device)

        # Identify cluster labels only for the new embeddings
        new_labels = cluster_labels_all[-embeddings.size(0):]

        # Compute the silhouette score if needed
        if self.number_points_for_silhouette_score is not None:
            indices = torch.randperm(all_embeddings.size(0))[:self.number_points_for_silhouette_score]
            silhouette_score = silhouette_score(
                X=all_embeddings[indices].detach(),
                labels=cluster_labels_all[indices].detach()
            )
            if self.logger is not None:
                self.logger('silhouette_score', silhouette_score, on_epoch=True)

        return {'cluster_labels': new_labels, 'centroids': centroids}


