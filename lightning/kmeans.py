import torch
import torch.nn as nn
from cuml import KMeans
from cuml.metrics.cluster.silhouette_score import cython_silhouette_score


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
        self.memory = None  # Mémoire pour les anciens embeddings

    def forward(self, embeddings):
        device = embeddings.device

        # Ajouter les nouveaux embeddings à la mémoire
        if self.memory is None:
            self.memory = embeddings.detach()
        else:
            self.memory = torch.cat((self.memory, embeddings.detach()), dim=0)
            max_data_size = self.max_memory_size + embeddings.size(0)

            if self.memory.size(0) > max_data_size:
                self.memory = self.memory[-max_data_size:]

        # Clusterisation sur les embeddings mémorisés + nouveaux
        all_embeddings = self.memory
        cluster_labels_all = torch.tensor(
            self.kmeans.fit_predict(all_embeddings), device=device
        )
        centroids = torch.tensor(self.kmeans.cluster_centers_, device=device)

        # Identifier les labels uniquement pour les nouveaux embeddings
        new_labels = cluster_labels_all[-embeddings.size(0):]

        # Calculer le score de silhouette si nécessaire
        if self.number_points_for_silhouette_score is not None:
            indices = torch.randperm(all_embeddings.size(0))[:self.number_points_for_silhouette_score]
            silhouette_score = cython_silhouette_score(
                X=all_embeddings[indices].detach(),
                labels=cluster_labels_all[indices].detach()
            )
            if self.logger is not None:
                self.logger('silhouette_score', silhouette_score, on_epoch=True)

        return {'cluster_labels': new_labels, 'centroids': centroids}

