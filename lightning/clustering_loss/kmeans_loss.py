import torch
from cuml import KMeans
import torch.nn as nn


class KmeansLoss(nn.Module):
    def __init__(
            self,
            number_cluster: int,
            margin_between_clusters: float = 4.0,
            logger=None,
    ):
        super(KmeansLoss, self).__init__()
        self.number_cluster = number_cluster
        self.margin_between_clusters: float = margin_between_clusters
        self.logger = logger
        self.centroids = None
        self.cluster_labels = None

    def compute_new_centroid(self, embeddings):
        device = embeddings.device
        kmeans = KMeans(n_clusters=self.number_cluster)
        new_cluster_labels = torch.tensor(kmeans.fit_predict(embeddings.detach()), device=device)
        new_centroids = torch.tensor(kmeans.cluster_centers_, device=device)

        self.centroids = new_centroids
        self.cluster_labels = new_cluster_labels

    def forward(self, embeddings: torch.Tensor, current_global_step):
        self.compute_new_centroid(embeddings)

        device = embeddings.device
        attraction_loss = torch.tensor(0.0, device=device)
        repulsion_loss = torch.tensor(0.0, device=device)
        distance_intra_cluster = 0.0

        # Compute attraction and repulsion loss
        for i in range(self.number_cluster):
            current_centroid = self.centroids[i].unsqueeze(dim=0)
            ixd_points_current_cluster = torch.nonzero(self.cluster_labels == i).squeeze(dim=1)
            points_current_cluster = embeddings[ixd_points_current_cluster]
            other_centroids = torch.cat([self.centroids[:i], self.centroids[i + 1:]])

            # Attraction and repulsion terms
            attraction_distances = torch.cdist(points_current_cluster, current_centroid)
            attraction_loss += torch.mean(attraction_distances ** 2)

            repulsion_distances = torch.cdist(points_current_cluster, other_centroids)
            repulsion_loss += torch.mean(torch.nn.functional.relu((self.margin_between_clusters - repulsion_distances) ** 2))

            # Intra-cluster distance
            distance_intra_cluster += torch.mean(torch.norm(points_current_cluster - current_centroid, dim=1)).item()

        total_loss = (attraction_loss + repulsion_loss) / self.number_cluster

        # Logging
        if self.logger:
            matrix_distance_centroids = torch.cdist(self.centroids, self.centroids).triu(diagonal=1)
            distance_centroids = matrix_distance_centroids[matrix_distance_centroids != 0]
            self.logger('global_loss_coefficient', 1.0, on_epoch=True)
            self.logger('kmeans_loss_average_distance_centroids', torch.mean(distance_centroids).item(), on_epoch=True)
            self.logger('kmeans_loss_min_distance_centroids', distance_centroids.min().item(), on_epoch=True)
            self.logger('kmeans_loss_max_distance_centroids', distance_centroids.max().item(), on_epoch=True)
            self.logger('kmeans_loss_mean_distance_intra_cluster', distance_intra_cluster, on_epoch=True)
            self.logger('kmeans_loss_attraction_loss', attraction_loss.item(), on_epoch=True)
            self.logger('kmeans_loss_repulsion_loss', repulsion_loss.item(), on_epoch=True)
            self.logger('kmeans_loss_total_loss', total_loss.item(), on_epoch=True)

        return total_loss
