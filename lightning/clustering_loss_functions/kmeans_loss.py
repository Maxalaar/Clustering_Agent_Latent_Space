import torch
from cuml import KMeans
import torch.nn as nn
import torch.nn.functional as F
import ot


class KMeansLoss(nn.Module):
    def __init__(
            self, number_cluster: int,
            margin_between_clusters: int = 10,
            sliding_centroids: bool = True,
            centroid_learning_rate: float = 0.2,
            logger=None,
    ):
        super(KMeansLoss, self).__init__()
        self.number_cluster = number_cluster
        self.logger = logger
        self.centroids = None
        self.cluster_labels = None
        self.sliding_centroids = sliding_centroids
        self.centroid_learning_rate: float = centroid_learning_rate
        self.margin_between_clusters = margin_between_clusters

    def compute_new_centroid(self, embeddings):
        device = embeddings.device
        kmeans = KMeans(n_clusters=self.number_cluster)
        new_cluster_labels = torch.tensor(kmeans.fit_predict(embeddings.detach()), device=device)
        new_centroids = torch.tensor(kmeans.cluster_centers_, device=device)

        if not self.sliding_centroids:
            self.centroids = new_centroids
            self.cluster_labels = new_cluster_labels

        else:
            if self.centroids is None:
                self.centroids = new_centroids
                self.cluster_labels = new_cluster_labels
                return

            self.cluster_labels = -1 * torch.ones_like(new_cluster_labels)

            new_centroid_distribution = torch.tensor(ot.unif(self.number_cluster)).to(device)
            centroid_distribution = torch.tensor(ot.unif(self.number_cluster)).to(device)

            cost_matrix = ot.dist(self.centroids, new_centroids, metric='euclidean')
            transport_matrix = ot.emd(new_centroid_distribution, centroid_distribution, cost_matrix)

            for i in range(self.number_cluster):
                idx_centroid_to_match = transport_matrix[i].nonzero().item()
                centroid_to_match = new_centroids[idx_centroid_to_match]
                error = centroid_to_match - self.centroids[i]
                error_norm = torch.norm(error)

                if error_norm < self.centroid_learning_rate:
                    self.centroids[i] = centroid_to_match
                else:
                    self.centroids[i] += self.centroid_learning_rate * (error/error_norm)

                self.cluster_labels[torch.nonzero(new_cluster_labels == idx_centroid_to_match).squeeze(dim=1)] = i

    def forward(self, embeddings: torch.Tensor):
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
            repulsion_loss += torch.mean(F.relu(self.margin_between_clusters - repulsion_distances) ** 2)

            # Intra-cluster distance
            distance_intra_cluster += torch.mean(torch.norm(points_current_cluster - current_centroid, dim=1)).item()

        total_loss = attraction_loss + repulsion_loss

        # Logging
        if self.logger:
            matrix_distance_centroids = torch.cdist(self.centroids, self.centroids).triu(diagonal=1)
            distance_centroids = matrix_distance_centroids[matrix_distance_centroids != 0]
            self.logger('kmeans_loss_average_distance_centroids', torch.mean(distance_centroids).item(), on_epoch=True)
            self.logger('kmeans_loss_min_distance_centroids', distance_centroids.min().item(), on_epoch=True)
            self.logger('kmeans_loss_max_distance_centroids', distance_centroids.max().item(), on_epoch=True)
            self.logger('kmeans_loss_mean_distance_intra_cluster', distance_intra_cluster, on_epoch=True)
            self.logger('kmeans_loss_attraction_loss', attraction_loss.item(), on_epoch=True)
            self.logger('kmeans_loss_repulsion_loss', repulsion_loss.item(), on_epoch=True)
            self.logger('kmeans_loss_total_loss', total_loss.item(), on_epoch=True)

        return total_loss
