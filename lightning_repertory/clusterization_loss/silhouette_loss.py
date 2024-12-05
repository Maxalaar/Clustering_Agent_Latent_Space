import torch
import torch.nn as nn
import torch.nn.functional as F


class SilhouetteLoss(nn.Module):
    def __init__(self, logger=None):
        """
        Initializes the silhouette-based loss.
        """
        super(SilhouetteLoss, self).__init__()
        self.logger = logger

    def forward(self, embeddings, cluster_labels, **kwargs):
        """
        Computes the silhouette-based loss.

        Arguments:
        - embeddings: Tensor of shape (n_samples, embedding_dim), the embeddings of the points.
        - labels: Tensor of shape (n_samples,), the cluster labels (integers).

        Returns:
        - loss: Scalar value representing the silhouette-based loss.
        """
        unique_labels = torch.unique(cluster_labels)
        n_samples = embeddings.size(0)

        # Initialize intra-cluster and nearest-cluster distances
        a = torch.zeros(n_samples, device=embeddings.device)  # Mean intra-cluster distance
        b = torch.full((n_samples,), float('inf'),
                       device=embeddings.device)  # Mean distance to the nearest other cluster

        # Compute distances
        for label in unique_labels:
            mask_intra = cluster_labels == label
            mask_inter = cluster_labels != label

            # Points belonging to the current cluster
            cluster_points = embeddings[mask_intra]

            # Mean intra-cluster distance
            if cluster_points.size(0) > 1:
                a[mask_intra] = torch.mean(F.pairwise_distance(cluster_points.unsqueeze(1), cluster_points.unsqueeze(0)), dim=1)

            # Mean distance to the nearest other cluster
            other_points = embeddings[mask_inter]
            if other_points.size(0) > 0:
                # Compute pairwise distances and take the mean for each point in the current cluster
                pairwise_distances = F.pairwise_distance(cluster_points.unsqueeze(1), other_points.unsqueeze(0))
                mean_distances = torch.mean(pairwise_distances, dim=1)
                b[mask_intra] = mean_distances  # Assign the mean distances to b for the current cluster points

        # Compute silhouette scores for each point
        silhouette_scores = (b - a) / torch.max(a, b)
        silhouette_scores = 1 + silhouette_scores   #torch.nan_to_num(silhouette_scores, nan=0.0, posinf=0.0, neginf=0.0)

        # The loss is the negative mean silhouette score
        loss = -torch.mean(silhouette_scores)
        return loss
