import torch
import torch.nn as nn
from cuml import KMeans
from cuml.metrics.cluster.silhouette_score import cython_silhouette_score


class Kmeans(nn.Module):
    def __init__(
            self,
            number_cluster: int,
            logger=None,
            number_points_for_silhouette_score: int = 2000,
    ):
        super(Kmeans, self).__init__()
        self.number_cluster = number_cluster
        self.logger = logger
        self.number_points_for_silhouette_score: int = number_points_for_silhouette_score
        self.kmeans = KMeans(n_clusters=self.number_cluster)

    def forward(self, embeddings):
        device = embeddings.device
        cluster_labels = torch.tensor(self.kmeans.fit_predict(embeddings.detach()), device=device)
        centroids = torch.tensor(self.kmeans.cluster_centers_, device=device)

        if self.number_points_for_silhouette_score is not None:
            indices = torch.randperm(embeddings.size(0))[:self.number_points_for_silhouette_score]
            silhouette_score = cython_silhouette_score(X=embeddings[indices].detach(), labels=cluster_labels[indices].detach())
            if self.logger is not None:
                self.logger('silhouette_score', silhouette_score, on_epoch=True)

        return {'cluster_labels': cluster_labels, 'centroids': centroids}
