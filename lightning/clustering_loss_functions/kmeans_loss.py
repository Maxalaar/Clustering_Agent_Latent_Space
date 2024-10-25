import torch
from cuml import KMeans


def kmeans_loss(
        embeddings: torch.Tensor,
        number_cluster: int,
        logger=None,
):
    attraction_loss = torch.Tensor([0]).to(embeddings.device)
    repulsion_loss = torch.Tensor([0]).to(embeddings.device)

    kmeans = KMeans(n_clusters=number_cluster)
    kmeans_input = embeddings.detach()
    kmeans.fit(kmeans_input)
    cluster_labels = torch.tensor(kmeans.predict(kmeans_input)).to(embeddings.device)
    centroids = torch.tensor(kmeans.cluster_centers_).to(embeddings.device)
    distance_intra_cluster = 0

    for i in range(number_cluster):
        current_centroid = centroids[i].unsqueeze(dim=0)
        ixd_points_current_cluster = torch.nonzero(cluster_labels == i).squeeze(dim=1)
        points_current_cluster = embeddings[ixd_points_current_cluster]
        other_centroids = torch.cat([centroids[:i], centroids[i + 1:]])

        attraction_loss += torch.mean((torch.cdist(points_current_cluster, current_centroid)) ** 2)
        repulsion_loss += torch.mean((1 / (torch.cdist(points_current_cluster, other_centroids) + 1e-3)) ** 2)

        distance_intra_cluster += torch.mean(torch.norm(torch.cdist(points_current_cluster, current_centroid), dim=1)).item()

    total_loss = attraction_loss + repulsion_loss

    if logger:
        matrix_distance_centroids = torch.cdist(centroids, centroids).triu(diagonal=1)
        distance_centroids = matrix_distance_centroids[matrix_distance_centroids != 0]

        logger('average_distance_centroids_kmeans', torch.mean(distance_centroids).item(), on_epoch=True)
        logger('min_distance_centroids_kmeans', distance_centroids.min().item(), on_epoch=True)
        logger('max_distance_centroids_kmeans', distance_centroids.max().item(), on_epoch=True)
        logger('mean_distance_intra_cluster_kmeans', distance_intra_cluster, on_epoch=True)
        logger('attraction_loss_kmeans', attraction_loss.item(), on_epoch=True)
        logger('repulsion_loss_kmeans', repulsion_loss.item(), on_epoch=True)
        logger('total_loss_kmeans', total_loss.item(), on_epoch=True)

    return total_loss

