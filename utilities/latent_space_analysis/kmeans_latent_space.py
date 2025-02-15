from pathlib import Path
from typing import Optional

import torch
from cuml import KMeans


def kmeans_latent_space(
        embeddings: torch.Tensor,
        number_cluster: int,
        save_path: Optional[Path] = None,
        number_points_for_silhouette_score: int = 10_000,
):
    from cuml.metrics.cluster import silhouette_score
    kmeans = KMeans(n_clusters=number_cluster)
    kmeans.fit(embeddings)
    cluster_labels = torch.Tensor(kmeans.predict(embeddings)).int()

    indices = torch.randperm(embeddings.size(0))[:number_points_for_silhouette_score]
    silhouette_score = silhouette_score(X=embeddings[indices].detach(), labels=cluster_labels[indices].detach())
    information = 'Kmeans in latent space silhouette score : ' + str(silhouette_score)
    print(information)
    if save_path is not None:
        with open(save_path / 'information.txt', 'a') as file:
            file.write(information)

    return cluster_labels, kmeans