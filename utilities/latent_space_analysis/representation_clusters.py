import os

import numpy as np
import torch
from pathlib import Path
from lightning_repertory.surrogate_policy import SurrogatePolicy
from utilities.latent_space_analysis.projection_clusterization_latent_space import projection_clusterization_latent_space
from PIL import Image


def representation_clusters(
        observations,
        renderings,
        latent_space_analysis_storage_path: Path,
        surrogate_policy: SurrogatePolicy,
        clusterization_model,
):
    embeddings = projection_clusterization_latent_space(
        observations=observations,
        surrogate_policy=surrogate_policy,
    )
    cluster_labels = clusterization_model.predict(embeddings)
    unique_cluster_labels = np.unique(cluster_labels)

    for label in unique_cluster_labels:
        cluster_path: Path = latent_space_analysis_storage_path / ('cluster_' + str(label))
        os.makedirs(cluster_path, exist_ok=True)
        ixd_points_current_cluster = np.where(cluster_labels == label)[0]
        renderings_current_cluster = renderings[ixd_points_current_cluster.get()]

        for i in range(len(renderings_current_cluster)):
            image = Image.fromarray(renderings_current_cluster[i].cpu().numpy())
            image.save(cluster_path / f'image_{i}.png')