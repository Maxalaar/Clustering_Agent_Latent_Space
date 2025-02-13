import numpy as np
import torch
from pathlib import Path
from lightning_repertory.surrogate_policy import SurrogatePolicy
from utilities.latent_space_analysis.projection_clusterization_latent_space import projection_clusterization_latent_space


def representation_actions_by_cluster(
        latent_space_analysis_storage_path: Path,
        trajectory_dataset_with_rending_file_path: Path,
        surrogate_policy: SurrogatePolicy,
        clusterization_model,
        device=torch.device('cpu'),
):
    observations, renderings = get_observations_with_rending(
        trajectory_dataset_with_rending_file_path=trajectory_dataset_with_rending_file_path,
        device=device,
    )
    embeddings = projection_clusterization_latent_space(
        observations=observations,
        surrogate_policy=surrogate_policy,
    )
    cluster_labels = clusterization_model.predict(embeddings)
    unique_cluster_labels = np.unique(cluster_labels)