import torch
from lightning_repertory.surrogate_policy import SurrogatePolicy


def projection_clusterization_latent_space(
        observations: torch.Tensor,
        surrogate_policy: SurrogatePolicy,
):
    with torch.no_grad():
        embeddings = surrogate_policy.projection_clustering_space(observations)
    return embeddings
