import torch
from pathlib import Path
from cuml import TSNE
from bokeh.plotting import figure, output_file, save
from bokeh.palettes import Viridis256


def latent_space_projection_2d(
        embeddings: torch.Tensor,
        cluster_labels,
        save_path: Path,
        number_data: int,
):
    sample_indices = torch.randperm(embeddings.size(0))[:number_data]
    sample_embeddings = embeddings[sample_indices]
    sample_cluster_labels = cluster_labels[sample_indices]

    projector_2d = TSNE(
        n_components=2,
        # perplexity=100,
        # n_neighbors=60,
        # n_iter=10_000,
    )
    projection_2d = projector_2d.fit_transform(sample_embeddings)

    sample_cluster_labels = sample_cluster_labels
    min_value = sample_cluster_labels.min()
    max_value = sample_cluster_labels.max()
    normalized_values = (sample_cluster_labels - min_value) / (max_value - min_value) * (len(Viridis256) - 1)
    colors = [Viridis256[int(val)] for val in normalized_values]

    plot = figure(title='Two-Dimensional Projection of Clusterize Latent Space', width=800, height=600)
    plot.scatter(projection_2d.get()[:, 0], projection_2d.get()[:, 1], size=5, color=colors, alpha=0.6)
    output_file(save_path / 'projection_2d.html')
    save(plot)
