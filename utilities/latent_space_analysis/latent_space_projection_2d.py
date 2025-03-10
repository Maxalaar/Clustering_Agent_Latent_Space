import torch
from pathlib import Path
from cuml import TSNE
from bokeh.plotting import figure, output_file, save
from bokeh.palettes import Viridis256
import matplotlib.pyplot as plt

def latent_space_projection_2d(embeddings: torch.Tensor, cluster_labels, save_path: Path, number_data: int):
    sample_indices = torch.randperm(embeddings.size(0))[:number_data]
    sample_embeddings = embeddings[sample_indices]
    sample_cluster_labels = cluster_labels[sample_indices]

    projector_2d = TSNE(n_components=2)
    projection_2d = projector_2d.fit_transform(sample_embeddings)

    min_value = sample_cluster_labels.min()
    max_value = sample_cluster_labels.max()
    normalized_values = (sample_cluster_labels - min_value) / (max_value - min_value) * (len(Viridis256) - 1)
    colors = [Viridis256[int(val)] for val in normalized_values]

    plot = figure(
        title='Two-Dimensional Projection of Clustered Latent Space',
        width=800,
        height=600,
        output_backend='svg'
    )
    plot.scatter(projection_2d.get()[:, 0], projection_2d.get()[:, 1], size=5, color=colors, alpha=0.6)

    output_file(save_path / 'projection_2d.html')
    save(plot)

    if isinstance(sample_cluster_labels, torch.Tensor):
        sample_cluster_labels = sample_cluster_labels.cpu().numpy()

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        projection_2d.get()[:, 0],
        projection_2d.get()[:, 1],
        c=sample_cluster_labels,
        cmap='viridis',
        s=5,
        alpha=0.6
    )
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_frame_on(False)
    plt.gca().axis('off')

    pdf_path = save_path / 'projection_2d.pdf'
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
