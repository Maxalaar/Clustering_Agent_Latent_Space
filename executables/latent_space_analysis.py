import torch
from itertools import islice

from cuml import KMeans, TSNE

from configurations.structure.experimentation_configuration import ExperimentationConfiguration
from lightning.h5_data_module import H5DataModule
from lightning.surrogate_policy import SurrogatePolicy

from bokeh.plotting import figure, output_file, save
from bokeh.palettes import Viridis256


def latent_space_analysis(experimentation_configuration: ExperimentationConfiguration, model_checkpoint_path):
    data_module = H5DataModule(
        h5_file_path=experimentation_configuration.trajectory_dataset_file_path,
        input_dataset_name='observation',
        batch_size=50_000,
        chunk_size=300_000,
        number_workers=2,
    )
    data_module.setup()

    surrogate_policy: SurrogatePolicy = SurrogatePolicy.load_from_checkpoint(model_checkpoint_path)
    surrogate_policy.eval()
    embeddings_in_clustering_space = []

    with torch.no_grad():
        for i, batch in enumerate(islice(data_module.train_dataloader(), 1)):
            batch = batch.to(surrogate_policy.device)
            embeddings_in_clustering_space.append(surrogate_policy.projection_clustering_space(batch))

    embeddings_in_clustering_space = torch.cat(embeddings_in_clustering_space, dim=0)
    kmeans = KMeans(n_clusters=surrogate_policy.clusterization_loss.number_cluster)
    kmeans.fit(embeddings_in_clustering_space)
    cluster_labels = kmeans.predict(embeddings_in_clustering_space)

    tsne = TSNE(n_components=2)
    tsne_projection = tsne.fit_transform(embeddings_in_clustering_space)

    cluster_labels = cluster_labels.get().astype(int)
    min_value = cluster_labels.min()
    max_value = cluster_labels.max()
    normalized_values = (cluster_labels - min_value) / (max_value - min_value) * (len(Viridis256) - 1)
    colors = [Viridis256[int(val)] for val in normalized_values]

    plot = figure(title='t-SNE', width=800, height=600)
    plot.scatter(tsne_projection.get()[:, 0], tsne_projection.get()[:, 1], size=5, color=colors, alpha=0.6)
    output_file(experimentation_configuration.experimentation_storage_path / 'tsne.html')
    save(plot)


if __name__ == '__main__':
    import configurations.list_experimentation_configurations

    model_checkpoint_path = '/home/malaarabiou/Programming_Projects/Pycharm_Projects/Clustering_Agent_Latent_Space/experiments/pong_survivor_tow_balls/surrogate_policy/version_11/checkpoints/epoch=888-step=50659.ckpt'
    latent_space_analysis(configurations.list_experimentation_configurations.pong_survivor_two_balls, model_checkpoint_path)
