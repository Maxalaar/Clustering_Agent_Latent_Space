import os
from pathlib import Path
import shutil
from typing import Optional

import numpy as np
import torch
from itertools import islice
from PIL import Image

from cuml import KMeans, TSNE, UMAP
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

from configurations.structure.experimentation_configuration import ExperimentationConfiguration
from lightning.h5_data_module import H5DataModule
from lightning.surrogate_policy import SurrogatePolicy

from bokeh.plotting import figure, output_file, save
from bokeh.palettes import Viridis256

from utilities.display_h5_file_information import display_h5_file_information
import matplotlib.pyplot as plt


def get_observations(
        trajectory_dataset_file_path: Path,
        device=torch.device('cpu'),
):
    display_h5_file_information(trajectory_dataset_file_path)
    data_module = H5DataModule(
        h5_file_path=trajectory_dataset_file_path,
        input_dataset_name='observation',
        batch_size=10_000,
        number_mini_chunks=4,
        mini_chunk_size=50_000,
        number_workers=2,
    )
    data_module.setup()
    observations = []

    for i, batch in enumerate(islice(data_module.train_dataloader(), 100)):
        observations.append(batch)

    observations = torch.cat(observations, dim=0)
    return observations.clone().detach().to(device)


def projection_clusterization_latent_space(
        observations: torch.Tensor,
        surrogate_policy: SurrogatePolicy,
):
    with torch.no_grad():
        embeddings = surrogate_policy.projection_clustering_space(observations)
    return embeddings


def kmeans_latent_space(
        embeddings: torch.Tensor,
        number_cluster: int,
):
    kmeans = KMeans(n_clusters=number_cluster)
    kmeans.fit(embeddings)
    cluster_labels = torch.Tensor(kmeans.predict(embeddings)).int()
    return cluster_labels, kmeans


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


def train_decision_tree(
        observations: torch.Tensor,
        cluster_labels: torch.Tensor,
        save_path: Path,
        feature_names: Optional[list] = None,
):
    x_train, x_test, y_train, y_test = train_test_split(observations.cpu().numpy(), cluster_labels.cpu().numpy(), test_size=0.2)

    decision_tree = DecisionTreeClassifier(max_depth=3)
    decision_tree.fit(x_train, y_train)
    predict_y_test = decision_tree.predict(x_test)
    accuracy_value = accuracy_score(y_test, predict_y_test)
    information = 'Decision tree (observations -> cluster), accuracy: ' + str(accuracy_value) + '\n'
    print(information)
    with open(save_path / 'information.txt', 'a') as file:
        file.write(information)

    plt.figure(figsize=(12, 12))
    plot_tree(decision_tree, filled=True, feature_names=feature_names)
    plt.savefig(save_path / 'decision_tree.png', bbox_inches='tight', dpi=300)


def get_observations_with_rending(
        trajectory_dataset_with_rending_file_path: Path,
        device=torch.device('cpu'),
):
    display_h5_file_information(trajectory_dataset_with_rending_file_path)
    data_module = H5DataModule(
        h5_file_path=trajectory_dataset_with_rending_file_path,
        output_dataset_name='observation',
        input_dataset_name='rendering',
        batch_size=2000,
        number_mini_chunks=2,
        mini_chunk_size=3000,
        number_workers=2,
    )
    data_module.setup()
    observations = []
    renderings = []

    for i, batch in enumerate(islice(data_module.train_dataloader(), 100)):
        observations.append(batch[1])
        renderings.append(batch[0])

    observations = torch.cat(observations, dim=0)
    observations = observations.clone().detach().to(device)

    renderings = torch.cat(renderings, dim=0)
    renderings = renderings.clone().detach().to(device)

    return observations, renderings


def representation_clusters(
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

    for label in unique_cluster_labels:
        cluster_path: Path = latent_space_analysis_storage_path / ('cluster_' + str(label))
        os.makedirs(cluster_path, exist_ok=True)
        ixd_points_current_cluster = np.where(cluster_labels == label)[0]
        renderings_current_cluster = renderings[ixd_points_current_cluster.get()]

        for i in range(len(renderings_current_cluster)):
            image = Image.fromarray(renderings_current_cluster[i].cpu().numpy())
            image.save(cluster_path / f'image_{i}.png')


def latent_space_analysis(experimentation_configuration: ExperimentationConfiguration, model_checkpoint_path):
    if experimentation_configuration.latent_space_analysis_storage_path.exists() and experimentation_configuration.latent_space_analysis_storage_path.is_dir():
        shutil.rmtree(experimentation_configuration.latent_space_analysis_storage_path)
    os.makedirs(experimentation_configuration.latent_space_analysis_storage_path, exist_ok=True)

    surrogate_policy: SurrogatePolicy = SurrogatePolicy.load_from_checkpoint(model_checkpoint_path)
    surrogate_policy.eval()
    information = 'Surrogate policy checkpoint path: ' + str(model_checkpoint_path) + '\n'
    print(information)
    with open(experimentation_configuration.latent_space_analysis_storage_path / 'information.txt', 'a') as file:
        file.write(information)

    observations = get_observations(
        trajectory_dataset_file_path=experimentation_configuration.trajectory_dataset_file_path,
        device=surrogate_policy.device,
    )
    embeddings = projection_clusterization_latent_space(
        observations=observations,
        surrogate_policy=surrogate_policy,
    )
    cluster_labels, kmeans = kmeans_latent_space(
        embeddings=embeddings,
        number_cluster=surrogate_policy.clusterization_loss.number_cluster,
    )
    latent_space_projection_2d(
        embeddings=embeddings,
        cluster_labels=cluster_labels,
        save_path=experimentation_configuration.latent_space_analysis_storage_path,
        number_data=10_000,
    )
    train_decision_tree(
        observations=observations,
        cluster_labels=cluster_labels,
        # feature_names=[
        #     'ball_1_x', 'ball_1_y', 'ball_1_velocity_x', 'ball_1_velocity_y',
        #     'ball_2_x', 'ball_2_y', 'ball_2_velocity_x', 'ball_2_velocity_y',
        #     'paddle_x', 'paddle_y', 'time'
        # ],
        save_path=experimentation_configuration.latent_space_analysis_storage_path,
    )
    representation_clusters(
        latent_space_analysis_storage_path=experimentation_configuration.latent_space_analysis_storage_path,
        trajectory_dataset_with_rending_file_path=experimentation_configuration.trajectory_dataset_with_rending_file_path,
        surrogate_policy=surrogate_policy,
        clusterization_model=kmeans,
        device=surrogate_policy.device,
    )


if __name__ == '__main__':
    import configurations.list_experimentation_configurations

    model_checkpoint_path = '/home/malaarabiou/Programming_Projects/Pycharm_Projects/Clustering_Agent_Latent_Space/experiments/ant/surrogate_policy/version_0/checkpoints/epoch=114-step=17500.ckpt'
    latent_space_analysis(configurations.list_experimentation_configurations.ant, model_checkpoint_path)