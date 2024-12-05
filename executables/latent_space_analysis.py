import os
from pathlib import Path
import shutil
from typing import Optional

import matplotlib
import numpy as np
import ray
import torch
from itertools import islice
from PIL import Image

from cuml import KMeans, TSNE, UMAP
from cuml.metrics.cluster.silhouette_score import cython_silhouette_score

from ray.tune.registry import _Registry

from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, DecisionTreeRegressor
from sklearn.metrics import accuracy_score

from configurations.structure.experimentation_configuration import ExperimentationConfiguration
from environments.register_environments import register_environments
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
        input_dataset_name='observations',
        output_dataset_name='actions',
        batch_size=10_000,
        number_mini_chunks=4,
        mini_chunk_size=50_000,
        number_workers=2,
    )
    data_module.setup()
    observations = []
    actions = []

    for i, batch in enumerate(islice(data_module.train_dataloader(), 100)):
        observations.append(batch[0])
        actions.append(batch[1])

    observations = torch.cat(observations, dim=0)
    actions = torch.cat(actions, dim=0)
    return observations.clone().detach().to(device), actions.clone().detach().to(device)


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
        save_path: Path,
        number_points_for_silhouette_score: int = 10_000,
):
    kmeans = KMeans(n_clusters=number_cluster)
    kmeans.fit(embeddings)
    cluster_labels = torch.Tensor(kmeans.predict(embeddings)).int()

    indices = torch.randperm(embeddings.size(0))[:number_points_for_silhouette_score]
    silhouette_score = cython_silhouette_score(X=embeddings[indices].detach(), labels=cluster_labels[indices].detach())
    information = 'Kmeans in latent space silhouette score : ' + str(silhouette_score)
    print(information)
    with open(save_path / 'information.txt', 'a') as file:
        file.write(information)

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


def train_observations_clusters_decision_tree(
        observations: torch.Tensor,
        cluster_labels: torch.Tensor,
        save_path: Path,
        feature_names: Optional[list] = None,
):
    save_path = save_path / 'observations_clusters_decision_trees'
    os.makedirs(save_path, exist_ok=True)

    observations = observations.cpu().numpy()
    cluster_labels = cluster_labels.cpu().numpy()

    class_names = []
    for label in np.unique(cluster_labels):
        class_names.append('cluster_' + str(label))

    x_train, x_test, y_train, y_test = train_test_split(observations, cluster_labels, test_size=0.2)

    decision_tree = DecisionTreeClassifier(max_depth=3)
    decision_tree.fit(x_train, y_train)
    predict_y_test = decision_tree.predict(x_test)
    accuracy_value = accuracy_score(y_test, predict_y_test)
    information = 'Decision tree (observations -> all clusters), max depth: ' + str(decision_tree.max_depth) + ', accuracy: ' + str(accuracy_value) + '\n'
    print(information)
    with open(save_path / 'information.txt', 'a') as file:
        file.write(information)

    plt.figure(figsize=(12, 12))
    plot_tree(decision_tree, filled=True, feature_names=feature_names, class_names=class_names)
    plt.savefig(save_path / 'all_clusters_decision_tree.png', bbox_inches='tight', dpi=300)

    for label in np.unique(cluster_labels):
        y_binary = (cluster_labels == label).astype(int)
        class_names = ['other_cluster', 'cluster_' + str(label)]
        random_over_sampler = RandomOverSampler(sampling_strategy=1.0)
        x_balance, y_balance = random_over_sampler.fit_resample(observations, y_binary)
        x_train, x_test, y_train, y_test = train_test_split(x_balance, y_balance, test_size=0.2)

        decision_tree = DecisionTreeClassifier(max_depth=2)
        decision_tree.fit(x_train, y_train)
        predict_y_test = decision_tree.predict(x_test)
        accuracy_value = accuracy_score(y_test, predict_y_test)
        information = 'Decision tree (observations -> cluster ' + str(label) + '), max depth: ' + str(decision_tree.max_depth) + ', accuracy: ' + str(accuracy_value) + '\n'
        print(information)
        with open(save_path / 'information.txt', 'a') as file:
            file.write(information)

        plt.figure(figsize=(12, 12))
        plot_tree(decision_tree, filled=True, feature_names=feature_names, class_names=class_names)
        plt.savefig(save_path / ('cluster_' + str(label) + '_decision_tree.png'), bbox_inches='tight', dpi=300)
        matplotlib.pyplot.close()


def train_observations_actions_decision_tree(
        observations: torch.Tensor,
        actions: torch.Tensor,
        cluster_labels: torch.Tensor,
        save_path: Path,
        feature_names: Optional[list] = None,
        class_names: Optional[list] = None,
):
    save_path = save_path / 'observations_actions_decision_trees'
    os.makedirs(save_path, exist_ok=True)

    observations = observations.cpu().numpy()
    is_convertible_to_int = torch.all(actions == actions.to(torch.int))

    if not is_convertible_to_int:
        return

    actions = actions.cpu().numpy()
    cluster_labels = cluster_labels.cpu().numpy()

    for label in np.unique(cluster_labels):
        indices = np.where(cluster_labels == label)[0]
        observations_current_cluster = observations[indices]
        actions_current_cluster = actions[indices]

        random_over_sampler = RandomOverSampler()
        x_balance, y_balance = random_over_sampler.fit_resample(observations_current_cluster, actions_current_cluster)
        x_train, x_test, y_train, y_test = train_test_split(x_balance, y_balance, test_size=0.2)

        if is_convertible_to_int:
            decision_tree = DecisionTreeClassifier(max_depth=2)
            decision_tree.fit(x_train, y_train)
        else:
            return
            # decision_tree = DecisionTreeRegressor()
            # decision_tree.fit(x_train, y_train)

        predict_y_test = decision_tree.predict(x_test)
        accuracy_value = accuracy_score(y_test, predict_y_test)
        information = 'Decision tree cluster ' + str(label) + ' (observations -> actions), max depth: ' + str(decision_tree.max_depth) + ', accuracy: ' + str(accuracy_value) + '\n'
        print(information)
        with open(save_path / 'information.txt', 'a') as file:
            file.write(information)

        plt.figure(figsize=(12, 12))
        plot_tree(decision_tree, filled=True, feature_names=feature_names, class_names=class_names)
        plt.savefig(save_path / ('cluster_' + str(label) + '_observations_actions_decision_tree.png'), bbox_inches='tight', dpi=300)
        matplotlib.pyplot.close()


def get_observations_with_rending(
        trajectory_dataset_with_rending_file_path: Path,
        device=torch.device('cpu'),
):
    display_h5_file_information(trajectory_dataset_with_rending_file_path)
    data_module = H5DataModule(
        h5_file_path=trajectory_dataset_with_rending_file_path,
        output_dataset_name='observations',
        input_dataset_name='renderings',
        batch_size=2000,
        number_mini_chunks=2,
        mini_chunk_size=3000,
        number_workers=2,
        shuffle=False,
    )
    data_module.setup()
    observations = []
    renderings = []

    for i, batch in enumerate(islice(data_module.train_dataloader(), 5)):
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


def latent_space_analysis(experimentation_configuration: ExperimentationConfiguration, surrogate_policy_checkpoint_path):
    if experimentation_configuration.latent_space_analysis_storage_path.exists() and experimentation_configuration.latent_space_analysis_storage_path.is_dir():
        shutil.rmtree(experimentation_configuration.latent_space_analysis_storage_path)
    os.makedirs(experimentation_configuration.latent_space_analysis_storage_path, exist_ok=True)

    ray.init()
    register_environments()
    environment_creator = _Registry().get('env_creator', experimentation_configuration.environment_name)
    environment = environment_creator(experimentation_configuration.environment_configuration)

    surrogate_policy: SurrogatePolicy = SurrogatePolicy.load_from_checkpoint(surrogate_policy_checkpoint_path)
    surrogate_policy.eval()
    information = 'Surrogate policy checkpoint path: ' + str(surrogate_policy_checkpoint_path) + '\n'
    print(information)
    with open(experimentation_configuration.latent_space_analysis_storage_path / 'information.txt', 'a') as file:
        file.write(information)

    observations, actions = get_observations(
        trajectory_dataset_file_path=experimentation_configuration.trajectory_dataset_file_path,
        device=surrogate_policy.device,
    )
    embeddings = projection_clusterization_latent_space(
        observations=observations,
        surrogate_policy=surrogate_policy,
    )
    cluster_labels, kmeans = kmeans_latent_space(
        embeddings=embeddings,
        number_cluster=surrogate_policy.clusterization_function.number_cluster,
        save_path=experimentation_configuration.latent_space_analysis_storage_path,
    )
    latent_space_projection_2d(
        embeddings=embeddings,
        cluster_labels=cluster_labels,
        save_path=experimentation_configuration.latent_space_analysis_storage_path,
        number_data=10_000,
    )
    train_observations_clusters_decision_tree(
        observations=observations,
        cluster_labels=cluster_labels,
        feature_names=getattr(environment, 'observation_labels', None),
        save_path=experimentation_configuration.latent_space_analysis_storage_path,
    )
    train_observations_actions_decision_tree(
        observations=observations,
        actions=actions,
        cluster_labels=cluster_labels,
        feature_names=getattr(environment, 'observation_labels', None),
        class_names=getattr(environment, 'action_labels', None),
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

    surrogate_policy_checkpoint_path = '/home/malaarabiou/Programming_Projects/Pycharm_Projects/Clustering_Agent_Latent_Space/experiments/pong_survivor_tow_balls/surrogate_policy/version_0/checkpoints/epoch=106-step=93862.ckpt'
    latent_space_analysis(configurations.list_experimentation_configurations.pong_survivor_two_balls, surrogate_policy_checkpoint_path)
