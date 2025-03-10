import argparse
import os
import shutil

import numpy as np
import ray
import torch

from typing import List
from pathlib import Path
from ray.tune.registry import _Registry
from configurations.structure.experimentation_configuration import ExperimentationConfiguration
from environments.register_environments import register_environments
from lightning_repertory.surrogate_policy import SurrogatePolicy

from utilities.get_configuration_class import get_configuration_class
from utilities.get_last_directory_name import get_last_directory_name
from utilities.latent_space_analysis.compare_clustering_between_surrogate_policies import \
    compare_clustering_between_surrogate_policies
from utilities.latent_space_analysis.create_latent_space_analysis_directories import \
    create_latent_space_analysis_directories
from utilities.latent_space_analysis.distribute_actions_by_cluster import distribution_actions_by_cluster
from utilities.latent_space_analysis.get_data import get_data
from utilities.latent_space_analysis.kmeans_latent_space import kmeans_latent_space
from utilities.latent_space_analysis.latent_space_projection_2d import latent_space_projection_2d
from utilities.latent_space_analysis.projection_clusterization_latent_space import \
    projection_clusterization_latent_space
from utilities.latent_space_analysis.representation_clusters import representation_clusters
from utilities.latent_space_analysis.train_observations_actions_decision_tree import \
    train_observations_actions_decision_tree
from utilities.latent_space_analysis.train_observations_clusters_decision_tree import \
    train_observations_clusters_decision_tree
from utilities.load_surrogate_policies import load_surrogate_policies
from utilities.process_surrogate_policy_checkpoint_paths import process_surrogate_policy_checkpoint_paths
from utilities.save_dictionary_to_file import save_dictionary_to_file


def latent_space_analysis(
        experimentation_configuration: ExperimentationConfiguration,
        trajectory_dataset_path: Path,
        surrogate_policy_checkpoint_paths,
        device=torch.device('cuda'),
):

    if not ray.is_initialized():
        ray.init()

    register_environments()
    environment_creator = _Registry().get('env_creator', experimentation_configuration.environment_name)
    environment = environment_creator(experimentation_configuration.environment_configuration)

    save_directory_name = get_last_directory_name(surrogate_policy_checkpoint_paths)
    surrogate_policy_checkpoint_paths: List[Path] = process_surrogate_policy_checkpoint_paths(surrogate_policy_checkpoint_paths)
    surrogate_policies = load_surrogate_policies(surrogate_policy_checkpoint_paths)
    latent_space_analysis_storage_paths = create_latent_space_analysis_directories(surrogate_policy_checkpoint_paths, experimentation_configuration)

    surrogate_policies_cluster_labels = []
    observations_cluster_accuracy_values = []

    observations, actions = get_data(
        dataset_names=['observations', 'actions'],
        data_number=experimentation_configuration.latent_space_analysis_configuration.number_data,
        trajectory_dataset_file_path=trajectory_dataset_path / 'trajectory_dataset.h5',
        device=device,
    )

    observations_with_rending, renderings, actions_with_rending = get_data(
        dataset_names=['observations', 'renderings', 'actions'],
        data_number=experimentation_configuration.latent_space_analysis_configuration.number_data_with_rending,
        trajectory_dataset_file_path=trajectory_dataset_path / 'trajectory_dataset_with_rending.h5',
        device=device,
    )

    for latent_space_analysis_storage_path, surrogate_policy in zip(latent_space_analysis_storage_paths, surrogate_policies):
        embeddings = projection_clusterization_latent_space(
            observations=observations,
            surrogate_policy=surrogate_policy,
        )
        cluster_labels, kmeans = kmeans_latent_space(
            embeddings=embeddings,
            number_cluster=surrogate_policy.clusterization_function.number_cluster,
            save_path=latent_space_analysis_storage_path,
        )
        surrogate_policies_cluster_labels.append(cluster_labels)
        latent_space_projection_2d(
            embeddings=embeddings,
            cluster_labels=cluster_labels,
            save_path=latent_space_analysis_storage_path,
            number_data=experimentation_configuration.latent_space_analysis_configuration.number_data_projection_2d,
        )
        observations_cluster_accuracy_values_one_policy = train_observations_clusters_decision_tree(
            observations=observations,
            cluster_labels=cluster_labels,
            feature_names=getattr(environment, 'observation_labels', None),
            save_path=latent_space_analysis_storage_path,
            tree_max_depth_observations_to_all_clusters=experimentation_configuration.latent_space_analysis_configuration.tree_max_depth_observations_to_all_clusters,
            tree_max_depth_observations_to_cluster=experimentation_configuration.latent_space_analysis_configuration.tree_max_depth_observations_to_cluster,
        )
        observations_cluster_accuracy_values = observations_cluster_accuracy_values + observations_cluster_accuracy_values_one_policy

        distribution_actions_by_cluster(
            observations=observations,
            cluster_labels=cluster_labels,
            actions=actions,
            save_path=latent_space_analysis_storage_path,
            class_names=getattr(environment, 'action_labels', None),
        )
        train_observations_actions_decision_tree(
            observations=observations,
            actions=actions,
            cluster_labels=cluster_labels,
            feature_names=getattr(environment, 'observation_labels', None),
            class_names=getattr(environment, 'action_labels', None),
            save_path=latent_space_analysis_storage_path,
            tree_max_depth_observations_to_actions=experimentation_configuration.latent_space_analysis_configuration.tree_max_depth_observations_to_actions,
        )

        representation_clusters(
            observations=observations_with_rending,
            renderings=renderings,
            save_path=latent_space_analysis_storage_path,
            surrogate_policy=surrogate_policy,
            clusterization_model=kmeans,
        )

    information = {}

    if len(surrogate_policies) > 0:
        information['clustering_between_surrogate_policies'] = compare_clustering_between_surrogate_policies(
            surrogate_policies_cluster_labels=surrogate_policies_cluster_labels,
        )

    information['observations_clusters_decision_tree'] = {
        'observations_clusters_decision_tree_mean': np.array(observations_cluster_accuracy_values).mean(),
        'observations_clusters_decision_tree_standard_deviation': np.array(observations_cluster_accuracy_values).std()
    }

    for key, value in information.items():
        print(f"{key}: {value}")

    if save_directory_name:
        save_dictionary_to_file(
            dictionary=information,
            name='latent_space_analysis_information',
            path=experimentation_configuration.latent_space_analysis_storage_path / save_directory_name,
        )



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyse of the clustering latent space for explainability.')
    parser.add_argument(
        '--experimentation_configuration_file',
        type=str,
        help="The path of the experimentation configuration file (e.g., './configurations/experimentation/cartpole.py')"
    )

    parser.add_argument(
        '--trajectory_dataset_path',
        type=str,
        help="The path of trajectory dataset directory (e.g., './experiments/cartpole/datasets/base/')"
    )

    parser.add_argument(
        '--surrogate_policy_checkpoint_paths',
        type=str,
        nargs='+',
        help="Path(s) to the policy checkpoint (e.g., './experiments/cartpole/surrogate_policy/base/version_[...]/checkpoints/[...].ckpt' or a directory containing .ckpt files)"
    )

    arguments = parser.parse_args()
    configuration_class = get_configuration_class(arguments.experimentation_configuration_file)

    trajectory_dataset_path = Path(arguments.trajectory_dataset_path)
    if not trajectory_dataset_path.is_absolute():
        trajectory_dataset_path = Path.cwd() / trajectory_dataset_path

    for path in arguments.surrogate_policy_checkpoint_paths:
        latent_space_analysis(configuration_class, trajectory_dataset_path, Path(path))
