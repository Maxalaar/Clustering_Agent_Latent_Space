import argparse
import os
import shutil
import ray
import torch

from typing import List
from pathlib import Path
from ray.tune.registry import _Registry
from configurations.structure.experimentation_configuration import ExperimentationConfiguration
from environments.register_environments import register_environments
from lightning_repertory.surrogate_policy import SurrogatePolicy

from utilities.get_configuration_class import get_configuration_class
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


def latent_space_analysis(
        experimentation_configuration: ExperimentationConfiguration,
        trajectory_dataset_path: Path,
        surrogate_policy_checkpoint_paths: List[Path],
        device=torch.device('cuda'),
):
    ray.init()
    register_environments()
    environment_creator = _Registry().get('env_creator', experimentation_configuration.environment_name)
    environment = environment_creator(experimentation_configuration.environment_configuration)

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

    latent_space_analysis_storage_paths = []
    surrogate_policies = []
    for surrogate_policy_checkpoint_path in surrogate_policy_checkpoint_paths:

        latent_space_analysis_storage_path = experimentation_configuration.latent_space_analysis_storage_path / surrogate_policy_checkpoint_path.parents[2].name / surrogate_policy_checkpoint_path.parents[1].name
        if latent_space_analysis_storage_path.exists() and latent_space_analysis_storage_path.is_dir():
            shutil.rmtree(latent_space_analysis_storage_path)
        os.makedirs(latent_space_analysis_storage_path, exist_ok=True)
        latent_space_analysis_storage_paths.append(latent_space_analysis_storage_path)

        surrogate_policy: SurrogatePolicy = SurrogatePolicy.load_from_checkpoint(surrogate_policy_checkpoint_path)
        surrogate_policy.eval()
        surrogate_policies.append(surrogate_policy)

        information = 'Surrogate policy checkpoint path: ' + str(surrogate_policy_checkpoint_path) + '\n'
        print(information)
        with open(latent_space_analysis_storage_path / 'information.txt', 'a') as file:
            file.write(information)

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
        latent_space_projection_2d(
            embeddings=embeddings,
            cluster_labels=cluster_labels,
            save_path=latent_space_analysis_storage_path,
            number_data=10_000,
        )
        train_observations_clusters_decision_tree(
            observations=observations,
            cluster_labels=cluster_labels,
            feature_names=getattr(environment, 'observation_labels', None),
            save_path=latent_space_analysis_storage_path,
            tree_max_depth_observations_to_all_clusters=3,
            tree_max_depth_observations_to_cluster=2,
        )
        train_observations_actions_decision_tree(
            observations=observations,
            actions=actions,
            cluster_labels=cluster_labels,
            feature_names=getattr(environment, 'observation_labels', None),
            class_names=getattr(environment, 'action_labels', None),
            save_path=latent_space_analysis_storage_path,
        )
        representation_clusters(
            observations=observations_with_rending,
            renderings=renderings,
            latent_space_analysis_storage_path=latent_space_analysis_storage_path,
            surrogate_policy=surrogate_policy,
            clusterization_model=kmeans,
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
        help="The path of repository with the surrogate policy checkpoint (e.g., './experiments/cartpole/surrogate_policy/base/version_[...]/checkpoints/[...].ckpt')"
    )

    arguments = parser.parse_args()
    configuration_class = get_configuration_class(arguments.experimentation_configuration_file)

    trajectory_dataset_path = Path(arguments.trajectory_dataset_path)
    if not trajectory_dataset_path.is_absolute():
        trajectory_dataset_path = Path.cwd() / trajectory_dataset_path

    surrogate_policy_checkpoint_paths = []
    for surrogate_policy_checkpoint_path in arguments.surrogate_policy_checkpoint_paths:
        surrogate_policy_checkpoint_path = Path(surrogate_policy_checkpoint_path)
        if not surrogate_policy_checkpoint_path.is_absolute():
            surrogate_policy_checkpoint_path = Path.cwd() / surrogate_policy_checkpoint_path
        surrogate_policy_checkpoint_paths.append(surrogate_policy_checkpoint_path)

    latent_space_analysis(configuration_class, trajectory_dataset_path, surrogate_policy_checkpoint_paths)
