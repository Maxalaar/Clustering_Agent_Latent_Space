from turtledemo.sorting_animate import qsort
from typing import Optional, Dict

import ray
import warnings
import h5py
import numpy as np
from pathlib import Path

from ray.rllib import BaseEnv
from ray.rllib.algorithms import Algorithm, AlgorithmConfig, PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation.episode_v2 import EpisodeV2

from configurations.structure.experimentation_configuration import ExperimentationConfiguration
from environments.register_environments import register_environments
from rllib.register_architectures import register_architectures
from rllib.find_best_checkpoints_path import find_best_checkpoints_path


@ray.remote
class H5FileHandler:
    def __init__(self, path_file: Path):
        path_file.parent.mkdir(parents=True, exist_ok=True)
        self.file = h5py.File(path_file, 'a')

    def get_last_episode_id(self):
        if 'episode_id' in self.file:
            return self.file['episode_id'][-1]
        else:
            return -1

    def write(self, episodes_id, dataset_dictionary: Dict[str, np.ndarray]):
        global_episode_id = episodes_id + self.get_last_episode_id() + 1
        dataset_dictionary['episodes_id'] = global_episode_id

        for dataset_name in dataset_dictionary.keys():
            data = dataset_dictionary[dataset_name]

            if dataset_name in self.file:
                dataset = self.file[dataset_name]
                dataset.resize((dataset.shape[0] + data.shape[0]), axis=0)
                dataset[-data.shape[0]:] = data
            else:
                max_shape = (None,) + data.shape[1:]
                self.file.create_dataset(dataset_name, data=data, maxshape=max_shape)


def trajectory_dataset_generation(experimentation_configuration: ExperimentationConfiguration):
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    ray.init(local_mode=experimentation_configuration.ray_local_mode)
    register_environments()
    register_architectures()

    save_rendering: bool = experimentation_configuration.trajectory_dataset_generation_configuration.save_rendering
    if save_rendering:
        path_file: Path = experimentation_configuration.trajectory_dataset_with_rending_file_path
    else:
        path_file: Path = experimentation_configuration.trajectory_dataset_file_path
    h5_file_handler = H5FileHandler.remote(path_file)

    class SaveTrajectoryCallback(DefaultCallbacks):
        def __init__(self):
            self.save_rendering: bool = experimentation_configuration.trajectory_dataset_generation_configuration.save_rendering
            self.h5_file_handler: H5FileHandler = h5_file_handler

            self.rendering_by_episode_id: dict = {}
            self.image_compression_function = experimentation_configuration.trajectory_dataset_generation_configuration.image_compression_function
            self.image_compression_configuration = experimentation_configuration.trajectory_dataset_generation_configuration.image_compression_configuration

        def on_episode_created(
                self,
                episode: EpisodeV2,
                **kwargs,
        ) -> None:
            if self.save_rendering:
                self.rendering_by_episode_id[episode.episode_id] = []

        def on_episode_step(
                self,
                episode: EpisodeV2,
                env_index: int,
                base_env: BaseEnv,
                **kwargs,
        ) -> None:
            if self.save_rendering:
                rending = base_env.get_sub_environments()[env_index].render()

                if self.image_compression_function is not None:
                    rending = self.image_compression_function(image=rending, **self.image_compression_configuration)

                self.rendering_by_episode_id[episode.episode_id].append(rending)

        def on_sample_end(
                self,
                samples,
                env_runner=None,
                metrics_logger=None,
                **kwargs,
        ):
            unique_values, indices = np.unique(samples['default_policy']['eps_id'], return_index=True)
            sorted_indices = np.argsort(indices)
            ordered_unique_values = unique_values[sorted_indices]
            local_episode_id = np.zeros_like(samples['default_policy']['eps_id'])
            for i, value in enumerate(ordered_unique_values):
                local_episode_id[samples['default_policy']['eps_id'] == value] = i

            dataset_dictionary = {
                'episode_current_timestep': samples['default_policy']['t'],
                'observation': samples['default_policy']['obs'],
                'action_distribution_inputs': samples['default_policy']['action_dist_inputs'],
                'action': samples['default_policy']['actions'],
                'prediction_value_function': samples['default_policy']['vf_preds'],
                'value_bootstrapped': samples['default_policy']['values_bootstrapped'],
                'real_value': samples['default_policy']['value_targets'],
            }

            if self.save_rendering:
                rendering = []
                for episode_id in ordered_unique_values:
                    rendering.extend(self.rendering_by_episode_id[episode_id])
                    del self.rendering_by_episode_id[episode_id]
                dataset_dictionary['rendering'] = np.stack(rendering)

            self.h5_file_handler.write.remote(local_episode_id, dataset_dictionary)

    best_checkpoints_path: Path = find_best_checkpoints_path(experimentation_configuration)
    algorithm: Algorithm = Algorithm.from_checkpoint(
        path=str(best_checkpoints_path),
        trainable=experimentation_configuration.reinforcement_learning_configuration.algorithm_name,
    )
    algorithm_configuration: AlgorithmConfig = algorithm.config.copy(copy_frozen=False)
    del algorithm

    if experimentation_configuration.trajectory_dataset_generation_configuration.save_rendering:
        algorithm_configuration.env_config.update({'render_mode': 'rgb_array'})
    algorithm_configuration.learners(num_learners=0)
    algorithm_configuration.training(
        train_batch_size=experimentation_configuration.trajectory_dataset_generation_configuration.minimal_steps_per_iteration,
    )
    if type(algorithm_configuration) is PPOConfig:
        algorithm_configuration: PPOConfig
        algorithm_configuration.training(
            mini_batch_size_per_learner=experimentation_configuration.trajectory_dataset_generation_configuration.minimal_steps_per_iteration,
            sgd_minibatch_size=experimentation_configuration.trajectory_dataset_generation_configuration.minimal_steps_per_iteration,
        )

    algorithm_configuration.env_runners(
        explore=False,
        num_env_runners=experimentation_configuration.trajectory_dataset_generation_configuration.number_environment_runners,
        num_envs_per_env_runner=experimentation_configuration.trajectory_dataset_generation_configuration.number_environment_per_environment_runners,
        num_cpus_per_env_runner=experimentation_configuration.trajectory_dataset_generation_configuration.number_cpus_per_environment_runners,
        num_gpus_per_env_runner=experimentation_configuration.trajectory_dataset_generation_configuration.number_gpus_per_environment_runners,
    )

    algorithm_configuration.callbacks(SaveTrajectoryCallback)
    algorithm: Algorithm = algorithm_configuration.build()

    algorithm.restore(str(best_checkpoints_path))

    def sample_for_trajectory_dataset_generation(worker):
        for _ in range(experimentation_configuration.trajectory_dataset_generation_configuration.number_iterations):
            worker.sample()

    algorithm.env_runner_group.foreach_worker(sample_for_trajectory_dataset_generation, local_env_runner=False)


if __name__ == '__main__':
    import configurations.list_experimentation_configurations

    trajectory_dataset_generation(configurations.list_experimentation_configurations.ant)
