import ray
import warnings
import h5py
import fcntl
import hashlib
import numpy as np
from pathlib import Path

from ray.rllib.algorithms import Algorithm, AlgorithmConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks

from configurations.structure.experimentation_configuration import ExperimentationConfiguration
from environments.register_environments import register_environments
from rllib.register_architectures import register_architectures
from rllib.find_best_checkpoints_path import find_best_checkpoints_path

from rllib.register_information_environment_creator import register_information_environment_creator


class SystemMutex:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        lock_id = hashlib.md5(self.name.encode('utf8')).hexdigest()
        self.fp = open(f'/tmp/.lock-{lock_id}.lck', 'wb')
        fcntl.flock(self.fp.fileno(), fcntl.LOCK_EX)

    def __exit__(self, _type, value, tb):
        fcntl.flock(self.fp.fileno(), fcntl.LOCK_UN)
        self.fp.close()


def trajectory_dataset_generation(experimentation_configuration: ExperimentationConfiguration):
    class SaveTrajectoryCallback(DefaultCallbacks):
        def __init__(self):
            self.path_file: Path = experimentation_configuration.trajectory_dataset_file_path
            self.save_rendering: bool = experimentation_configuration.trajectory_dataset_generation_configuration.save_rendering

        def on_sample_end(
                self,
                samples,
                env_runner=None,
                metrics_logger=None,
                **kwargs,
        ):
            with SystemMutex('critical-save-section'):
                unique_values, indices = np.unique(samples['default_policy']['eps_id'], return_index=True)
                sorted_indices = np.argsort(indices)
                ordered_unique_values = unique_values[sorted_indices]
                local_episode_id = np.zeros_like(samples['default_policy']['eps_id'])
                for i, value in enumerate(ordered_unique_values):
                    local_episode_id[samples['default_policy']['eps_id'] == value] = i

                self.path_file.parent.mkdir(parents=True, exist_ok=True)

                self.increment_episode_id(local_episode_id)
                self.save('episode_current_timestep', samples['default_policy']['t'])
                self.save('observation', samples['default_policy']['obs'])
                self.save('action_distribution_inputs', samples['default_policy']['action_dist_inputs'])
                self.save('action', samples['default_policy']['actions'])
                self.save('prediction_value_function', samples['default_policy']['vf_preds'])
                self.save('value_bootstrapped', samples['default_policy']['values_bootstrapped'])
                self.save('real_value', samples['default_policy']['value_targets'])

                if self.save_rendering:
                    self.increment_episode_id_with_rendering(local_episode_id)
                    self.save('rendering', np.stack([info['rendering'] for info in samples['default_policy']['infos'][1:]]))

        def increment_episode_id(self, local_episode_id):
            with h5py.File(self.path_file, 'a') as h5_file:
                if 'episode_id' in h5_file:
                    last_episode_id = h5_file['episode_id'][-1]
                else:
                    last_episode_id = -1
            h5_file.close()

            global_episode_id = local_episode_id + last_episode_id + 1
            self.save('episode_id', global_episode_id)

        def increment_episode_id_with_rendering(self, local_episode_id):
            with h5py.File(self.path_file, 'a') as h5_file:
                if 'episode_id_with_rendering' in h5_file:
                    last_episode_id = h5_file['episode_id_with_rendering'][-1]
                else:
                    last_episode_id = -1
            h5_file.close()

            global_episode_id = local_episode_id + last_episode_id + 1
            self.save('episode_id', global_episode_id)

        def save(self, dataset_name: str, data):
            with h5py.File(self.path_file, 'a') as h5file:
                if dataset_name in h5file:
                    dataset = h5file[dataset_name]
                    dataset.resize((dataset.shape[0] + data.shape[0]), axis=0)
                    dataset[-data.shape[0]:] = data
                else:
                    max_shape = (None,) + data.shape[1:]
                    h5file.create_dataset(dataset_name, data=data, maxshape=max_shape)

    warnings.filterwarnings('ignore', category=DeprecationWarning)
    ray.init(local_mode=experimentation_configuration.ray_local_mode)
    register_environments()
    register_architectures()

    best_checkpoints_path: Path = find_best_checkpoints_path(experimentation_configuration)
    algorithm: Algorithm = Algorithm.from_checkpoint(
        path=str(best_checkpoints_path),
        trainable=experimentation_configuration.reinforcement_learning_configuration.algorithm_name,
    )
    algorithm_configuration: AlgorithmConfig = algorithm.config.copy(copy_frozen=False)
    del algorithm

    information_environment_name = register_information_environment_creator(
        environment_name=algorithm_configuration.env,
        save_rendering=experimentation_configuration.trajectory_dataset_generation_configuration.save_rendering,
    )

    algorithm_configuration.environment(env=information_environment_name)
    if experimentation_configuration.trajectory_dataset_generation_configuration.save_rendering:
        algorithm_configuration.env_config.update({'render_mode': 'rgb_array'})
    algorithm_configuration.learners(num_learners=0)
    algorithm_configuration.training(
        train_batch_size=experimentation_configuration.trajectory_dataset_generation_configuration.minimal_steps_per_iteration,
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
    from configurations.experimentation.lunar_lander import lunar_lander

    trajectory_dataset_generation(lunar_lander)
