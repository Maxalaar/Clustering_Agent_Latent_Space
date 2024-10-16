import os

import ray
import warnings
import h5py
import fcntl
import hashlib
import numpy as np
from pathlib import Path
from ray.rllib.algorithms import Algorithm, AlgorithmConfig
from configurations.structure.experimentation_configuration import ExperimentationConfiguration
from environments.register_environments import register_environments
from models.architectures.rienforcement.register_architectures import register_architectures
from utilities.find_best_checkpoints_path import find_best_checkpoints_path
from ray.rllib.algorithms.callbacks import DefaultCallbacks


from utilities.register_information_environment_creator import register_information_environment_creator


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


def trajectory_dataset_generation(experimentation_configuration: ExperimentationConfiguration, save_rendering: bool = False):
    class SaveTrajectoryCallback(DefaultCallbacks):
        def __init__(self):
            self.path_file: Path = experimentation_configuration.trajectory_dataset_file

        def on_sample_end(
                self,
                samples,
                env_runner=None,
                metrics_logger=None,
                **kwargs,
        ):
            with SystemMutex('critical-save-section'):
                self.path_file.parent.mkdir(parents=True, exist_ok=True)
                self.increment_index(len(samples['default_policy']) - 1)
                self.save('observation', samples['default_policy']['obs'][:-1])
                self.save('action', np.stack([info['action'] for info in samples['default_policy']['infos'][1:]]))

                self.save('prediction_value_function', samples['default_policy']['vf_preds'][:-1])
                self.save('value_bootstrapped', samples['default_policy']['values_bootstrapped'][:-1])
                self.save('real_value', samples['default_policy']['value_targets'][:-1])

        def increment_index(self, number_steps):
            with h5py.File(self.path_file, 'a') as h5_file:
                if 'episode_id' in h5_file:
                    last_episode_id = h5_file['episode_id'][-1, 0]
                else:
                    last_episode_id = -1
            h5_file.close()

            current_episode_id = last_episode_id + 1
            episode_id = np.full((number_steps, 1), current_episode_id)
            self.save('episode_id', episode_id)

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
    ray.init(local_mode=False)
    register_environments()
    register_architectures()

    best_checkpoints_path: Path = find_best_checkpoints_path(experimentation_configuration)
    algorithm: Algorithm = Algorithm.from_checkpoint(
        path=str(best_checkpoints_path),
        trainable=experimentation_configuration.reinforcement_learning_configuration.algorithm,
    )
    algorithm_configuration: AlgorithmConfig = algorithm.config.copy(copy_frozen=False)
    del algorithm

    information_environment_name = register_information_environment_creator(
        environment_name=algorithm_configuration.env,
        save_rendering=save_rendering,
    )

    algorithm_configuration.environment(env=information_environment_name)
    if save_rendering:
        algorithm_configuration.env_config.update({'render_mode': 'rgb_array'})
    algorithm_configuration.learners(num_learners=0)
    algorithm_configuration.env_runners(
        num_env_runners=0,
        num_envs_per_env_runner=experimentation_configuration.trajectory_dataset_generation_configuration.number_environment_per_environment_runners,
        num_cpus_per_env_runner=experimentation_configuration.trajectory_dataset_generation_configuration.number_cpus_per_environment_runners,
        num_gpus_per_env_runner=experimentation_configuration.trajectory_dataset_generation_configuration.number_gpus_per_environment_runners,
    )
    algorithm_configuration.evaluation(
        evaluation_interval=1,
        evaluation_num_env_runners=experimentation_configuration.trajectory_dataset_generation_configuration.number_environment_runners,
        evaluation_duration=experimentation_configuration.trajectory_dataset_generation_configuration.minimal_number_episodes,
        evaluation_parallel_to_training=False,
    )
    algorithm_configuration.callbacks(SaveTrajectoryCallback)
    algorithm: Algorithm = algorithm_configuration.build()

    algorithm.restore(str(best_checkpoints_path))
    algorithm.evaluate()


if __name__ == '__main__':
    from configurations.experimentation.cartpole import cartpole
    from configurations.experimentation.lunar_lander import lunar_lander
    from configurations.experimentation.bipedal_walker import bipedal_walker
    from configurations.experimentation.ant import ant
    from configurations.experimentation.pong_survivor_two_balls import pong_survivor_two_balls

    trajectory_dataset_generation(cartpole)
