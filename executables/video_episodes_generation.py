import os
from pathlib import Path
import string
import random

import ray
import gymnasium
from ray.rllib.algorithms import Algorithm, AlgorithmConfig
from ray.tune.registry import _Registry, register_env

from configurations.structure.experimentation_configuration import ExperimentationConfiguration
from environments.register_environments import register_environments
from utilities.path_best_checkpoints import path_best_checkpoints


def delete_non_videos(path: Path):
    video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.mpeg', '.webm'}

    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)

        if os.path.isfile(file_path):
            _, ext = os.path.splitext(filename)

            if ext.lower() not in video_extensions:
                os.remove(file_path)


class RandomString:
    def __init__(self, length=10):
        self.length = length

    def __str__(self):
        characters = string.ascii_letters + string.digits
        return ''.join(random.choice(characters) for _ in range(self.length))


def video_episode_generation(experimentation_configuration: ExperimentationConfiguration):
    ray.init(local_mode=False)
    register_environments()

    checkpoint_path: Path = path_best_checkpoints(experimentation_configuration)
    algorithm: Algorithm = Algorithm.from_checkpoint(
        path=str(checkpoint_path),
        trainable=experimentation_configuration.reinforcement_learning_configuration.algorithm,
    )
    algorithm_configuration: AlgorithmConfig = algorithm.config.copy(copy_frozen=False)
    del algorithm

    video_environment_name = algorithm_configuration.env + 'Video'

    def video_environment_creator(configuration: dict):
        environment_creator = _Registry().get('env_creator', algorithm_configuration.env)
        random_string_instance = RandomString(12)
        environment = environment_creator(configuration)
        video_environment = gymnasium.wrappers.RecordVideo(
            env=environment,
            video_folder=str(experimentation_configuration.video_episodes_storage_path),
            name_prefix=random_string_instance,
            episode_trigger=lambda x: True,
            disable_logger=True,
        )
        return video_environment

    register_env(name=video_environment_name, env_creator=video_environment_creator)

    algorithm_configuration.environment(
        env=video_environment_name
    )
    algorithm_configuration.env_config.update({'render_mode': 'rgb_array'})
    algorithm_configuration.learners(num_learners=0)
    algorithm_configuration.env_runners(
        num_env_runners=0,
        num_envs_per_worker=experimentation_configuration.video_episodes_generation_configuration.number_environment_per_environment_runners,
        num_cpus_per_env_runner=experimentation_configuration.video_episodes_generation_configuration.number_cpus_per_environment_runners,
        num_gpus_per_env_runner=experimentation_configuration.video_episodes_generation_configuration.number_gpus_per_environment_runners,
    )
    algorithm_configuration.evaluation(
        evaluation_interval=1,
        evaluation_num_env_runners=experimentation_configuration.video_episodes_generation_configuration.number_environment_runners,
        evaluation_duration=experimentation_configuration.video_episodes_generation_configuration.minimal_number_videos,
        evaluation_parallel_to_training=False,
    )
    algorithm: Algorithm = algorithm_configuration.build()

    algorithm.restore(str(checkpoint_path))
    algorithm.evaluate()
    algorithm.eval_env_runner_group.stop()

    delete_non_videos(experimentation_configuration.video_episodes_storage_path)


if __name__ == '__main__':
    from configurations.experimentation.cartpole import cartpole
    from configurations.experimentation.lunar_lander import lunar_lander
    from configurations.experimentation.bipedal_walker import bipedal_walker

    video_episode_generation(bipedal_walker)
