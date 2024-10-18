import os
from pathlib import Path
import warnings

import ray
from ray.rllib.algorithms import Algorithm, AlgorithmConfig

from configurations.structure.experimentation_configuration import ExperimentationConfiguration
from environments.register_environments import register_environments
from models.architectures.rllib.register_architectures import register_architectures
from utilities.find_best_checkpoints_path import find_best_checkpoints_path
from utilities.register_video_environment_creator import register_video_environment_creator


def delete_non_videos(path: Path):
    video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.mpeg', '.webm'}

    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)

        if os.path.isfile(file_path):
            _, ext = os.path.splitext(filename)

            if ext.lower() not in video_extensions:
                os.remove(file_path)


def video_episode_generation(experimentation_configuration: ExperimentationConfiguration):
    warnings.filterwarnings("ignore", category=DeprecationWarning)
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

    video_environment_name = register_video_environment_creator(
        environment_name=algorithm_configuration.env,
        video_episodes_storage_path=experimentation_configuration.video_episodes_storage_path,
    )

    algorithm_configuration.environment(
        env=video_environment_name
    )
    algorithm_configuration.env_config.update({'render_mode': 'rgb_array'})
    algorithm_configuration.learners(num_learners=0)
    algorithm_configuration.env_runners(
        num_env_runners=0,
        num_envs_per_env_runner=experimentation_configuration.video_episodes_generation_configuration.number_environment_per_environment_runners,
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

    algorithm.restore(str(best_checkpoints_path))
    algorithm.evaluate()
    # algorithm.eval_env_runner_group.stop()
    #
    # delete_non_videos(experimentation_configuration.video_episodes_storage_path)


if __name__ == '__main__':
    from configurations.experimentation.cartpole import cartpole
    from configurations.experimentation.lunar_lander import lunar_lander
    from configurations.experimentation.bipedal_walker import bipedal_walker
    from configurations.experimentation.ant import ant
    from configurations.experimentation.pong_survivor_two_balls import pong_survivor_two_balls

    video_episode_generation(bipedal_walker)
