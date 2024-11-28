from pathlib import Path
import warnings

import ray
from ray.rllib.algorithms import Algorithm, AlgorithmConfig

from configurations.structure.experimentation_configuration import ExperimentationConfiguration
from environments.register_environments import register_environments
from rllib.find_best_checkpoint_path import find_best_checkpoint_path
from rllib.get_checkpoint_algorithm_configuration import get_checkpoint_algorithm_configuration
from rllib.register_video_environment_creator import register_video_environment_creator


def video_episode_generation(experimentation_configuration: ExperimentationConfiguration):
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    ray.init(local_mode=False)
    register_environments()

    best_checkpoints_path: Path = find_best_checkpoint_path(experimentation_configuration)
    algorithm_configuration = get_checkpoint_algorithm_configuration(best_checkpoints_path)

    video_environment_name = register_video_environment_creator(
        environment_name=algorithm_configuration.env,
        video_episodes_storage_path=experimentation_configuration.video_episodes_storage_path,
    )

    algorithm_configuration.environment(
        env=video_environment_name
    )
    algorithm_configuration.env_config.update({'render_mode': 'rgb_array'})
    algorithm_configuration.learners(
        num_learners=0,
        num_gpus_per_learner=0,
    )

    algorithm_configuration.env_runners(
        num_env_runners=0,
        num_envs_per_env_runner=experimentation_configuration.video_episodes_generation_configuration.number_environment_per_environment_runners,
        num_cpus_per_env_runner=experimentation_configuration.video_episodes_generation_configuration.number_cpus_per_environment_runners,
        num_gpus_per_env_runner=experimentation_configuration.video_episodes_generation_configuration.number_gpus_per_environment_runners,
    )

    algorithm_configuration.evaluation(
        evaluation_num_env_runners=experimentation_configuration.video_episodes_generation_configuration.number_environment_runners,
        evaluation_duration=experimentation_configuration.video_episodes_generation_configuration.minimal_number_videos,
        evaluation_parallel_to_training=False,
        evaluation_duration_unit='episodes',
        evaluation_sample_timeout_s=60*10,
    )

    algorithm: Algorithm = algorithm_configuration.build()

    algorithm.restore(str(best_checkpoints_path))
    information = algorithm.evaluate()

    print('The average reward on the evaluation is :' + str(information['env_runners']['episode_return_mean']))


if __name__ == '__main__':
    import configurations.list_experimentation_configurations

    video_episode_generation(configurations.list_experimentation_configurations.tetris)
