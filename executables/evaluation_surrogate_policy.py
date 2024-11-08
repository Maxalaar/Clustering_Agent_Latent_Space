import warnings
from pathlib import Path

import ray
from ray.rllib.algorithms import Algorithm, AlgorithmConfig

from configurations.structure.experimentation_configuration import ExperimentationConfiguration
from environments.register_environments import register_environments
from rllib.architectures.lightning import LightningModel
from rllib.find_best_checkpoints_path import find_best_checkpoints_path
from rllib.register_architectures import register_architectures


def evaluation_surrogate_policy(experimentation_configuration: ExperimentationConfiguration, model_checkpoint_path):
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    ray.init(local_mode=False)
    register_environments()
    register_architectures()

    best_checkpoints_path: Path = find_best_checkpoints_path(experimentation_configuration)
    algorithm: Algorithm = Algorithm.from_checkpoint(
        path=str(best_checkpoints_path),
        trainable=experimentation_configuration.reinforcement_learning_configuration.algorithm_name,
    )
    algorithm_configuration: AlgorithmConfig = algorithm.config.copy(copy_frozen=False)
    del algorithm

    algorithm_configuration.env_runners(
        num_env_runners=5,
        num_gpus_per_env_runner=1/5,
        num_cpus_per_env_runner=1,
    )
    algorithm_configuration.training(
        model={
            'custom_model': LightningModel,
            'custom_model_config': {
                'checkpoint_path': model_checkpoint_path,
                'use_gpu': algorithm_configuration.num_gpus_per_env_runner > 0
            },
        }
    )
    algorithm_configuration.evaluation(
        evaluation_duration=1000,
    )
    algorithm = algorithm_configuration.build()
    information = algorithm.evaluate()
    print(information['env_runners']['episode_reward_mean'])


if __name__ == '__main__':
    import configurations.list_experimentation_configurations

    model_checkpoint_path = '/home/malaarabiou/Programming_Projects/Pycharm_Projects/Clustering_Agent_Latent_Space/experiments/pong_survivor_tow_balls/surrogate_policy/version_1/checkpoints/epoch=2887-step=958691.ckpt'
    evaluation_surrogate_policy(configurations.list_experimentation_configurations.pong_survivor_two_balls, model_checkpoint_path)

