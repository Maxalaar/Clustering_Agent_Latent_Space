from pathlib import Path
from typing import Union, Optional, Dict

import numpy as np
import h5py
import gymnasium as gym

import ray
from ray.rllib import BaseEnv, Policy
from ray.rllib.algorithms import Algorithm, AlgorithmConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.core.rl_module import RLModule
from ray.rllib.evaluation import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from ray.rllib.utils.typing import EpisodeType, PolicyID

from configurations.structure.experimentation_configuration import ExperimentationConfiguration
from environments.register_environments import register_environments
from utilities.path_best_checkpoints import path_best_checkpoints


class TrajectorySavingCallback(DefaultCallbacks):
    def __init__(self):
        super().__init__()

    def on_episode_step(
        self,
        *,
        episode: Union[EpisodeType, Episode, EpisodeV2],
        env_runner: Optional['EnvRunner'] = None,
        metrics_logger: Optional[MetricsLogger] = None,
        env: Optional[gym.Env] = None,
        env_index: int,
        rl_module: Optional[RLModule] = None,
        # Deprecate these args
        worker: Optional['EnvRunner'] = None,
        base_env: Optional[BaseEnv] = None,
        policies: Optional[Dict[PolicyID, Policy]] = None,
        **kwargs,
    ):
        episode: EpisodeV2
        pass

    def on_episode_end(
        self,
        *,
        episode: Union[EpisodeType, Episode, EpisodeV2],
        env_runner: Optional['EnvRunner'] = None,
        metrics_logger: Optional[MetricsLogger] = None,
        env: Optional[gym.Env] = None,
        env_index: int,
        rl_module: Optional[RLModule] = None,
        # Deprecate these args.
        worker: Optional['EnvRunner'] = None,
        base_env: Optional[BaseEnv] = None,
        policies: Optional[Dict[PolicyID, Policy]] = None,
        **kwargs,
    ):
        pass


def trajectory_dataset_generation(experimentation_configuration: ExperimentationConfiguration):
    ray.init(local_mode=True)
    register_environments()

    checkpoint_path: Path = path_best_checkpoints(experimentation_configuration)
    algorithm: Algorithm = Algorithm.from_checkpoint(
        path=str(checkpoint_path),
        trainable=experimentation_configuration.reinforcement_learning_configuration.algorithm,
    )
    algorithm_configuration: AlgorithmConfig = algorithm.config.copy(copy_frozen=False)
    del algorithm

    algorithm_configuration.learners(num_learners=0)
    algorithm_configuration.env_runners(
        num_env_runners=0,
        num_envs_per_worker=1,
        num_cpus_per_env_runner=1,
        num_gpus_per_env_runner=0,
    )
    algorithm_configuration.evaluation(
        evaluation_interval=1,
        evaluation_num_env_runners=1,
        evaluation_duration=5,
        evaluation_parallel_to_training=False,
    )
    algorithm_configuration.callbacks(TrajectorySavingCallback)
    algorithm: Algorithm = algorithm_configuration.build()

    algorithm.restore(str(checkpoint_path))
    algorithm.evaluate()


if __name__ == '__main__':
    from configurations.experimentation.cartpole import cartpole
    from configurations.experimentation.lunar_lander import lunar_lander
    from configurations.experimentation.bipedal_walker import bipedal_walker

    trajectory_dataset_generation(lunar_lander)
