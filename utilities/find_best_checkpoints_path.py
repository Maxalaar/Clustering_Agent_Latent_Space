from pathlib import Path

import ray
from ray.air import Result
from ray.tune import Tuner, ExperimentAnalysis

from configurations.structure.experimentation_configuration import ExperimentationConfiguration


def find_best_checkpoints_path(
        experimentation_configuration: ExperimentationConfiguration,
):
    analysis = ExperimentAnalysis(experimentation_configuration.reinforcement_learning_storage_path)
    metric = 'evaluation/env_runners/episode_reward_mean'

    best_checkpoint = analysis.get_best_checkpoint(
        trial=analysis.get_best_trial(metric=metric, mode='max'),
        metric=metric,
        mode='max'
    )

    return best_checkpoint.path
