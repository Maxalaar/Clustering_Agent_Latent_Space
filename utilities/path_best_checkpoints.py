from pathlib import Path
from ray.tune import Tuner

from configurations.structure.experimentation_configuration import ExperimentationConfiguration


def path_best_checkpoints(experimentation_configuration: ExperimentationConfiguration):
    tuner = Tuner.restore(
        path=str(experimentation_configuration.reinforcement_learning_storage_path),
        trainable=experimentation_configuration.reinforcement_learning_configuration.algorithm,
    )
    result_grid = tuner.get_results()
    best_result = result_grid.get_best_result('evaluation/env_runners/episode_reward_mean', mode='max')

    if len(best_result.best_checkpoints) <= 0:
        raise ValueError("No checkpoints found in the results.")

    path_checkpoint: Path = best_result.best_checkpoints[0][0].path
    return path_checkpoint
