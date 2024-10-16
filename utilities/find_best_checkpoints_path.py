from pathlib import Path

from ray.air import Result
from ray.tune import Tuner

from configurations.structure.experimentation_configuration import ExperimentationConfiguration


def find_best_checkpoints_path(
        experimentation_configuration: ExperimentationConfiguration,
):
    tuner: Tuner = Tuner.restore(
        path=str(experimentation_configuration.reinforcement_learning_storage_path),
        trainable=experimentation_configuration.reinforcement_learning_configuration.algorithm,
    )

    result_grid = tuner.get_results()
    best_result: Result = result_grid.get_best_result()

    path_checkpoint: Path = best_result.best_checkpoints[0][0].path
    return path_checkpoint
