import warnings
from pathlib import Path

from ray.air import Result
from ray.tune import Tuner

from configurations.structure.experimentation_configuration import ExperimentationConfiguration


def find_best_checkpoints_path(
        experimentation_configuration: ExperimentationConfiguration,
):
    target_attribute: str = experimentation_configuration.reinforcement_learning_configuration.checkpoint_score_attribute
    mode: str = experimentation_configuration.reinforcement_learning_configuration.checkpoint_score_order

    tuner: Tuner = Tuner.restore(
        path=str(experimentation_configuration.reinforcement_learning_storage_path),
        trainable=experimentation_configuration.reinforcement_learning_configuration.algorithm,
    )

    result_grid = tuner.get_results()
    best_result: Result = result_grid.get_best_result(
        target_attribute,
        mode,
    )

    # result_grid._experiment_analysis.trials[0].run_metadata.checkpoint_manager.best_checkpoint_results
    if len(best_result.best_checkpoints) <= 0:
        raise ValueError('No checkpoints found in the results.')

    best_reward = best_result.metrics.get(target_attribute)

    if best_reward is None:
        warnings.warn('Warning: ' + str(target_attribute) + ' key not found in metrics.')
    else:
        print('Best ' + str(target_attribute) + ' used for checkpoint selection: ' + str(best_reward))

    path_checkpoint: Path = best_result.best_checkpoints[0][0].path
    return path_checkpoint
