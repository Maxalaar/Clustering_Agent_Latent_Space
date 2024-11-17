from typing import List, Optional
from datetime import datetime
from pathlib import Path

from ray.train import Checkpoint
from ray.train._internal.session import _TrainingResult
from ray.tune import ExperimentAnalysis
from ray.tune.experiment import Trial

from configurations.structure.experimentation_configuration import ExperimentationConfiguration


def find_best_checkpoint_path(
        experimentation_configuration: ExperimentationConfiguration,
):
    analysis = ExperimentAnalysis(experimentation_configuration.reinforcement_learning_storage_path)
    metric = 'evaluation/env_runners/episode_return_mean'

    best_trial: Trial = analysis.get_best_trial(metric=metric, mode='max')

    best_checkpoint: Checkpoint = analysis.get_best_checkpoint(
        trial=best_trial,
        metric=metric,
        mode='max'
    )
    best_checkpoint_metric: Optional[dict] = None

    if best_checkpoint is None:
        raise ValueError("Error: 'best_checkpoint' is None. Ray couldn't find any checkpoint.")

    training_results: List[_TrainingResult] = best_trial.run_metadata.checkpoint_manager.best_checkpoint_results
    for training_result in training_results:
        if training_result.checkpoint.path == best_checkpoint.path:
            best_checkpoint_metric = training_result.metrics
            break

    creation_time = datetime.strptime(best_checkpoint_metric['date'], '%Y-%m-%d_%H-%M-%S')

    print('Best checkpoint is create on: ' + creation_time.strftime("On %B %d, %Y at %H:%M and %S seconds."))
    print('Best checkpoint reward mean in evaluation: ' + str(best_checkpoint_metric['evaluation']['env_runners']['episode_return_mean']))
    print('Best checkpoint path : ' + str(best_checkpoint.path))
    print()

    return Path(best_checkpoint.path)
