import argparse
from pathlib import Path
from typing import List

import numpy as np
import ray
from ray.rllib.core.rl_module import RLModuleSpec

from configurations.structure.experimentation_configuration import ExperimentationConfiguration
from environments.register_environments import register_environments
from rllib_repertory.architectures.lightning import Lightning
from rllib_repertory.find_best_checkpoint_path import find_best_reinforcement_learning_checkpoint_path
from rllib_repertory.get_checkpoint_algorithm_configuration import get_checkpoint_algorithm_configuration
from utilities.get_configuration_class import get_configuration_class
from utilities.get_last_directory_name import get_last_directory_name
from utilities.process_surrogate_policy_checkpoint_paths import process_surrogate_policy_checkpoint_paths
from utilities.save_dictionary_to_file import save_dictionary_to_file


def evaluation_surrogate_policy(
        experimentation_configuration: ExperimentationConfiguration,
        reinforcement_learning_path: Path,
        surrogate_policy_checkpoint_paths: Path,
):
    if not ray.is_initialized():
        ray.init()
    register_environments()

    save_directory_name = get_last_directory_name(surrogate_policy_checkpoint_paths)
    surrogate_policy_checkpoint_paths: List[Path] = process_surrogate_policy_checkpoint_paths(surrogate_policy_checkpoint_paths)

    best_checkpoints_path: Path = find_best_reinforcement_learning_checkpoint_path(reinforcement_learning_path)
    algorithm_configuration = get_checkpoint_algorithm_configuration(best_checkpoints_path)

    algorithm_configuration.learners(
        num_learners=0,
        num_gpus_per_learner=0,
        num_cpus_per_learner=1,
    )

    algorithm_configuration.env_runners(
        num_env_runners=0,
        num_gpus_per_env_runner=experimentation_configuration.surrogate_policy_evaluation_configuration.number_gpus_per_environment_runners,
        num_cpus_per_env_runner=experimentation_configuration.surrogate_policy_evaluation_configuration.number_cpus_per_environment_runners,
    )

    algorithm_configuration.evaluation(
        evaluation_num_env_runners=experimentation_configuration.surrogate_policy_evaluation_configuration.number_environment_runners,
        evaluation_duration=experimentation_configuration.surrogate_policy_evaluation_configuration.evaluation_duration,
        evaluation_sample_timeout_s=np.inf,
    )

    algorithm = algorithm_configuration.build()
    algorithm.restore(str(best_checkpoints_path))
    original_policy_evaluation_information = algorithm.evaluate()
    original_policy_return_mean = original_policy_evaluation_information['env_runners']['episode_return_mean']
    del algorithm

    surrogate_policy_return_means = []
    for surrogate_policy_checkpoint_path in surrogate_policy_checkpoint_paths:
        algorithm_configuration.rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=Lightning,
                model_config={
                    'checkpoint_path': surrogate_policy_checkpoint_path,
                },
            ),
        )
        algorithm = algorithm_configuration.build()
        surrogate_policy_evaluation_information = algorithm.evaluate()
        del algorithm
        surrogate_policy_return_means.append(surrogate_policy_evaluation_information['env_runners']['episode_return_mean'])

    surrogate_policies_ratio_means = np.array(surrogate_policy_return_means) / original_policy_return_mean
    surrogate_policies_reward_mean = np.array(surrogate_policy_return_means).mean()
    surrogate_policies_ratio_mean = surrogate_policies_ratio_means.mean()
    surrogate_policies_standard_deviation = surrogate_policies_ratio_means.std()

    print('Number of episode : ' + str(experimentation_configuration.surrogate_policy_evaluation_configuration.evaluation_duration))
    print('Original policies reward mean : ' + str(original_policy_return_mean))
    print('Surrogate policies reward mean : ' + str(surrogate_policies_reward_mean))
    print('Surrogate policies reward means : ' + str(surrogate_policy_return_means))
    print('Surrogate policies ratio means : ' + str(surrogate_policies_ratio_means))
    print('Surrogate policies ratio mean : ' + str(surrogate_policies_ratio_mean))
    print('Surrogate policies standard deviation : ' + str(surrogate_policies_standard_deviation))

    information = {
        'number_episode': experimentation_configuration.surrogate_policy_evaluation_configuration.evaluation_duration,
        'original_policies_return_mean': original_policy_return_mean,
        'surrogate_policies_return_mean': surrogate_policies_reward_mean,
        'surrogate_policies_return_means': surrogate_policy_return_means,
        'surrogate_policies_ratio_means': surrogate_policies_ratio_means.tolist(),
        'surrogate_policies_ratio_mean': surrogate_policies_ratio_mean,
        'surrogate_policies_standard_deviation': surrogate_policies_standard_deviation,
    }

    if save_directory_name:
        save_dictionary_to_file(
            dictionary=information,
            name='evaluation_surrogate_policy_information',
            path=experimentation_configuration.surrogate_policy_evaluation_storage_path / save_directory_name,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate surrogate policy.')
    parser.add_argument(
        '--experimentation_configuration_file',
        type=str,
        help="The path of the experimentation configuration file (e.g., './configurations/experimentation/cartpole.py')"
    )

    parser.add_argument(
        '--reinforcement_learning_path',
        type=str,
        help="The path of repository with the reinforcement learning checkpoint (e.g., './experiments/cartpole/reinforcement_learning/base')"
    )

    parser.add_argument(
        '--surrogate_policy_checkpoint_paths',
        type=str,
        nargs='+',
        help="Path(s) to the policy checkpoint (e.g., './experiments/cartpole/surrogate_policy/base/version_[...]/checkpoints/[...].ckpt' or a directory containing .ckpt files)"
    )

    arguments = parser.parse_args()
    configuration_class = get_configuration_class(arguments.experimentation_configuration_file)

    reinforcement_learning_path = Path(arguments.reinforcement_learning_path)
    if not reinforcement_learning_path.is_absolute():
        reinforcement_learning_path = Path.cwd() / reinforcement_learning_path

    for path in arguments.surrogate_policy_checkpoint_paths:
        evaluation_surrogate_policy(configuration_class, reinforcement_learning_path, Path(path))





