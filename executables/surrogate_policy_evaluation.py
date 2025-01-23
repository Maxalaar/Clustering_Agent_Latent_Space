import argparse
from pathlib import Path

import ray
from ray.rllib.core.rl_module import RLModuleSpec

from configurations.structure.experimentation_configuration import ExperimentationConfiguration
from environments.register_environments import register_environments
from rllib_repertory.architectures.lightning import Lightning
from rllib_repertory.find_best_checkpoint_path import find_best_reinforcement_learning_checkpoint_path
from rllib_repertory.get_checkpoint_algorithm_configuration import get_checkpoint_algorithm_configuration
from utilities.get_configuration_class import get_configuration_class


def evaluation_surrogate_policy(
        experimentation_configuration: ExperimentationConfiguration,
        reinforcement_learning_path: Path,
        surrogate_policy_checkpoint_path: Path
):
    ray.init(local_mode=experimentation_configuration.ray_local_mode)
    register_environments()

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
    )

    algorithm = algorithm_configuration.build()
    algorithm.restore(str(best_checkpoints_path))
    original_policy_evaluation_information = algorithm.evaluate()
    del algorithm

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

    print('Original policy average reward on ' + str(experimentation_configuration.surrogate_policy_evaluation_configuration.evaluation_duration) + ' iterations : ' + str(original_policy_evaluation_information['env_runners']['episode_return_mean']))
    print('Surrogate policy average reward on ' + str(experimentation_configuration.surrogate_policy_evaluation_configuration.evaluation_duration) + ' iterations : ' + str(surrogate_policy_evaluation_information['env_runners']['episode_return_mean']))


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
        '--surrogate_policy_checkpoint_path',
        type=str,
        help="The path of repository with the surrogate policy checkpoint (e.g., './experiments/cartpole/surrogate_policy/base/version_[...]/checkpoints/[...].ckpt')"
    )

    arguments = parser.parse_args()
    configuration_class = get_configuration_class(arguments.experimentation_configuration_file)

    reinforcement_learning_path = Path(arguments.reinforcement_learning_path)
    if not reinforcement_learning_path.is_absolute():
        reinforcement_learning_path = Path.cwd() / reinforcement_learning_path

    surrogate_policy_checkpoint_path = Path(arguments.surrogate_policy_checkpoint_path)
    if not surrogate_policy_checkpoint_path.is_absolute():
        surrogate_policy_checkpoint_path = Path.cwd() / surrogate_policy_checkpoint_path

    evaluation_surrogate_policy(configuration_class, reinforcement_learning_path, surrogate_policy_checkpoint_path)





