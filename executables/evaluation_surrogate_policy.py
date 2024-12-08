from pathlib import Path

import ray
from ray.rllib.core.rl_module import RLModuleSpec

from configurations.structure.experimentation_configuration import ExperimentationConfiguration
from environments.register_environments import register_environments
from rllib_repertory.architectures.lightning import Lightning
from rllib_repertory.find_best_checkpoint_path import find_best_checkpoint_path
from rllib_repertory.get_checkpoint_algorithm_configuration import get_checkpoint_algorithm_configuration


def evaluation_surrogate_policy(experimentation_configuration: ExperimentationConfiguration, model_checkpoint_path):
    ray.init(local_mode=experimentation_configuration.ray_local_mode)
    register_environments()

    best_checkpoints_path: Path = find_best_checkpoint_path(experimentation_configuration)
    algorithm_configuration = get_checkpoint_algorithm_configuration(best_checkpoints_path)

    algorithm_configuration.learners(
        num_learners=0,
        num_gpus_per_learner=0,
        num_cpus_per_learner=1,
    )

    algorithm_configuration.env_runners(
        num_env_runners=0,
        num_gpus_per_env_runner=0,
        num_cpus_per_env_runner=1,
    )

    algorithm_configuration.evaluation(
        evaluation_num_env_runners=10,
        evaluation_duration=100,
    )

    algorithm = algorithm_configuration.build()
    algorithm.restore(str(best_checkpoints_path))
    original_policy_evaluation_information = algorithm.evaluate()
    del algorithm

    algorithm_configuration.rl_module(
        rl_module_spec=RLModuleSpec(
            module_class=Lightning,
            model_config={
                'checkpoint_path': model_checkpoint_path,
                'use_gpu': False,
            },
        ),
    )
    algorithm = algorithm_configuration.build()
    surrogate_policy_evaluation_information = algorithm.evaluate()
    del algorithm

    print('Original policy average reward on the evaluation :' + str(original_policy_evaluation_information['env_runners']['episode_return_mean']))
    print('Surrogate policy average reward on the evaluation :' + str(surrogate_policy_evaluation_information['env_runners']['episode_return_mean']))


if __name__ == '__main__':
    import configurations.list_experimentation_configurations

    surrogate_policy_checkpoint_path = '/experiments/bipedal_walker_old/surrogate_policy/version_0/checkpoints/epoch=64-step=38966.ckpt'
    evaluation_surrogate_policy(configurations.list_experimentation_configurations.bipedal_walker, surrogate_policy_checkpoint_path)





