from pathlib import Path

import ray
from ray.rllib.core.rl_module import RLModuleSpec

from configurations.structure.experimentation_configuration import ExperimentationConfiguration
from environments.register_environments import register_environments
from rllib.architectures.lightning import Lightning
from rllib.find_best_checkpoint_path import find_best_checkpoint_path
from rllib.get_checkpoint_algorithm_configuration import get_checkpoint_algorithm_configuration


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

    # model_checkpoint_path = '/home/malaarabiou/Programming_Projects/Pycharm_Projects/Clustering_Agent_Latent_Space/experiments/bipedal_walker/surrogate_policy/version_0/checkpoints/epoch=300-step=45001.ckpt'
    # evaluation_surrogate_policy(configurations.list_experimentation_configurations.bipedal_walker, model_checkpoint_path)

    # model_checkpoint_path = '/home/malaarabiou/Programming_Projects/Pycharm_Projects/Clustering_Agent_Latent_Space/experiments/pong_survivor_tow_balls/surrogate_policy/version_1/checkpoints/epoch=954-step=143217.ckpt'
    # evaluation_surrogate_policy(configurations.list_experimentation_configurations.pong_survivor_two_balls, model_checkpoint_path)

    # model_checkpoint_path = '/home/malaarabiou/Programming_Projects/Pycharm_Projects/Clustering_Agent_Latent_Space/experiments/lunar_lander/surrogate_policy/version_0/checkpoints/epoch=24-step=21650.ckpt'
    # evaluation_surrogate_policy(configurations.list_experimentation_configurations.lunar_lander, model_checkpoint_path)

    surrogate_policy_checkpoint_path = '/temporary_good/experiments/taxi/surrogate_policy/version_0/checkpoints/epoch=133-step=19957.ckpt'
    evaluation_surrogate_policy(configurations.list_experimentation_configurations.taxi, surrogate_policy_checkpoint_path)





