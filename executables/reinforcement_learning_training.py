import os

import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig, PPO
from ray.rllib.algorithms.dqn import DQNConfig, DQN
from ray.rllib.utils.from_config import NotProvided
from ray.rllib.algorithms import AlgorithmConfig, Algorithm

from configurations.structure.experimentation_configuration import ExperimentationConfiguration
from environments.register_environments import register_environments
from models.architectures.rllib.register_architectures import register_architectures


def reinforcement_learning_training(experimentation_configuration: ExperimentationConfiguration):
    reinforcement_learning_configuration = experimentation_configuration.reinforcement_learning_configuration
    reinforcement_learning_configuration.to_yaml_file(experimentation_configuration.reinforcement_learning_storage_path)

    ray_initialization = False
    if not ray.is_initialized():
        ray.init(local_mode=experimentation_configuration.reinforcement_learning_configuration.ray_local_mode)
        ray_initialization = True

    register_environments()
    register_architectures()

    algorithm: Algorithm
    algorithm_configuration: AlgorithmConfig

    if reinforcement_learning_configuration.algorithm == 'PPO':
        algorithm = PPO
        algorithm_configuration = PPOConfig()
    elif reinforcement_learning_configuration.algorithm == 'DQN':
        algorithm = DQN
        algorithm_configuration = DQNConfig()
    else:
        raise ValueError('Unsupported algorithm ' + str(reinforcement_learning_configuration.algorithm) + '.')

    algorithm_configuration.framework(reinforcement_learning_configuration.framework)
    algorithm_configuration.resources(num_gpus=reinforcement_learning_configuration.number_gpu)

    # Environment
    algorithm_configuration.environment(
        env=experimentation_configuration.environment_name,
        env_config=experimentation_configuration.environment_configuration,
    )

    # Training
    if reinforcement_learning_configuration.architecture_name is not NotProvided:
        algorithm_configuration.training(
            model={
                'custom_model': reinforcement_learning_configuration.architecture_name,
                'custom_model_config': reinforcement_learning_configuration.architecture_configuration,
            }
        )
    algorithm_configuration.training(
        train_batch_size=reinforcement_learning_configuration.train_batch_size,
        lr=reinforcement_learning_configuration.learning_rate,
    )

    if type(algorithm_configuration) is PPOConfig:
        algorithm_configuration: PPOConfig
        algorithm_configuration.training(
            use_gae=True,
            mini_batch_size_per_learner=reinforcement_learning_configuration.mini_batch_size_per_learner,
            sgd_minibatch_size=reinforcement_learning_configuration.mini_batch_size_per_learner,
            num_sgd_iter=reinforcement_learning_configuration.num_sgd_iter,
            lambda_=reinforcement_learning_configuration.lambda_gae,
            grad_clip=reinforcement_learning_configuration.clip_all_parameter,
            clip_param=reinforcement_learning_configuration.clip_policy_parameter,
            vf_clip_param=reinforcement_learning_configuration.clip_value_function_parameter,
        )

    # Environment runners
    algorithm_configuration.env_runners(
        batch_mode=reinforcement_learning_configuration.batch_mode,
        num_env_runners=reinforcement_learning_configuration.number_environment_runners,
        num_envs_per_env_runner=reinforcement_learning_configuration.number_environment_per_environment_runners,
        num_cpus_per_env_runner=reinforcement_learning_configuration.number_cpus_per_environment_runners,
        num_gpus_per_env_runner=reinforcement_learning_configuration.number_gpus_per_environment_runners,
    )

    # Learners
    algorithm_configuration.learners(
        num_learners=reinforcement_learning_configuration.number_learners,
        num_cpus_per_learner=reinforcement_learning_configuration.number_cpus_per_learner,
        num_gpus_per_learner=reinforcement_learning_configuration.number_gpus_per_learner
    )

    # Evaluation
    algorithm_configuration.evaluation(
        evaluation_interval=reinforcement_learning_configuration.evaluation_interval,
        evaluation_num_env_runners=reinforcement_learning_configuration.evaluation_num_environment_runners,
        evaluation_duration=reinforcement_learning_configuration.evaluation_duration,
        evaluation_parallel_to_training=reinforcement_learning_configuration.evaluation_parallel_to_training,
    )

    # Callbacks
    if reinforcement_learning_configuration.callback is not NotProvided:
        algorithm_configuration.callbacks(reinforcement_learning_configuration.callback)

    tuner_save_path = os.path.join(experimentation_configuration.experimentation_storage_path, str(experimentation_configuration.reinforcement_learning_storage_path), 'tuner.pkl')
    if os.path.exists(tuner_save_path):
        tuner = tune.Tuner.restore(
            os.path.dirname(tuner_save_path),
            trainable=algorithm,
            resume_unfinished=True,
            resume_errored=True,
            restart_errored=True,
        )

    else:
        tuner = tune.Tuner(
            trainable=algorithm,
            param_space=algorithm_configuration,
            run_config=air.RunConfig(
                name=str(experimentation_configuration.reinforcement_learning_storage_path),
                storage_path=str(experimentation_configuration.experimentation_storage_path),
                stop=reinforcement_learning_configuration.stopping_criterion,
                checkpoint_config=air.CheckpointConfig(
                    num_to_keep=reinforcement_learning_configuration.number_checkpoint_to_keep,
                    checkpoint_score_attribute=reinforcement_learning_configuration.checkpoint_score_attribute,
                    checkpoint_score_order=reinforcement_learning_configuration.checkpoint_score_order,
                    checkpoint_frequency=reinforcement_learning_configuration.checkpoint_frequency,
                    checkpoint_at_end=True,
                )
            ),
        )

    tuner.fit()

    if ray_initialization:
        ray.shutdown()


if __name__ == '__main__':
    from configurations.experimentation.cartpole import cartpole
    from configurations.experimentation.bipedal_walker import bipedal_walker
    from configurations.experimentation.lunar_lander import lunar_lander
    from configurations.experimentation.ant import ant
    from configurations.experimentation.pong_survivor_two_balls import pong_survivor_two_balls
    from configurations.experimentation.test_new_architecture import test_new_architecture

    reinforcement_learning_training(bipedal_walker)
