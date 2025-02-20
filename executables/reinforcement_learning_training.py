import argparse
import os

import ray
from ray import air, tune
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.rllib.algorithms.dreamerv3 import DreamerV3, DreamerV3Config
from ray.rllib.algorithms.ppo import PPOConfig, PPO
from ray.rllib.algorithms.dqn import DQNConfig, DQN
from ray.rllib.connectors.env_to_module import FlattenObservations
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.utils.from_config import NotProvided
from ray.rllib.algorithms import AlgorithmConfig, Algorithm
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from configurations.structure.experimentation_configuration import ExperimentationConfiguration
from environments.register_environments import register_environments
from utilities.get_configuration_class import get_configuration_class


def reinforcement_learning_training(experimentation_configuration: ExperimentationConfiguration):
    ray.init(local_mode=experimentation_configuration.ray_local_mode)

    reinforcement_learning_configuration = experimentation_configuration.reinforcement_learning_configuration
    reinforcement_learning_configuration.to_yaml_file(experimentation_configuration.reinforcement_learning_storage_path / experimentation_configuration.reinforcement_learning_configuration.training_name)

    register_environments()

    algorithm: Algorithm
    algorithm_configuration: AlgorithmConfig

    if reinforcement_learning_configuration.algorithm_name == 'PPO':
        algorithm = PPO
        algorithm_configuration = PPOConfig()
    elif reinforcement_learning_configuration.algorithm_name == 'DQN':
        algorithm = DQN
        algorithm_configuration = DQNConfig()
    elif reinforcement_learning_configuration.algorithm_name == 'DreamerV3':
        algorithm = DreamerV3
        algorithm_configuration = DreamerV3Config()
    else:
        raise ValueError('Unsupported algorithm ' + str(reinforcement_learning_configuration.algorithm_name) + '.')

    algorithm_configuration.framework(reinforcement_learning_configuration.framework)

    # Environment
    algorithm_configuration.environment(
        env=experimentation_configuration.environment_name,
        env_config=experimentation_configuration.environment_configuration,
    )

    # Reinforcement Learning Module
    if reinforcement_learning_configuration.architecture is not None:
        algorithm_configuration.rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=reinforcement_learning_configuration.architecture,
            ),
        )
    if reinforcement_learning_configuration.architecture_configuration is not None:
        algorithm_configuration.rl_module(
            model_config=reinforcement_learning_configuration.architecture_configuration,
        )

    # Training
    algorithm_configuration.training(
        gamma=reinforcement_learning_configuration.gamma,
        train_batch_size_per_learner=reinforcement_learning_configuration.train_batch_size,
        lr=reinforcement_learning_configuration.learning_rate,
        learner_connector=reinforcement_learning_configuration.learner_connector,
        num_epochs=reinforcement_learning_configuration.number_epochs,
        grad_clip=reinforcement_learning_configuration.gradient_clip,
        grad_clip_by=reinforcement_learning_configuration.gradient_clip_by,
    )
    if reinforcement_learning_configuration.exploration_configuration is not NotProvided:
        algorithm_configuration.exploration_config = reinforcement_learning_configuration.exploration_configuration

    if type(algorithm_configuration) is PPOConfig:
        algorithm_configuration: PPOConfig
        algorithm_configuration.training(
            use_gae=reinforcement_learning_configuration.use_generalized_advantage_estimator,
            minibatch_size=reinforcement_learning_configuration.minibatch_size,
            lambda_=reinforcement_learning_configuration.lambda_gae,
            grad_clip=reinforcement_learning_configuration.gradient_clip,
            clip_param=reinforcement_learning_configuration.clip_policy_parameter,
            vf_clip_param=reinforcement_learning_configuration.clip_value_function_parameter,
            learner_connector=reinforcement_learning_configuration.learner_connector,
            entropy_coeff=reinforcement_learning_configuration.entropy_coefficient,
            use_kl_loss=reinforcement_learning_configuration.use_kullback_leibler_loss,
            kl_coeff=reinforcement_learning_configuration.kullback_leibler_coefficient,
            kl_target=reinforcement_learning_configuration.kullback_leibler_target,
            vf_loss_coeff=reinforcement_learning_configuration.value_function_loss_coefficient,
        )

    if type(algorithm_configuration) is DQNConfig:
        algorithm_configuration: DQNConfig
        algorithm_configuration.training(
            replay_buffer_config=reinforcement_learning_configuration.replay_buffer_configuration,
            epsilon=reinforcement_learning_configuration.epsilon,
            target_network_update_freq=reinforcement_learning_configuration.target_network_update_frequency,
            training_intensity=reinforcement_learning_configuration.training_intensity,
            num_steps_sampled_before_learning_starts=reinforcement_learning_configuration.number_steps_sampled_before_learning_starts,
            n_step=reinforcement_learning_configuration.number_step_return,
            noisy=reinforcement_learning_configuration.use_noisy_exploration,
            dueling=reinforcement_learning_configuration.use_dueling_dqn,
            double_q=reinforcement_learning_configuration.use_double_q_function,
        )

    # Environment runners
    algorithm_configuration.env_runners(
        batch_mode=reinforcement_learning_configuration.batch_mode,
        num_env_runners=reinforcement_learning_configuration.number_environment_runners,
        num_envs_per_env_runner=reinforcement_learning_configuration.number_environment_per_environment_runners,
        num_cpus_per_env_runner=reinforcement_learning_configuration.number_cpus_per_environment_runners,
        num_gpus_per_env_runner=reinforcement_learning_configuration.number_gpus_per_environment_runners,
        compress_observations=reinforcement_learning_configuration.compress_observations,
    )

    if reinforcement_learning_configuration.flatten_observations:
        algorithm_configuration.env_runners(
            env_to_module_connector=lambda env: FlattenObservations(),
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

    # New API Stack
    algorithm_configuration.api_stack(
        enable_rl_module_and_learner=True,
        enable_env_runner_and_connector_v2=True,
    )

    tuner_save_path = os.path.join(str(experimentation_configuration.reinforcement_learning_storage_path), reinforcement_learning_configuration.training_name, 'tuner.pkl')
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
                name=experimentation_configuration.reinforcement_learning_configuration.training_name,
                storage_path=str(experimentation_configuration.reinforcement_learning_storage_path),
                stop=reinforcement_learning_configuration.stopping_criterion,
                checkpoint_config=air.CheckpointConfig(
                    num_to_keep=reinforcement_learning_configuration.number_checkpoint_to_keep,
                    checkpoint_score_attribute=reinforcement_learning_configuration.checkpoint_score_attribute,
                    checkpoint_score_order=reinforcement_learning_configuration.checkpoint_score_order,
                    checkpoint_frequency=reinforcement_learning_configuration.checkpoint_frequency,
                    checkpoint_at_end=True,
                ),
                # callbacks=[WandbLoggerCallback(project="Optimization_Project")]
            ),
        )

    tuner.fit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run reinforcement learning training.')
    parser.add_argument(
        '--experimentation_configuration_file',
        type=str,
        help="The path of the experimentation configuration file (e.g., './configurations/experimentation/cartpole.py')"
    )

    arguments = parser.parse_args()
    configuration_class = get_configuration_class(arguments.experimentation_configuration_file)

    reinforcement_learning_training(configuration_class)
