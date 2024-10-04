import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig, PPO
from ray.rllib.algorithms.dqn import DQNConfig, DQN
from ray.rllib.utils.from_config import NotProvided
from ray.rllib.algorithms import AlgorithmConfig, Algorithm

from configuration.reinforcement_learning.reinforcement_learning_configuration import ReinforcementLearningConfiguration


def reinforcement_learning_training(reinforcement_learning_configuration: ReinforcementLearningConfiguration):
    ray_initialization = False
    if not ray.is_initialized():
        ray.init()
        ray_initialization = True

    algorithm: Algorithm
    algorithm_configuration: AlgorithmConfig

    if reinforcement_learning_configuration.algorithm == 'PPO':
        algorithm = PPO
        algorithm_configuration = PPOConfig()
    elif reinforcement_learning_configuration.algorithm == 'DQN':
        algorithm = DQN
        algorithm_configuration = DQNConfig()
    else:
        raise ValueError('Unsupported algorithm ' + str(reinforcement_learning_configuration.algorithme) + '.')

    algorithm_configuration.framework(reinforcement_learning_configuration.framework)
    algorithm_configuration.resources(num_gpus=reinforcement_learning_configuration.number_gpu)

    # Environment
    algorithm_configuration.environment(
        env=reinforcement_learning_configuration.environment_name,
        env_config=reinforcement_learning_configuration.environment_configuration,
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

    if algorithm_configuration is PPOConfig:
        algorithm_configuration: PPOConfig
        algorithm_configuration.training(
            mini_batch_size_per_learner=reinforcement_learning_configuration.mini_batch_size_per_learner,
            sgd_minibatch_size=reinforcement_learning_configuration.sgd_minibatch_size,
            num_sgd_iter=reinforcement_learning_configuration.num_sgd_iter,
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

    tuner = tune.Tuner(
        trainable=algorithm,
        param_space=algorithm_configuration,
        run_config=air.RunConfig(
            name=reinforcement_learning_configuration.learning_name,
            storage_path=reinforcement_learning_configuration.storage_path,
            stop=reinforcement_learning_configuration.stopping_criterion,
            checkpoint_config=air.CheckpointConfig(
                num_to_keep=1,
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
    from configuration.reinforcement_learning.minimal_cartpole import minimal_cartpole

    reinforcement_learning_training(minimal_cartpole)
