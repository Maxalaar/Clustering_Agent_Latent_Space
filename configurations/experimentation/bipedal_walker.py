from configurations.structure.experimentation_configuration import ExperimentationConfiguration


bipedal_walker = ExperimentationConfiguration(
    experimentation_name='bipedal_walker',
    environment_name='BipedalWalkerRllib',
)

bipedal_walker.reinforcement_learning_configuration.number_gpu = 1
bipedal_walker.reinforcement_learning_configuration.number_gpus_per_learner = 1

# https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml
# bipedal_walker.reinforcement_learning_configuration.learning_rate = 3e-4
# bipedal_walker.reinforcement_learning_configuration.clip_policy_parameter = 0.18
# bipedal_walker.reinforcement_learning_configuration.lambda_gae = 0.95
bipedal_walker.reinforcement_learning_configuration.train_batch_size = 80_000
bipedal_walker.reinforcement_learning_configuration.mini_batch_size_per_learner = 20_000

bipedal_walker.reinforcement_learning_configuration.clip_all_parameter = 0.1
bipedal_walker.reinforcement_learning_configuration.clip_policy_parameter = 0.1
bipedal_walker.reinforcement_learning_configuration.clip_value_function_parameter = 0.1
