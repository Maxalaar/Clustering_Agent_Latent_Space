from torch.nn import LeakyReLU

from configurations.structure.experimentation_configuration import ExperimentationConfiguration
from rllib_repertory.architectures.dense_ppo import DensePPO

bipedal_walker_hardcore = ExperimentationConfiguration(
    experimentation_name='bipedal_walker_hardcore',
    environment_name='BipedalWalkerRllib',
)
bipedal_walker_hardcore.environment_configuration = {'hardcore': True}

# Ray
bipedal_walker_hardcore.ray_local_mode = False
bipedal_walker_hardcore.reinforcement_learning_configuration.number_gpus_per_learner = 1
bipedal_walker_hardcore.reinforcement_learning_configuration.number_environment_runners = 16
bipedal_walker_hardcore.reinforcement_learning_configuration.number_environment_per_environment_runners = 2

# Reinforcement Learning
# https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml
# https://github.com/ovechkin-dm/ppo-lstm-parallel
bipedal_walker_hardcore.reinforcement_learning_configuration.training_name = 'Dense_V3'
bipedal_walker_hardcore.reinforcement_learning_configuration.architecture = DensePPO
bipedal_walker_hardcore.reinforcement_learning_configuration.architecture_configuration = {
    'configuration_hidden_layers': [1024, 1024, 1024, 1024],
    'activation_function': LeakyReLU(),
    'layer_normalization': True,
}
bipedal_walker_hardcore.reinforcement_learning_configuration.number_gpus_per_learner = 1
bipedal_walker_hardcore.reinforcement_learning_configuration.number_environment_runners = 6
bipedal_walker_hardcore.reinforcement_learning_configuration.number_environment_per_environment_runners = 2

bipedal_walker_hardcore.reinforcement_learning_configuration.learning_rate = 3e-4
bipedal_walker_hardcore.reinforcement_learning_configuration.use_generalized_advantage_estimator = True
bipedal_walker_hardcore.reinforcement_learning_configuration.lambda_gae = 0.95
bipedal_walker_hardcore.reinforcement_learning_configuration.train_batch_size = 1024 #40_000
bipedal_walker_hardcore.reinforcement_learning_configuration.minibatch_size = 1024 #40_000
bipedal_walker_hardcore.reinforcement_learning_configuration.number_epochs = 128
bipedal_walker_hardcore.reinforcement_learning_configuration.entropy_coefficient = 0.0

# bipedal_walker_hardcore.reinforcement_learning_configuration.architecture = DensePPO
# bipedal_walker_hardcore.reinforcement_learning_configuration.architecture_configuration = {
#     'use_same_encoder_actor_critic': False,
#     # 'configuration_encoder_hidden_layers': [64, 128, 256],
#     'configuration_hidden_layers': [1024, 1024, 1024, 1024],
#     'activation_function': LeakyReLU(),
#     'layer_normalization': True,
#     'dropout': False,
# }
#
# bipedal_walker_hardcore.reinforcement_learning_configuration.learning_rate = 1e-4
# bipedal_walker_hardcore.reinforcement_learning_configuration.use_generalized_advantage_estimator = True
# bipedal_walker_hardcore.reinforcement_learning_configuration.lambda_gae = 0.95
# bipedal_walker_hardcore.reinforcement_learning_configuration.train_batch_size = 40_000
# bipedal_walker_hardcore.reinforcement_learning_configuration.minibatch_size = 40_000
# bipedal_walker_hardcore.reinforcement_learning_configuration.number_epochs = 128
# bipedal_walker_hardcore.reinforcement_learning_configuration.entropy_coefficient = 0.001
#
# # bipedal_walker_hardcore.reinforcement_learning_configuration.gradient_clip = 0.1
# # bipedal_walker_hardcore.reinforcement_learning_configuration.gradient_clip_by = 'global_norm'
# # bipedal_walker_hardcore.reinforcement_learning_configuration.clip_all_parameter = 0.2
# # bipedal_walker_hardcore.reinforcement_learning_configuration.clip_value_function_parameter = 0.2
#
# bipedal_walker_hardcore.reinforcement_learning_configuration.batch_mode = 'complete_episodes'
# bipedal_walker_hardcore.reinforcement_learning_configuration.gamma = 0.99
