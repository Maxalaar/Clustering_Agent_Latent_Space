from torch.nn import LeakyReLU

from configurations.structure.experimentation_configuration import ExperimentationConfiguration
from rllib_repertory.architectures.dense_ppo import DensePPO
from rllib_repertory.architectures.transformer_ppo import TransformerPPO

bipedal_walker_hardcore = ExperimentationConfiguration(
    experimentation_name='bipedal_walker_hardcore',
    environment_name='BipedalWalkerRllib',
)
bipedal_walker_hardcore.environment_configuration = {'hardcore': True}

# Ray
bipedal_walker_hardcore.ray_local_mode = False
bipedal_walker_hardcore.reinforcement_learning_configuration.number_gpus_per_learner = 1
bipedal_walker_hardcore.reinforcement_learning_configuration.number_environment_runners = 16
bipedal_walker_hardcore.reinforcement_learning_configuration.number_environment_per_environment_runners = 1

# Reinforcement Learning

# Dense
# https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml
# https://github.com/ovechkin-dm/ppo-lstm-parallel
bipedal_walker_hardcore.reinforcement_learning_configuration.training_name = 'New_PPO_V16'

bipedal_walker_hardcore.reinforcement_learning_configuration.architecture = DensePPO
bipedal_walker_hardcore.reinforcement_learning_configuration.architecture_configuration = {
    'use_same_encoder_actor_critic': False,
    'configuration_hidden_layers': [128, 128, 128, 128],
    'activation_function': LeakyReLU(),
}

bipedal_walker_hardcore.reinforcement_learning_configuration.learning_rate = 1e-4
bipedal_walker_hardcore.reinforcement_learning_configuration.use_generalized_advantage_estimator = True
bipedal_walker_hardcore.reinforcement_learning_configuration.lambda_gae = 0.95
bipedal_walker_hardcore.reinforcement_learning_configuration.train_batch_size = 40_000    # 40_000
bipedal_walker_hardcore.reinforcement_learning_configuration.minibatch_size = 40_000  #4 0_000
bipedal_walker_hardcore.reinforcement_learning_configuration.number_epochs = 16
bipedal_walker_hardcore.reinforcement_learning_configuration.entropy_coefficient = 0.001

bipedal_walker_hardcore.reinforcement_learning_configuration.checkpoint_frequency = 200
bipedal_walker_hardcore.reinforcement_learning_configuration.evaluation_interval = 200
bipedal_walker_hardcore.reinforcement_learning_configuration.evaluation_duration = 10

bipedal_walker_hardcore.reinforcement_learning_configuration.use_kullback_leibler_loss = True
# bipedal_walker_hardcore.reinforcement_learning_configuration.kullback_leibler_coefficient = 0.02
bipedal_walker_hardcore.reinforcement_learning_configuration.clip_policy_parameter = 0.05

# bipedal_walker_hardcore.reinforcement_learning_configuration.gradient_clip = 0.1
# bipedal_walker_hardcore.reinforcement_learning_configuration.gradient_clip_by = 'global_norm'
# bipedal_walker_hardcore.reinforcement_learning_configuration.clip_value_function_parameter = 0.2

bipedal_walker_hardcore.reinforcement_learning_configuration.batch_mode = 'truncate_episodes'   # 'complete_episodes'
bipedal_walker_hardcore.reinforcement_learning_configuration.gamma = 0.99

# # Transformer
# bipedal_walker_hardcore.reinforcement_learning_configuration.training_name = 'Transformer_V6'
# bipedal_walker_hardcore.reinforcement_learning_configuration.architecture = TransformerPPO
# bipedal_walker_hardcore.reinforcement_learning_configuration.architecture_configuration = {
#     'action_token_projector_layer_shapes': [8],
#     'critic_token_projector_layer_shapes': [8],
#     'dimension_token': 16,
#     'number_heads': 2,
#     'dimension_feedforward': 32,
#     'number_transformer_layers': 2,
#     'use_multiple_projectors': True,
#     'use_same_encoder_actor_critic': True,
# }
#
# bipedal_walker_hardcore.reinforcement_learning_configuration.learning_rate = 1e-4
# bipedal_walker_hardcore.reinforcement_learning_configuration.use_generalized_advantage_estimator = True
# bipedal_walker_hardcore.reinforcement_learning_configuration.lambda_gae = 0.95
# bipedal_walker_hardcore.reinforcement_learning_configuration.train_batch_size = 2048 * 20
# bipedal_walker_hardcore.reinforcement_learning_configuration.minibatch_size = 2048 * 20
# bipedal_walker_hardcore.reinforcement_learning_configuration.number_epochs = 64
# # bipedal_walker_hardcore.reinforcement_learning_configuration.entropy_coefficient = 0.001
# # bipedal_walker_hardcore.reinforcement_learning_configuration.batch_mode = 'complete_episodes'
# bipedal_walker_hardcore.reinforcement_learning_configuration.gamma = 0.99
#
# # bipedal_walker_hardcore.reinforcement_learning_configuration.gradient_clip = 0.1
# # bipedal_walker_hardcore.reinforcement_learning_configuration.gradient_clip_by = 'global_norm'
# # bipedal_walker_hardcore.reinforcement_learning_configuration.clip_all_parameter = 0.2
# # bipedal_walker_hardcore.reinforcement_learning_configuration.clip_value_function_parameter = 0.2
