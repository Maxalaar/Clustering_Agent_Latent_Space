from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.utils.exploration import Random
from ray.rllib.utils.replay_buffers import PrioritizedEpisodeReplayBuffer
from torch.nn import LeakyReLU
from ray.rllib.algorithms.dqn.dqn_rainbow_rl_module import DQNRainbowRLModule

from configurations.structure.experimentation_configuration import ExperimentationConfiguration
from rllib_repertory.architectures.dense_dqn import DenseDQN
from rllib_repertory.architectures.dense_ppo import DensePPO
from rllib_repertory.architectures.transformer_ppo import TransformerPPO

tetris_dense = ExperimentationConfiguration(
    experimentation_name='tetris',
    environment_name='TetrisRllib',
)

tetris_dense.ray_local_mode = False

tetris_dense.reinforcement_learning_configuration.number_environment_runners = 6
tetris_dense.reinforcement_learning_configuration.number_environment_per_environment_runners = 1
tetris_dense.reinforcement_learning_configuration.number_gpus_per_environment_runners = 1 / tetris_dense.reinforcement_learning_configuration.number_environment_runners
tetris_dense.reinforcement_learning_configuration.number_gpus_per_learner = 1

# PPO Dense
tetris_dense.reinforcement_learning_configuration.training_name = 'dense_ppo_v33'
tetris_dense.reinforcement_learning_configuration.algorithm_name = 'PPO'
tetris_dense.reinforcement_learning_configuration.flatten_observations = True
tetris_dense.reinforcement_learning_configuration.use_generalized_advantage_estimator = True
tetris_dense.reinforcement_learning_configuration.train_batch_size = 25_000
tetris_dense.reinforcement_learning_configuration.minibatch_size = 25_000
tetris_dense.reinforcement_learning_configuration.number_epochs = 32
tetris_dense.reinforcement_learning_configuration.batch_mode = 'truncate_episodes' # 'truncate_episodes' or 'complete_episodes'
tetris_dense.reinforcement_learning_configuration.architecture = DensePPO
tetris_dense.reinforcement_learning_configuration.architecture_configuration = {
    'configuration_hidden_layers': [2048, 1024, 512, 256, 128, 64],
    'activation_function': LeakyReLU,
}

tetris_dense.reinforcement_learning_configuration.learning_rate = 1e-5
tetris_dense.reinforcement_learning_configuration.clip_policy_parameter = 0.1

# tetris_dense.reinforcement_learning_configuration.clip_policy_parameter = 0.01
# tetris_dense.reinforcement_learning_configuration.kullback_leibler_coefficient = 0.3
# tetris_dense.reinforcement_learning_configuration.entropy_coefficient = 0.03

# tetris_dense.reinforcement_learning_configuration.use_kullback_leibler_loss = True
# tetris_dense.reinforcement_learning_configuration.kullback_leibler_coefficient = 0.3
# tetris_dense.reinforcement_learning_configuration.number_epochs = 32
# tetris_dense.reinforcement_learning_configuration.learning_rate = 3e-4
# tetris_dense.reinforcement_learning_configuration.use_generalized_advantage_estimator = True
# tetris_dense.reinforcement_learning_configuration.lambda_gae = 0.95
# tetris_dense.reinforcement_learning_configuration.train_batch_size = 2048 * 10
# tetris_dense.reinforcement_learning_configuration.minibatch_size = 2048
# tetris_dense.reinforcement_learning_configuration.batch_mode = 'complete_episodes'
# tetris_dense.reinforcement_learning_configuration.number_epochs = 8
# tetris_dense.reinforcement_learning_configuration.entropy_coefficient = 0.01
#
# tetris_dense.reinforcement_learning_configuration.gradient_clip = 0.1
# tetris_dense.reinforcement_learning_configuration.clip_policy_parameter = 0.025
# tetris_dense.reinforcement_learning_configuration.use_kullback_leibler_loss = True
# tetris_dense.reinforcement_learning_configuration.kullback_leibler_coefficient = 0.5
#
# tetris_dense.reinforcement_learning_configuration.checkpoint_frequency = 1000
# tetris_dense.reinforcement_learning_configuration.evaluation_interval = 1000

# # DQN
# # https://github.com/Max-We/Tetris-Gymnasium/blob/main/examples/train_cnn.py
# tetris.reinforcement_learning_configuration.algorithm_name = 'DQN'
#
# tetris.environment_configuration = {'observation_rgb': True}
# tetris.reinforcement_learning_configuration.flatten_observations = False
# tetris.reinforcement_learning_configuration.compress_observations = True
#
# # tetris.reinforcement_learning_configuration.architecture = TetrisPPOCNN
#
# tetris.reinforcement_learning_configuration.learning_rate = 1e-4
# tetris.reinforcement_learning_configuration.replay_buffer_configuration = {
#     # 'type': 'PrioritizedEpisodeReplayBuffer',
#     'capacity': 1_000_000,
#     # 'alpha': 0.5,
#     # 'beta': 0.5,
# }
# tetris.reinforcement_learning_configuration.gamma = 0.99
# tetris.reinforcement_learning_configuration.tau = 1.0
# tetris.reinforcement_learning_configuration.target_network_update_frequency = 1000
# tetris.reinforcement_learning_configuration.batch_size = 32
# tetris.reinforcement_learning_configuration.epsilon = [[0, 1], [2_000_000, 0.1]]
# tetris.reinforcement_learning_configuration.number_steps_sampled_before_learning_starts = 80_000
# tetris.reinforcement_learning_configuration.training_intensity = 4


# # PPO CNN
# tetris.reinforcement_learning_configuration.algorithm_name = 'PPO'
# tetris.environment_configuration = {'observation_rgb': True}
# tetris.reinforcement_learning_configuration.flatten_observations = False
# tetris.reinforcement_learning_configuration.compress_observations = True
# tetris.reinforcement_learning_configuration.architecture = TetrisPPOCNN
# tetris.reinforcement_learning_configuration.use_generalized_advantage_estimator = True
# tetris.reinforcement_learning_configuration.train_batch_size = 2048
# tetris.reinforcement_learning_configuration.minibatch_size = 2048
# tetris.reinforcement_learning_configuration.number_epochs = 32
# tetris.reinforcement_learning_configuration.batch_mode = 'complete_episodes'

# # PPO Transformer
# tetris.reinforcement_learning_configuration.algorithm_name = 'PPO'
# tetris.reinforcement_learning_configuration.flatten_observations = True
# tetris.reinforcement_learning_configuration.use_generalized_advantage_estimator = True
# tetris.reinforcement_learning_configuration.architecture = TransformerPPO
# tetris.reinforcement_learning_configuration.architecture_configuration = {
#     'dimension_token': 4,
#     'number_heads': 2,
#     'dimension_feedforward': 32,
#     'number_transformer_layers': 2,
#     'action_dense_layer_shapes': [2048, 1024, 512, 256, 128, 64, 32],
#     'critic_dense_layer_shape': [2048, 1024, 512, 256, 128, 64, 32],
# }
# tetris.reinforcement_learning_configuration.train_batch_size = 512
# tetris.reinforcement_learning_configuration.minibatch_size = 512
# tetris.reinforcement_learning_configuration.number_epochs = 16
# # tetris.reinforcement_learning_configuration.batch_mode = 'complete_episodes'
