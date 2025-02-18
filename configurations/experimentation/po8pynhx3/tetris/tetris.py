from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.utils.exploration import Random
from ray.rllib.utils.replay_buffers import PrioritizedEpisodeReplayBuffer
from torch.nn import LeakyReLU
from ray.rllib.algorithms.dqn.dqn_rainbow_rl_module import DQNRainbowRLModule

from configurations.structure.experimentation_configuration import ExperimentationConfiguration
from rllib_repertory.architectures.dense_dqn import DenseDQN
from rllib_repertory.architectures.dense_ppo import DensePPO

tetris = ExperimentationConfiguration(
    experimentation_name='tetris',
    environment_name='TetrisRllib',
)

tetris.ray_local_mode = False

tetris.reinforcement_learning_configuration.number_environment_runners = 8
tetris.reinforcement_learning_configuration.number_environment_per_environment_runners = 2
tetris.reinforcement_learning_configuration.number_gpus_per_environment_runners = 1 / tetris.reinforcement_learning_configuration.number_environment_runners
tetris.reinforcement_learning_configuration.number_gpus_per_learner = 1

tetris.reinforcement_learning_configuration.training_name = 'Transformer_V1'

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
# tetris.reinforcement_learning_configuration.training_name = 'base'
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
# tetris.reinforcement_learning_configuration.architecture = TetrisPPOTransformer
# tetris.reinforcement_learning_configuration.architecture_configuration = {
#     'dimension_token': 16,
#     'number_heads': 2,
#     'dimension_feedforward': 32,
#     'number_transformer_layers': 2,
# }
# tetris.reinforcement_learning_configuration.train_batch_size = 512
# tetris.reinforcement_learning_configuration.minibatch_size = 512
# tetris.reinforcement_learning_configuration.number_epochs = 16
# tetris.reinforcement_learning_configuration.batch_mode = 'complete_episodes'
