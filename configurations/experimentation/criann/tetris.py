from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.utils.exploration import Random
from ray.rllib.utils.replay_buffers import PrioritizedEpisodeReplayBuffer
from torch.nn import LeakyReLU
from ray.rllib.algorithms.dqn.dqn_rainbow_rl_module import DQNRainbowRLModule

from configurations.structure.experimentation_configuration import ExperimentationConfiguration
from rllib_repertory.architectures.dense_dqn import DenseDQN
from rllib_repertory.architectures.dense_ppo import DensePPO
from rllib_repertory.architectures.tetris_ppo import TetrisPPOTransformer, TetrisPPOCNN

tetris = ExperimentationConfiguration(
    experimentation_name='tetris',
    environment_name='TetrisRllib',
)

tetris.ray_local_mode = False

tetris.reinforcement_learning_configuration.number_environment_runners = 16
tetris.reinforcement_learning_configuration.number_environment_per_environment_runners = 1
tetris.reinforcement_learning_configuration.number_gpus_per_environment_runners = 0
tetris.reinforcement_learning_configuration.number_gpus_per_learner = 1

# PPO CNN
tetris.reinforcement_learning_configuration.training_name = 'CNN_V1'
tetris.reinforcement_learning_configuration.algorithm_name = 'PPO'
tetris.environment_configuration = {'observation_rgb': True}
tetris.reinforcement_learning_configuration.flatten_observations = False
tetris.reinforcement_learning_configuration.compress_observations = True
tetris.reinforcement_learning_configuration.architecture = TetrisPPOCNN
tetris.reinforcement_learning_configuration.use_generalized_advantage_estimator = True
tetris.reinforcement_learning_configuration.train_batch_size = 2048
tetris.reinforcement_learning_configuration.minibatch_size = 2048
tetris.reinforcement_learning_configuration.number_epochs = 32
tetris.reinforcement_learning_configuration.batch_mode = 'complete_episodes'
tetris.reinforcement_learning_configuration.entropy_coefficient = 0.01


# # PPO Transformer
# tetris.reinforcement_learning_configuration.training_name = 'Transformer_V1'
# tetris.reinforcement_learning_configuration.algorithm_name = 'PPO'
# tetris.reinforcement_learning_configuration.flatten_observations = True
# tetris.reinforcement_learning_configuration.use_generalized_advantage_estimator = True
# tetris.reinforcement_learning_configuration.architecture = TetrisPPOTransformer
# tetris.reinforcement_learning_configuration.architecture_configuration = {
#     'dimension_token': 32,
#     'number_heads': 2,
#     'dimension_feedforward': 64,
#     'number_transformer_layers': 2,
# }
# tetris.reinforcement_learning_configuration.train_batch_size = 1024
# tetris.reinforcement_learning_configuration.minibatch_size = 1024
# tetris.reinforcement_learning_configuration.number_epochs = 32
# tetris.reinforcement_learning_configuration.batch_mode = 'complete_episodes'


# # PPO Dense
# tetris.reinforcement_learning_configuration.training_name = 'Dense_V1'
# tetris.reinforcement_learning_configuration.algorithm_name = 'PPO'
# tetris.reinforcement_learning_configuration.flatten_observations = True
# tetris.reinforcement_learning_configuration.use_generalized_advantage_estimator = True
# tetris.reinforcement_learning_configuration.architecture = DensePPO
# tetris.reinforcement_learning_configuration.architecture_configuration = {
#     'configuration_hidden_layers': [2048, 1024, 512, 256, 128, 64],
#     'activation_function': LeakyReLU(),
# }
# tetris.reinforcement_learning_configuration.train_batch_size = 40_000
# tetris.reinforcement_learning_configuration.minibatch_size = 40_000
# tetris.reinforcement_learning_configuration.number_epochs = 32
# tetris.reinforcement_learning_configuration.batch_mode = 'complete_episodes'
# tetris.reinforcement_learning_configuration.entropy_coefficient = 0.01
