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

# PPO
tetris.reinforcement_learning_configuration.training_name = 'base'
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