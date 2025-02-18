from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.utils.exploration import Random
from ray.rllib.utils.replay_buffers import PrioritizedEpisodeReplayBuffer
from torch.nn import LeakyReLU
from ray.rllib.algorithms.dqn.dqn_rainbow_rl_module import DQNRainbowRLModule

from configurations.structure.experimentation_configuration import ExperimentationConfiguration
from rllib_repertory.architectures.cnn_ppo import CNNPPO
from rllib_repertory.architectures.dense_dqn import DenseDQN
from rllib_repertory.architectures.dense_ppo import DensePPO
from rllib_repertory.architectures.tetris_ppo import TetrisPPOTransformer, TetrisPPOCNN

tetris_ppo_dense = ExperimentationConfiguration(
    experimentation_name='tetris_ppo_dense',
    environment_name='TetrisRllib',
)

tetris_ppo_dense.ray_local_mode = False

tetris_ppo_dense.reinforcement_learning_configuration.number_environment_runners = 8
tetris_ppo_dense.reinforcement_learning_configuration.number_environment_per_environment_runners = 1
tetris_ppo_dense.reinforcement_learning_configuration.number_gpus_per_environment_runners = 0
tetris_ppo_dense.reinforcement_learning_configuration.number_gpus_per_learner = 1

tetris_ppo_dense.reinforcement_learning_configuration.training_name = 'debug_dense_ppo_v3'
tetris_ppo_dense.reinforcement_learning_configuration.algorithm_name = 'PPO'
tetris_ppo_dense.reinforcement_learning_configuration.flatten_observations = True
tetris_ppo_dense.reinforcement_learning_configuration.use_generalized_advantage_estimator = True
tetris_ppo_dense.reinforcement_learning_configuration.train_batch_size = 2048 #1024
tetris_ppo_dense.reinforcement_learning_configuration.minibatch_size = 256 #1024
tetris_ppo_dense.reinforcement_learning_configuration.number_epochs = 10
tetris_ppo_dense.reinforcement_learning_configuration.batch_mode = 'complete_episodes'
tetris_ppo_dense.reinforcement_learning_configuration.learning_rate = 5e-6
tetris_ppo_dense.reinforcement_learning_configuration.clip_policy_parameter = 0.05

tetris_ppo_dense.reinforcement_learning_configuration.architecture = DensePPO
tetris_ppo_dense.reinforcement_learning_configuration.architecture_configuration = {
    'configuration_hidden_layers': [1024, 512, 128, 64],
    'activation_function': LeakyReLU(),
}