from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.utils.exploration import Random
from ray.rllib.utils.replay_buffers import PrioritizedEpisodeReplayBuffer
from torch.nn import LeakyReLU
from ray.rllib.algorithms.dqn.dqn_rainbow_rl_module import DQNRainbowRLModule

from configurations.structure.experimentation_configuration import ExperimentationConfiguration
from rllib.architectures.dense_dqn import DenseDQN
from rllib.architectures.dense_ppo import DensePPO
from rllib.architectures.tetris_ppo import TetrisPPO

tetris = ExperimentationConfiguration(
    experimentation_name='tetris',
    environment_name='TetrisRllib',
)
tetris.ray_local_mode = False
tetris.reinforcement_learning_configuration.number_environment_runners = 10
tetris.reinforcement_learning_configuration.number_gpus_per_learner = 1
tetris.reinforcement_learning_configuration.number_environment_per_environment_runners = 1

# DQN
# tetris.reinforcement_learning_configuration.algorithm_name = 'DQN'
# tetris.reinforcement_learning_configuration.architecture_configuration = {
#     'fcnet_hiddens': [256, 256],
#     'head_fcnet_hiddens': [128, 128],
#     'noisy': False,
#     'epsilon': [(0, 0.5), (300, 0.1), (400, 0.05)],
#     # 'epsilon': [(0, 1.0), (10_000, 0.25)],
# }
# # tetris.reinforcement_learning_configuration.replay_buffer_configuration = {
# #     'type': 'PrioritizedEpisodeReplayBuffer',
# #     'capacity': 300_000,
# #     'alpha': 0.5,
# #     'beta': 0.5,
# # }
# tetris.reinforcement_learning_configuration.train_batch_size = 10_000
# tetris.reinforcement_learning_configuration.evaluation_interval = 100
# tetris.reinforcement_learning_configuration.checkpoint_frequency = 100

# PPO
tetris.reinforcement_learning_configuration.algorithm_name = 'PPO'
tetris.reinforcement_learning_configuration.architecture = TetrisPPO
tetris.reinforcement_learning_configuration.use_generalized_advantage_estimator = True
tetris.reinforcement_learning_configuration.minibatch_size = 5_000
tetris.reinforcement_learning_configuration.batch_mode = 'complete_episodes'
tetris.reinforcement_learning_configuration.train_batch_size = 40_000
tetris.reinforcement_learning_configuration.flatten_observations = False

# Others
# tetris.reinforcement_learning_configuration.architecture = DenseDQN
# tetris.reinforcement_learning_configuration.exploration_configuration = Random
# tetris.reinforcement_learning_configuration.architecture_configuration = DefaultModelConfig(
#     fcnet_hiddens=[1024, 1024],
#     # head_fcnet_hiddens=[512, 126, 64],
# ),

