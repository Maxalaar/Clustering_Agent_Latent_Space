from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.utils.exploration import Random
from ray.rllib.utils.replay_buffers import PrioritizedEpisodeReplayBuffer
from torch.nn import LeakyReLU
from ray.rllib.algorithms.dqn.dqn_rainbow_rl_module import DQNRainbowRLModule

from configurations.structure.experimentation_configuration import ExperimentationConfiguration
from rllib_repertory.architectures.dense_dqn import DenseDQN
from rllib_repertory.architectures.dense_ppo import DensePPO
from rllib_repertory.architectures.tetris_ppo import TetrisPPO

tetris = ExperimentationConfiguration(
    experimentation_name='tetris',
    environment_name='TetrisRllib',
)
tetris.ray_local_mode = False

tetris.reinforcement_learning_configuration.number_environment_runners = 8
tetris.reinforcement_learning_configuration.number_environment_per_environment_runners = 1
tetris.reinforcement_learning_configuration.number_gpus_per_environment_runners = 1 / tetris.reinforcement_learning_configuration.number_environment_runners

tetris.reinforcement_learning_configuration.number_gpus_per_learner = 1


# PPO
tetris.reinforcement_learning_configuration.algorithm_name = 'PPO'
tetris.reinforcement_learning_configuration.architecture = DensePPO
tetris.reinforcement_learning_configuration.architecture_configuration = {
    'configuration_hidden_layers': [64, 128, 246, 512, 246, 128, 64],
    'activation_function': LeakyReLU(),
}
tetris.reinforcement_learning_configuration.use_generalized_advantage_estimator = True
tetris.reinforcement_learning_configuration.minibatch_size = 5_000
tetris.reinforcement_learning_configuration.batch_mode = 'complete_episodes'
tetris.reinforcement_learning_configuration.train_batch_size = 40_000
tetris.reinforcement_learning_configuration.flatten_observations = True
tetris.reinforcement_learning_configuration.number_epochs = 64


# Others
# tetris.reinforcement_learning_configuration.architecture = DenseDQN
# tetris.reinforcement_learning_configuration.exploration_configuration = Random
# tetris.reinforcement_learning_configuration.architecture_configuration = DefaultModelConfig(
#     fcnet_hiddens=[1024, 1024],
#     # head_fcnet_hiddens=[512, 126, 64],
# ),
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
