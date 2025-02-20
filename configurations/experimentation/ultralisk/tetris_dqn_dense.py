from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.utils.replay_buffers import PrioritizedEpisodeReplayBuffer

from configurations.structure.experimentation_configuration import ExperimentationConfiguration
from rllib_repertory.architectures.cnn_dqn import CNNDQN
from rllib_repertory.architectures.dense_dqn import DenseDQN

tetris_dqn_dense = ExperimentationConfiguration(
    experimentation_name='tetris_dqn_dense',
    environment_name='TetrisRllib',
)

tetris_dqn_dense.ray_local_mode = False

tetris_dqn_dense.reinforcement_learning_configuration.number_environment_runners = 6
tetris_dqn_dense.reinforcement_learning_configuration.number_environment_per_environment_runners = 1
tetris_dqn_dense.reinforcement_learning_configuration.number_gpus_per_environment_runners = 0
tetris_dqn_dense.reinforcement_learning_configuration.number_gpus_per_learner = 1

tetris_dqn_dense.reinforcement_learning_configuration.training_name = 'tetris_dqn_dense_v5'
tetris_dqn_dense.reinforcement_learning_configuration.algorithm_name = 'DQN'
tetris_dqn_dense.reinforcement_learning_configuration.architecture_configuration = DefaultModelConfig(
    head_fcnet_hiddens=[2048, 1024, 512, 256, 128, 64],
)
tetris_dqn_dense.reinforcement_learning_configuration.batch_mode = 'complete_episodes'

tetris_dqn_dense.reinforcement_learning_configuration.number_step_return = 3
tetris_dqn_dense.reinforcement_learning_configuration.use_noisy_exploration = True
tetris_dqn_dense.reinforcement_learning_configuration.use_dueling_dqn = True
tetris_dqn_dense.reinforcement_learning_configuration.use_double_q_function = True
tetris_dqn_dense.reinforcement_learning_configuration.learning_rate = 0.0001
tetris_dqn_dense.reinforcement_learning_configuration.replay_buffer_configuration = {
    "type": "PrioritizedEpisodeReplayBuffer",
    "capacity": 50_000,
    "alpha": 0.6,
    "beta": 0.4,
}
tetris_dqn_dense.reinforcement_learning_configuration.target_network_update_frequency = 500
tetris_dqn_dense.reinforcement_learning_configuration.train_batch_size = 32
tetris_dqn_dense.reinforcement_learning_configuration.number_epochs = 128
tetris_dqn_dense.reinforcement_learning_configuration.number_steps_sampled_before_learning_starts = 10_000

