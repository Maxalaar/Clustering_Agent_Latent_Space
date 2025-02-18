from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.utils.replay_buffers import PrioritizedEpisodeReplayBuffer

from configurations.structure.experimentation_configuration import ExperimentationConfiguration
from rllib_repertory.architectures.cnn_dqn import CNNDQN
from rllib_repertory.architectures.dense_dqn import DenseDQN

tetris_dqn_cnn = ExperimentationConfiguration(
    experimentation_name='tetris_dqn_cnn',
    environment_name='TetrisRllib',
)

tetris_dqn_cnn.ray_local_mode = False

tetris_dqn_cnn.reinforcement_learning_configuration.number_environment_runners = 8
tetris_dqn_cnn.reinforcement_learning_configuration.number_environment_per_environment_runners = 1
tetris_dqn_cnn.reinforcement_learning_configuration.number_gpus_per_environment_runners = 0
tetris_dqn_cnn.reinforcement_learning_configuration.number_gpus_per_learner = 1

tetris_dqn_cnn.reinforcement_learning_configuration.training_name = 'cnn_dqn_v6'
tetris_dqn_cnn.reinforcement_learning_configuration.algorithm_name = 'DQN'
# tetris_dqn_cnn.reinforcement_learning_configuration.architecture = DenseDQN
tetris_dqn_cnn.reinforcement_learning_configuration.architecture_configuration = DefaultModelConfig(
    conv_filters=[
        [16, 4, 2], #num_filters, kernel, stride
        [32, 4, 2],
        [64, 4, 2],
        # [128, 4, 2],
    ],
    conv_activation="silu",
    head_fcnet_hiddens=[1024, 512, 256, 128],
)
tetris_dqn_cnn.reinforcement_learning_configuration.replay_buffer_configuration = {
    # 'type': 'PrioritizedEpisodeReplayBuffer',
    'capacity': 40_000,
    # 'alpha': 0.5,
    # 'beta': 0.5,
}
tetris_dqn_cnn.environment_configuration = {'observation_rgb': True}
tetris_dqn_cnn.reinforcement_learning_configuration.flatten_observations = False
tetris_dqn_cnn.reinforcement_learning_configuration.compress_observations = True
tetris_dqn_cnn.reinforcement_learning_configuration.train_batch_size = 2048
tetris_dqn_cnn.reinforcement_learning_configuration.number_epochs = 256
tetris_dqn_cnn.reinforcement_learning_configuration.batch_mode = 'complete_episodes'
tetris_dqn_cnn.reinforcement_learning_configuration.learning_rate = 1e-5