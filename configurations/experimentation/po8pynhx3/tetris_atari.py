from torch.nn import LeakyReLU

from configurations.structure.experimentation_configuration import ExperimentationConfiguration
from rllib_repertory.architectures.cnn_ppo import CNNPPO

tetris_atari = ExperimentationConfiguration(
    experimentation_name='tetris_atari',
    environment_name='TetrisAtariRllib',
)
tetris_atari.ray_local_mode = False

tetris_atari.reinforcement_learning_configuration.training_name = 'v6'
tetris_atari.reinforcement_learning_configuration.flatten_observations = False
tetris_atari.reinforcement_learning_configuration.evaluation_duration = 10

tetris_atari.reinforcement_learning_configuration.architecture = CNNPPO
tetris_atari.reinforcement_learning_configuration.architecture_configuration = {
    'configuration_cnn': [(16, 8, 4), (32, 4, 4), (64, 3, 2), (128, 2, 1)],
    'configuration_hidden_layers': [64, 64],
    'activation_function_class': LeakyReLU,
    'use_layer_normalization_cnn': True,
    'use_unified_cnn': True,
}

tetris_atari.reinforcement_learning_configuration.number_gpus_per_learner = 1

tetris_atari.reinforcement_learning_configuration.number_environment_runners = 8
tetris_atari.reinforcement_learning_configuration.evaluation_num_environment_runners = 2
tetris_atari.reinforcement_learning_configuration.number_cpus_per_learner = 1
tetris_atari.reinforcement_learning_configuration.number_gpus_per_environment_runners = 0

tetris_atari.reinforcement_learning_configuration.train_batch_size = 512 * 32
tetris_atari.reinforcement_learning_configuration.compress_observations = True
tetris_atari.reinforcement_learning_configuration.minibatch_size = 128
# tetris_atari.reinforcement_learning_configuration.learning_rate = 1e-5
tetris_atari.reinforcement_learning_configuration.entropy_coefficient = 0.05

tetris_atari.reinforcement_learning_configuration.evaluation_num_environment_runners = 1

# tetris_atari.reinforcement_learning_configuration.compress_observations = True
# tetris_atari.reinforcement_learning_configuration.number_environment_runners = 2
# tetris_atari.reinforcement_learning_configuration.number_environment_per_environment_runners = 1
# 
# tetris_atari.reinforcement_learning_configuration.number_gpus_per_environment_runners = 1/tetris_atari.reinforcement_learning_configuration.number_environment_runners
# tetris_atari.reinforcement_learning_configuration.number_gpus_per_learner = 1
