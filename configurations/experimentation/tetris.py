from torch.nn import LeakyReLU
from configurations.structure.experimentation_configuration import ExperimentationConfiguration


tetris = ExperimentationConfiguration(
    experimentation_name='tetris',
    environment_name='TetrisRllib',
)

tetris.reinforcement_learning_configuration.architecture_name = 'dense'
tetris.reinforcement_learning_configuration.architecture_configuration = {
    'configuration_hidden_layers': [1024, 1024, 1024, 1024],
    'activation_function': LeakyReLU(),
}

tetris.reinforcement_learning_configuration.number_environment_runners = 8
# tetris.reinforcement_learning_configuration.batch_mode = 'truncate_episodes'
# tetris.reinforcement_learning_configuration.train_batch_size = 80_000
# tetris.reinforcement_learning_configuration.mini_batch_size_per_learner = 20_000
