from ray.rllib.utils.from_config import NotProvided
from torch.nn import LeakyReLU
from configurations.structure.experimentation_configuration import ExperimentationConfiguration


tetris = ExperimentationConfiguration(
    experimentation_name='tetris',
    environment_name='TetrisRllib',
)
tetris.reinforcement_learning_configuration.algorithm_name = 'DQN'
tetris.reinforcement_learning_configuration.architecture = NotProvided
# tetris.reinforcement_learning_configuration.framework = 'tf2'

# tetris.reinforcement_learning_configuration.architecture_name = 'dense'
# tetris.reinforcement_learning_configuration.architecture_configuration = {
#     'configuration_hidden_layers': [1024, 1024, 1024, 1024],
#     'activation_function': LeakyReLU(),
# }

tetris.reinforcement_learning_configuration.number_environment_runners = 1
tetris.reinforcement_learning_configuration.number_environment_per_environment_runners = 1
tetris.reinforcement_learning_configuration.batch_mode = 'complete_episodes'
tetris.reinforcement_learning_configuration.train_batch_size = 80_000
tetris.reinforcement_learning_configuration.mini_batch_size_per_learner = 20_000
