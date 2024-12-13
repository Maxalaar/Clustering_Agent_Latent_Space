from configurations.structure.experimentation_configuration import ExperimentationConfiguration


tetris_atari = ExperimentationConfiguration(
    experimentation_name='tetris_atari',
    environment_name='TetrisAtariRllib',
)

tetris_atari.reinforcement_learning_configuration.number_environment_runners = 8
tetris_atari.reinforcement_learning_configuration.number_environment_per_environment_runners = 1

tetris_atari.reinforcement_learning_configuration.number_gpus_per_environment_runners = 0
tetris_atari.reinforcement_learning_configuration.number_gpus_per_learner = 1
