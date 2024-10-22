from configurations.structure.experimentation_configuration import ExperimentationConfiguration


tetris = ExperimentationConfiguration(
    experimentation_name='tetris',
    environment_name='TetrisRllib',
)
tetris.reinforcement_learning_configuration.train_batch_size = 80_000
tetris.reinforcement_learning_configuration.mini_batch_size_per_learner = 20_000
