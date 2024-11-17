from configurations.structure.experimentation_configuration import ExperimentationConfiguration


flappy_bird = ExperimentationConfiguration(
    experimentation_name='flappy_bird',
    environment_name='FlappyBirdRllib',
)
flappy_bird.reinforcement_learning_configuration.batch_mode = 'truncate_episodes'
flappy_bird.reinforcement_learning_configuration.architecture = 'dense'
flappy_bird.reinforcement_learning_configuration.train_batch_size = 80_000
flappy_bird.reinforcement_learning_configuration.minibatch_size = 20_000
