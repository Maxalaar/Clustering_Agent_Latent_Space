from configurations.structure.experimentation_configuration import ExperimentationConfiguration


lunar_lander = ExperimentationConfiguration(
    experimentation_name='lunar_lander',
    environment_name='LunarLanderRllib',
)
lunar_lander.reinforcement_learning_configuration.train_batch_size = 40_000
lunar_lander.reinforcement_learning_configuration.mini_batch_size_per_learner = 10_000
