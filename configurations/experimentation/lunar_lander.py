from configurations.structure.experimentation_configuration import ExperimentationConfiguration


lunar_lander = ExperimentationConfiguration(
    experimentation_name='lunar_lander',
    environment_name='LunarLanderRllib',
)
lunar_lander.reinforcement_learning_configuration.train_batch_size = 40_000
lunar_lander.reinforcement_learning_configuration.mini_batch_size_per_learner = 10_000

lunar_lander.trajectory_dataset_generation_configuration.number_iterations = 100
lunar_lander.trajectory_dataset_generation_configuration.minimal_steps_per_iteration = 80_000
lunar_lander.trajectory_dataset_generation_configuration.number_environment_runners = 10
lunar_lander.trajectory_dataset_generation_configuration.number_environment_per_environment_runners = 2
