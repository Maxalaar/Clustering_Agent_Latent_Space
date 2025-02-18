from configurations.structure.experimentation_configuration import ExperimentationConfiguration


cartpole = ExperimentationConfiguration(
    experimentation_name='cartpole',
    environment_name='CartPoleRllib',
)

cartpole.reinforcement_learning_configuration.number_gpus_per_learner = 0

cartpole.reinforcement_learning_configuration.number_environment_runners = 1
cartpole.reinforcement_learning_configuration.number_gpus_per_environment_runners = 1
