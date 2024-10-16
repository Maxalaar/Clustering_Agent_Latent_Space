from configurations.structure.experimentation_configuration import ExperimentationConfiguration


cartpole = ExperimentationConfiguration(
    experimentation_name='cartpole',
    environment_name='CartPoleRllib',
)

cartpole.reinforcement_learning_configuration.ray_local_mode = False
