from configurations.reinforcement_learning.basic_reinforcement_learning_configuration import basic_reinforcement_learning_configuration
from configurations.structure.experimentation_configuration import ExperimentationConfiguration


cartpole = ExperimentationConfiguration(
    experimentation_name='cartpole',
    environment_name='CartPoleRllib',
)
