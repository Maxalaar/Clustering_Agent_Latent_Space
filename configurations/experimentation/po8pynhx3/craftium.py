from torch.nn import LeakyReLU

from configurations.structure.experimentation_configuration import ExperimentationConfiguration
from rllib_repertory.architectures.cnn_ppo import CNNPPO

craftium = ExperimentationConfiguration(
    experimentation_name='craftium',
    environment_name='CraftiumRllib',
)
craftium.ray_local_mode = False

craftium.reinforcement_learning_configuration.training_name = 'craftium_v2'
craftium.reinforcement_learning_configuration.flatten_observations = False
craftium.reinforcement_learning_configuration.evaluation_duration = 10

craftium.reinforcement_learning_configuration.architecture = CNNPPO
craftium.reinforcement_learning_configuration.architecture_configuration = {
    'configuration_cnn': [
        (16, 8, 4),   # Sortie: (16, 15, 15)
        (32, 3, 3),   # Sortie: (32, 5, 5)
        (64, 3, 2),   # Sortie: (64, 2, 2)
        (128, 2, 1)   # Sortie: (128, 1, 1)
    ],
    'configuration_hidden_layers': [64, 64],
    'activation_function_class': LeakyReLU,
    'use_layer_normalization_cnn': True,
}

craftium.reinforcement_learning_configuration.number_gpus_per_learner = 1

craftium.reinforcement_learning_configuration.number_environment_runners = 4
craftium.reinforcement_learning_configuration.number_cpus_per_environment_runners = 2
craftium.reinforcement_learning_configuration.number_gpus_per_environment_runners = 0

craftium.reinforcement_learning_configuration.evaluation_num_environment_runners = 1