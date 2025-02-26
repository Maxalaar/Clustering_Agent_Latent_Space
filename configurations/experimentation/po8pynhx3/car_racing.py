from torch.nn import LeakyReLU

from configurations.structure.experimentation_configuration import ExperimentationConfiguration
from rllib_repertory.architectures.cnn_ppo import CNNPPO

car_racing = ExperimentationConfiguration(
    experimentation_name='car_racing',
    environment_name='CarRacingRllib',
)
car_racing.ray_local_mode = False

car_racing.reinforcement_learning_configuration.training_name = 'car_racing_new_custom_model_v2'
car_racing.reinforcement_learning_configuration.flatten_observations = False
car_racing.reinforcement_learning_configuration.evaluation_duration = 10

car_racing.reinforcement_learning_configuration.architecture = CNNPPO
car_racing.reinforcement_learning_configuration.architecture_configuration = {
    'configuration_cnn': [(16, 8, 4), (32, 4, 4), (64, 3, 2), (128, 2, 1)],
    'configuration_hidden_layers': [64, 64],
    'activation_function_class': LeakyReLU,
    'use_layer_normalization_cnn': True,
}

car_racing.reinforcement_learning_configuration.number_gpus_per_learner = 1

car_racing.reinforcement_learning_configuration.number_environment_runners = 8
car_racing.reinforcement_learning_configuration.number_cpus_per_learner = 1
car_racing.reinforcement_learning_configuration.number_gpus_per_environment_runners = 0

car_racing.reinforcement_learning_configuration.evaluation_num_environment_runners = 8


