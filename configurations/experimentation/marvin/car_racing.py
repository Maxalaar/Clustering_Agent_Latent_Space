from torch.nn import LeakyReLU

from configurations.structure.experimentation_configuration import ExperimentationConfiguration
from rllib_repertory.architectures.cnn_ppo import CNNPPO

car_racing = ExperimentationConfiguration(
    experimentation_name='car_racing',
    environment_name='CarRacingRllib',
)
car_racing.ray_local_mode = False

car_racing.reinforcement_learning_configuration.training_name = 'car_racing_new_custom_model_v8'
car_racing.reinforcement_learning_configuration.flatten_observations = False
car_racing.reinforcement_learning_configuration.evaluation_duration = 10

car_racing.reinforcement_learning_configuration.architecture = CNNPPO
car_racing.reinforcement_learning_configuration.architecture_configuration = {
    'configuration_cnn': [(16, 8, 4), (32, 4, 4), (64, 3, 2), (128, 2, 1)],
    'configuration_hidden_layers': [64, 64],
    'activation_function_class': LeakyReLU,
    'use_layer_normalization_cnn': True,
    'use_unified_cnn': True,
}

car_racing.reinforcement_learning_configuration.number_gpus_per_learner = 1

car_racing.reinforcement_learning_configuration.number_environment_runners = 2
car_racing.reinforcement_learning_configuration.number_cpus_per_learner = 1
car_racing.reinforcement_learning_configuration.number_gpus_per_environment_runners = 0

car_racing.reinforcement_learning_configuration.train_batch_size = 512 * 4
car_racing.reinforcement_learning_configuration.compress_observations = True
car_racing.reinforcement_learning_configuration.minibatch_size = 256
# car_racing.reinforcement_learning_configuration.clip_policy_parameter = 0.1
car_racing.reinforcement_learning_configuration.learning_rate = 1e-5
# car_racing.reinforcement_learning_configuration.entropy_coefficient = 0.005

car_racing.reinforcement_learning_configuration.evaluation_num_environment_runners = 1
car_racing.reinforcement_learning_configuration.number_checkpoint_to_keep = 1_000_000


