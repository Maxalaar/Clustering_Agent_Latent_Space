from torch.nn import LeakyReLU

from configurations.structure.experimentation_configuration import ExperimentationConfiguration
from rllib.architectures.dense_ppo import DensePPO

ant = ExperimentationConfiguration(
    experimentation_name='ant',
    environment_name='AntRllib',
)

ant.ray_local_mode = False

# Learning
ant.reinforcement_learning_configuration.architecture = DensePPO
ant.reinforcement_learning_configuration.architecture_configuration = {
    'configuration_hidden_layers': [512, 512, 512],
    'activation_function': LeakyReLU(),
}

# Reinforcement Learning
ant.reinforcement_learning_configuration.number_environment_runners = 10
ant.reinforcement_learning_configuration.number_gpus_per_learner = 1

ant.reinforcement_learning_configuration.use_generalized_advantage_estimator = True
ant.reinforcement_learning_configuration.learning_rate = 3e-4
ant.reinforcement_learning_configuration.lambda_gae = 0.95
ant.reinforcement_learning_configuration.train_batch_size = 80_000
ant.reinforcement_learning_configuration.minibatch_size = 20_000

ant.reinforcement_learning_configuration.clip_all_parameter = 2.0
ant.reinforcement_learning_configuration.clip_policy_parameter = 2.0
ant.reinforcement_learning_configuration.clip_value_function_parameter = 2.0

# Trajectory Dataset Generation
# ant.trajectory_dataset_generation_configuration.number_environment_runners = 10
# ant.trajectory_dataset_generation_configuration.number_gpus_per_environment_runners = 1/ant.trajectory_dataset_generation_configuration.number_environment_runners
# ant.trajectory_dataset_generation_configuration.number_iterations = 300
# ant.trajectory_dataset_generation_configuration.minimal_steps_per_iteration_per_environment_runners = 1_000

# ant.trajectory_dataset_generation_configuration.save_rendering = True
# ant.trajectory_dataset_generation_configuration.number_environment_runners = 5
# ant.trajectory_dataset_generation_configuration.number_iterations = 10
# ant.trajectory_dataset_generation_configuration.minimal_steps_per_iteration = 1_000

# Surrogate Policy Training
ant.surrogate_policy_training_configuration.batch_size = 20_000
ant.surrogate_policy_training_configuration.mini_chunk_size = 100_000
ant.surrogate_policy_training_configuration.number_mini_chunks = 2
ant.surrogate_policy_training_configuration.architecture_configuration = {
    'shape_layers': [128, 64, 32, 16, 32, 64, 128],
    'latent_space_to_clusterize': [False, False, True, True, True, False, False],
}
ant.surrogate_policy_training_configuration.clusterization_loss_configuration.update({
    'number_cluster': 4,
    'margin_between_clusters': 10.0,
})
