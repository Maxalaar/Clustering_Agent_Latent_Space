from torch.nn import LeakyReLU
from configurations.structure.experimentation_configuration import ExperimentationConfiguration

bipedal_walker = ExperimentationConfiguration(
    experimentation_name='bipedal_walker',
    environment_name='BipedalWalkerRllib',
)

bipedal_walker.reinforcement_learning_configuration.number_gpu = 1
bipedal_walker.reinforcement_learning_configuration.number_gpus_per_learner = 1

bipedal_walker.reinforcement_learning_configuration.number_environment_runners = 8
bipedal_walker.reinforcement_learning_configuration.ray_local_mode = False

# https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml
bipedal_walker.reinforcement_learning_configuration.use_generalized_advantage_estimator = True
bipedal_walker.reinforcement_learning_configuration.learning_rate = 3e-4
bipedal_walker.reinforcement_learning_configuration.lambda_gae = 0.95
bipedal_walker.reinforcement_learning_configuration.train_batch_size = 2048 * 4
bipedal_walker.reinforcement_learning_configuration.minibatch_size = 64 * 4

bipedal_walker.reinforcement_learning_configuration.clip_all_parameter = 0.18
bipedal_walker.reinforcement_learning_configuration.clip_policy_parameter = 0.18
bipedal_walker.reinforcement_learning_configuration.clip_value_function_parameter = 0.18


bipedal_walker.reinforcement_learning_configuration.architecture_configuration = {
    'configuration_hidden_layers': [512, 512, 512],
    'activation_function': LeakyReLU(),
}

# Dataset
# bipedal_walker.trajectory_dataset_generation_configuration.number_environment_runners = 10
# bipedal_walker.trajectory_dataset_generation_configuration.number_gpus_per_environment_runners = 1/bipedal_walker.trajectory_dataset_generation_configuration.number_environment_runners
# bipedal_walker.trajectory_dataset_generation_configuration.number_iterations = 300
# bipedal_walker.trajectory_dataset_generation_configuration.minimal_steps_per_iteration = 10_000

bipedal_walker.trajectory_dataset_generation_configuration.number_environment_runners = 1
bipedal_walker.trajectory_dataset_generation_configuration.number_iterations = 1
bipedal_walker.trajectory_dataset_generation_configuration.minimal_steps_per_iteration_per_environment_runners = 10000
bipedal_walker.trajectory_dataset_generation_configuration.save_rendering = True

# Surrogate Policy Training
bipedal_walker.surrogate_policy_training_configuration.batch_size = 20_000
bipedal_walker.surrogate_policy_training_configuration.mini_chunk_size = 100_000
bipedal_walker.surrogate_policy_training_configuration.number_mini_chunks = 2
bipedal_walker.surrogate_policy_training_configuration.clusterization_loss_coefficient = 1.0
bipedal_walker.surrogate_policy_training_configuration.clusterization_loss_configuration.update({
    'number_cluster': 4,
})
