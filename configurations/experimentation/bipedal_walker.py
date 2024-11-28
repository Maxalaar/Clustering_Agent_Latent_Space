from torch.nn import LeakyReLU
from configurations.structure.experimentation_configuration import ExperimentationConfiguration
from rllib.architectures.dense_ppo import DensePPO

bipedal_walker = ExperimentationConfiguration(
    experimentation_name='bipedal_walker',
    environment_name='BipedalWalkerRllib',
)

# Ray
bipedal_walker.ray_local_mode = False

# Reinforcement Learning
# https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml
bipedal_walker.reinforcement_learning_configuration.architecture = DensePPO
bipedal_walker.reinforcement_learning_configuration.architecture_configuration = {
    'configuration_hidden_layers': [512, 512, 512],
    'activation_function': LeakyReLU(),
}
bipedal_walker.reinforcement_learning_configuration.number_gpus_per_learner = 1
bipedal_walker.reinforcement_learning_configuration.number_environment_runners = 15
bipedal_walker.reinforcement_learning_configuration.learning_rate = 3e-4
bipedal_walker.reinforcement_learning_configuration.use_generalized_advantage_estimator = True
bipedal_walker.reinforcement_learning_configuration.lambda_gae = 0.95
bipedal_walker.reinforcement_learning_configuration.train_batch_size = 2048 * 4
bipedal_walker.reinforcement_learning_configuration.minibatch_size = 64 * 4
bipedal_walker.reinforcement_learning_configuration.number_epochs = 10

bipedal_walker.reinforcement_learning_configuration.gradient_clip = 0.1
bipedal_walker.reinforcement_learning_configuration.clip_all_parameter = 0.18
bipedal_walker.reinforcement_learning_configuration.clip_value_function_parameter = 100

# Trajectory Dataset Generation
# bipedal_walker.trajectory_dataset_generation_configuration.number_environment_runners = 10
# bipedal_walker.trajectory_dataset_generation_configuration.number_gpus_per_environment_runners = 1/bipedal_walker.trajectory_dataset_generation_configuration.number_environment_runners
# bipedal_walker.trajectory_dataset_generation_configuration.number_iterations = 300
# bipedal_walker.trajectory_dataset_generation_configuration.minimal_steps_per_iteration_per_environment_runners = 1_000

bipedal_walker.trajectory_dataset_generation_configuration.save_rendering = True
bipedal_walker.trajectory_dataset_generation_configuration.number_rendering_to_stack = 15
bipedal_walker.trajectory_dataset_generation_configuration.number_environment_runners = 5
bipedal_walker.trajectory_dataset_generation_configuration.number_gpus_per_environment_runners = 1/bipedal_walker.trajectory_dataset_generation_configuration.number_environment_runners
bipedal_walker.trajectory_dataset_generation_configuration.number_iterations = 10
bipedal_walker.trajectory_dataset_generation_configuration.minimal_steps_per_iteration_per_environment_runners = 1_000

# Surrogate Policy Training
bipedal_walker.surrogate_policy_training_configuration.batch_size = 20_000
bipedal_walker.surrogate_policy_training_configuration.mini_chunk_size = 100_000
bipedal_walker.surrogate_policy_training_configuration.number_mini_chunks = 2
bipedal_walker.surrogate_policy_training_configuration.architecture_configuration = {
    'shape_layers': [128, 64, 32, 16, 32, 64, 128],
    'latent_space_to_clusterize': [False, False, True, True, True, False, False],
}
bipedal_walker.surrogate_policy_training_configuration.clusterization_loss_configuration.update({
    'number_cluster': 4,
    'margin_between_clusters': 10.0,
})

