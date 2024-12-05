from configurations.structure.experimentation_configuration import ExperimentationConfiguration
from lightning.clusterization_loss.silhouette_loss import SilhouetteLoss

lunar_lander = ExperimentationConfiguration(
    experimentation_name='lunar_lander',
    environment_name='LunarLanderRllib',
)

# Reinforcement Learning
lunar_lander.reinforcement_learning_configuration.number_environment_runners = 15
lunar_lander.reinforcement_learning_configuration.number_gpus_per_learner = 1
lunar_lander.reinforcement_learning_configuration.train_batch_size = 40_000
lunar_lander.reinforcement_learning_configuration.minibatch_size = 10_000

# Trajectory Dataset Generation
# lunar_lander.trajectory_dataset_generation_configuration.number_environment_runners = 10
# lunar_lander.trajectory_dataset_generation_configuration.number_gpus_per_environment_runners = 1/lunar_lander.trajectory_dataset_generation_configuration.number_environment_runners
# lunar_lander.trajectory_dataset_generation_configuration.number_iterations = 300
# lunar_lander.trajectory_dataset_generation_configuration.minimal_steps_per_iteration = 1_000

lunar_lander.trajectory_dataset_generation_configuration.save_rendering = True
lunar_lander.trajectory_dataset_generation_configuration.number_environment_runners = 5
lunar_lander.trajectory_dataset_generation_configuration.number_iterations = 10
lunar_lander.trajectory_dataset_generation_configuration.number_gpus_per_environment_runners = 1/lunar_lander.trajectory_dataset_generation_configuration.number_environment_runners
lunar_lander.trajectory_dataset_generation_configuration.minimal_steps_per_iteration_per_environment_runners = 1_000

# Surrogate Policy Training
lunar_lander.surrogate_policy_training_configuration.mini_chunk_size = 100_000
lunar_lander.surrogate_policy_training_configuration.number_mini_chunks = 2
lunar_lander.surrogate_policy_training_configuration.architecture_configuration = {
    'shape_layers': [64, 32, 32, 64],
    'indexes_latent_space_to_clusterize': [3, 5],
}
lunar_lander.surrogate_policy_training_configuration.clusterization_function_configuration.update({
    'number_cluster': 4,
    'number_points_for_silhouette_score': 1_000,
    'memory_size': 100_000,
})
lunar_lander.surrogate_policy_training_configuration.batch_size = 5_000
lunar_lander.surrogate_policy_training_configuration.clusterization_loss_configuration.update({
    'margin_between_clusters': 10.0,
})
