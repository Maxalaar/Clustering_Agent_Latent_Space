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
lunar_lander.trajectory_dataset_generation_configuration.number_rendering_to_stack = 15
lunar_lander.trajectory_dataset_generation_configuration.number_environment_runners = 5
lunar_lander.trajectory_dataset_generation_configuration.number_iterations = 10
lunar_lander.trajectory_dataset_generation_configuration.number_gpus_per_environment_runners = 1/lunar_lander.trajectory_dataset_generation_configuration.number_environment_runners
lunar_lander.trajectory_dataset_generation_configuration.minimal_steps_per_iteration_per_environment_runners = 1_000

# Surrogate Policy Training
lunar_lander.surrogate_policy_training_configuration.batch_size = 1_000 #20_000
lunar_lander.surrogate_policy_training_configuration.mini_chunk_size = 100_000
lunar_lander.surrogate_policy_training_configuration.number_mini_chunks = 2
# lunar_lander.surrogate_policy_training_configuration.clusterization_loss = None
# lunar_lander.surrogate_policy_training_configuration.architecture_configuration = {
#     'shape_layers': [128, 64, 32, 16, 32, 64, 128],
#     'latent_space_to_clusterize': [False, False, False, True, False, False, False],
# }
lunar_lander.surrogate_policy_training_configuration.architecture_configuration = {
    'shape_layers': [32, 16, 16, 32],
    'latent_space_to_clusterize': [False, True, True, False],
}
lunar_lander.surrogate_policy_training_configuration.clusterization_function_configuration.update({
    'number_cluster': 16,
    'number_points_for_silhouette_score': 1000,
    # 'memory_size': 0,
})

lunar_lander.surrogate_policy_training_configuration.clusterization_loss = SilhouetteLoss
lunar_lander.surrogate_policy_training_configuration.clusterization_loss_configuration.update({
    # 'margin_between_clusters': 10.0,
})
