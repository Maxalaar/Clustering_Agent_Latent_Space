from configurations.structure.experimentation_configuration import ExperimentationConfiguration

lunar_lander = ExperimentationConfiguration(
    experimentation_name='lunar_lander',
    environment_name='LunarLanderRllib',
)
lunar_lander.reinforcement_learning_configuration.train_batch_size = 40_000
lunar_lander.reinforcement_learning_configuration.minibatch_size = 10_000

# lunar_lander.trajectory_dataset_generation_configuration.number_environment_runners = 10
# lunar_lander.trajectory_dataset_generation_configuration.number_gpus_per_environment_runners = 1/lunar_lander.trajectory_dataset_generation_configuration.number_environment_runners
# lunar_lander.trajectory_dataset_generation_configuration.number_iterations = 300
# lunar_lander.trajectory_dataset_generation_configuration.minimal_steps_per_iteration = 10_000

lunar_lander.trajectory_dataset_generation_configuration.number_environment_runners = 1
lunar_lander.trajectory_dataset_generation_configuration.number_iterations = 1
lunar_lander.trajectory_dataset_generation_configuration.minimal_steps_per_iteration_per_environment_runners = 10000
lunar_lander.trajectory_dataset_generation_configuration.save_rendering = True

# lunar_lander.surrogate_policy_training_configuration.batch_size = 20_000
# lunar_lander.surrogate_policy_training_configuration.mini_chunk_size = 100_000
# lunar_lander.surrogate_policy_training_configuration.number_mini_chunks = 2
# lunar_lander.surrogate_policy_training_configuration.clusterization_loss_coefficient = 1.0
# lunar_lander.surrogate_policy_training_configuration.clusterization_loss_configuration.update({
#     'number_cluster': 4,
#     'sliding_centroids': True,
#     'margin_between_clusters': 1.0,
#     'margin_intra_cluster': 0.0,
#     'attraction_loss_coefficient': 1.0,
#     'repulsion_loss_coefficient': 1.0,
# })

lunar_lander.surrogate_policy_training_configuration.batch_size = 20_000
lunar_lander.surrogate_policy_training_configuration.mini_chunk_size = 100_000
lunar_lander.surrogate_policy_training_configuration.number_mini_chunks = 2
lunar_lander.surrogate_policy_training_configuration.clusterization_loss_configuration.update({
    'number_cluster': 4,
    # 'sliding_centroids': True,
    # 'margin_between_clusters': 2.0,
    # 'margin_intra_cluster': 0.0,
    # 'attraction_loss_coefficient': 1.0,
    # 'repulsion_loss_coefficient': 1.0,
})
