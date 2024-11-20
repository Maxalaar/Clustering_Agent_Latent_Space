from ray.rllib.examples.connectors.classes.count_based_curiosity import CountBasedCuriosity
from configurations.structure.experimentation_configuration import ExperimentationConfiguration

taxi = ExperimentationConfiguration(
    experimentation_name='taxi',
    environment_name='TaxiRllib',
)

# Ray
taxi.ray_local_mode = False

# Reinforcement Learning
taxi.reinforcement_learning_configuration.learner_connector = CountBasedCuriosity
taxi.reinforcement_learning_configuration.number_environment_runners = 15

# Trajectory Dataset Generation
# taxi.trajectory_dataset_generation_configuration.number_environment_runners = 10
# taxi.trajectory_dataset_generation_configuration.number_gpus_per_environment_runners = 1/taxi.trajectory_dataset_generation_configuration.number_environment_runners
# taxi.trajectory_dataset_generation_configuration.number_iterations = 300
# taxi.trajectory_dataset_generation_configuration.minimal_steps_per_iteration_per_environment_runners = 1_000

taxi.trajectory_dataset_generation_configuration.save_rendering = True
taxi.trajectory_dataset_generation_configuration.number_environment_runners = 5
taxi.trajectory_dataset_generation_configuration.number_gpus_per_environment_runners = 1/taxi.trajectory_dataset_generation_configuration.number_environment_runners
taxi.trajectory_dataset_generation_configuration.number_iterations = 10
taxi.trajectory_dataset_generation_configuration.minimal_steps_per_iteration_per_environment_runners = 1_000

# Surrogate Policy Training
taxi.surrogate_policy_training_configuration.batch_size = 20_000
taxi.surrogate_policy_training_configuration.mini_chunk_size = 100_000
taxi.surrogate_policy_training_configuration.number_mini_chunks = 2
taxi.surrogate_policy_training_configuration.clusterization_loss_configuration.update({
    'number_cluster': 4,
    'margin_between_clusters': 10.0,
})
