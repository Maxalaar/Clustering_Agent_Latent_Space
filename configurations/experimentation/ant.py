from configurations.structure.experimentation_configuration import ExperimentationConfiguration
from lightning.clustering_loss.new_kmeans_loss import NewKMeansLoss

ant = ExperimentationConfiguration(
    experimentation_name='ant',
    environment_name='AntRllib',
)

# Learning
# marvine_ant.reinforcement_learning_configuration.grad_clip = 0.5
ant.reinforcement_learning_configuration.use_generalized_advantage_estimator = True
ant.reinforcement_learning_configuration.learning_rate = 3e-4
ant.reinforcement_learning_configuration.lambda_gae = 0.95
# ant.reinforcement_learning_configuration.train_batch_size = 2048 * 4
# ant.reinforcement_learning_configuration.mini_batch_size_per_learner = 64 * 4

ant.reinforcement_learning_configuration.train_batch_size = 80_000
ant.reinforcement_learning_configuration.mini_batch_size_per_learner = 20_000
# ant.reinforcement_learning_configuration.number_gpus_per_learner = 0

ant.reinforcement_learning_configuration.clip_all_parameter = 2.0
ant.reinforcement_learning_configuration.clip_policy_parameter = 2.0
ant.reinforcement_learning_configuration.clip_value_function_parameter = 2.0

# Dataset
# ant.trajectory_dataset_generation_configuration.number_environment_runners = 10
# ant.trajectory_dataset_generation_configuration.number_gpus_per_environment_runners = 1/ant.trajectory_dataset_generation_configuration.number_environment_runners
# ant.trajectory_dataset_generation_configuration.number_iterations = 300
# ant.trajectory_dataset_generation_configuration.minimal_steps_per_iteration = 10_000

ant.trajectory_dataset_generation_configuration.number_environment_runners = 1
ant.trajectory_dataset_generation_configuration.number_iterations = 1
ant.trajectory_dataset_generation_configuration.minimal_steps_per_iteration = 10000
ant.trajectory_dataset_generation_configuration.save_rendering = True

# Surrogate Policy Training
ant.surrogate_policy_training_configuration.batch_size = 20_000
ant.surrogate_policy_training_configuration.mini_chunk_size = 100_000
ant.surrogate_policy_training_configuration.number_mini_chunks = 2
ant.surrogate_policy_training_configuration.clusterization_loss_coefficient = 1.0
ant.surrogate_policy_training_configuration.clusterization_loss = NewKMeansLoss
ant.surrogate_policy_training_configuration.clusterization_loss_configuration.update({
    'number_cluster': 4,
})
