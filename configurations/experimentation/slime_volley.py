from datetime import timedelta

from configurations.structure.experimentation_configuration import ExperimentationConfiguration
from rllib_repertory.architectures.dense_ppo import DensePPO

slime_volley = ExperimentationConfiguration(
    experimentation_name='slime_volley',
    environment_name='SlimeVolleyRllib',
)

# Ray
slime_volley.ray_local_mode = False

# Reinforcement Learning
slime_volley.reinforcement_learning_configuration.training_name = 'base'
slime_volley.reinforcement_learning_configuration.architecture = DensePPO
slime_volley.reinforcement_learning_configuration.number_environment_runners = 16
slime_volley.reinforcement_learning_configuration.number_environment_per_environment_runners = 1
slime_volley.reinforcement_learning_configuration.number_gpus_per_learner = 1
slime_volley.reinforcement_learning_configuration.train_batch_size = 10_000
slime_volley.reinforcement_learning_configuration.minibatch_size = 10_000
slime_volley.reinforcement_learning_configuration.batch_mode = 'complete_episodes'
slime_volley.reinforcement_learning_configuration.number_epochs = 32
slime_volley.reinforcement_learning_configuration.clip_policy_parameter = 0.1
slime_volley.reinforcement_learning_configuration.entropy_coefficient = 0.001
slime_volley.reinforcement_learning_configuration.number_checkpoint_to_keep = 10_000

# Video Episodes
slime_volley.video_episodes_generation_configuration.number_environment_runners = 2
slime_volley.video_episodes_generation_configuration.minimal_number_videos = 10

# Trajectory Dataset Generation
slime_volley.trajectory_dataset_generation_configuration.number_environment_runners = 10
slime_volley.trajectory_dataset_generation_configuration.number_gpus_per_environment_runners = 0.1
slime_volley.trajectory_dataset_generation_configuration.number_iterations = 100
slime_volley.trajectory_dataset_generation_configuration.minimal_steps_per_iteration_per_environment_runners = 1000

# Rendering Trajectory Dataset Generation
slime_volley.rendering_trajectory_dataset_generation_configuration.number_environment_runners = 5
slime_volley.rendering_trajectory_dataset_generation_configuration.number_iterations = 10
slime_volley.rendering_trajectory_dataset_generation_configuration.minimal_steps_per_iteration_per_environment_runners = 100

# Surrogate Policy Training
# slime_volley.surrogate_policy_training_configuration.training_name = 'new_architecture_0.0005_clusterization_loss'
# slime_volley.surrogate_policy_training_configuration.clusterization_loss_coefficient = 0.0005
# slime_volley.surrogate_policy_training_configuration.architecture_configuration = {
#     'shape_layers': [64, 64],
#     'indexes_latent_space_to_clusterize': [1, 3],
# }
# slime_volley.surrogate_policy_training_configuration.clusterization_function_configuration.update({
#     'number_cluster': 4,
# })
# slime_volley.surrogate_policy_training_configuration.clusterization_loss_configuration.update({
#     'number_centroids_repulsion': 1,
# })


# slime_volley.surrogate_policy_training_configuration.training_name = '0.0005_clusterization_loss'
# slime_volley.surrogate_policy_training_configuration.clusterization_loss_coefficient = 0.0005
# slime_volley.surrogate_policy_training_configuration.clusterization_function_configuration.update({
#     'number_cluster': 4,
# })
# slime_volley.surrogate_policy_training_configuration.clusterization_loss_configuration.update({
#     'number_centroids_repulsion': 1,
# })

# slime_volley.surrogate_policy_training_configuration.training_name = '4_cluster_2_repulsion_0.05_clusterization_loss'
# slime_volley.surrogate_policy_training_configuration.clusterization_loss_coefficient = 0.05
# slime_volley.surrogate_policy_training_configuration.clusterization_function_configuration.update({
#     'number_cluster': 8,
# })
# slime_volley.surrogate_policy_training_configuration.clusterization_loss_configuration.update({
#     'number_centroids_repulsion': 2,
# })

slime_volley.surrogate_policy_training_configuration.training_name = 'mass_0.0005_clusterization_loss'
slime_volley.surrogate_policy_training_configuration.clusterization_loss_coefficient = 0.0005
slime_volley.surrogate_policy_training_configuration.clusterization_function_configuration.update({
    'number_cluster': 4,
})
slime_volley.surrogate_policy_training_configuration.clusterization_loss_configuration.update({
    'number_centroids_repulsion': 1,
})
slime_volley.surrogate_policy_training_configuration.number_surrogate_policies_to_train = 5
slime_volley.surrogate_policy_training_configuration.maximum_training_time_by_policy = timedelta(hours=2, minutes=0)
