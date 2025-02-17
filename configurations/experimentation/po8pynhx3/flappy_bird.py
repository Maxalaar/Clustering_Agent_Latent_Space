from datetime import timedelta

from torch.nn import LeakyReLU

from configurations.structure.experimentation_configuration import ExperimentationConfiguration
from rllib_repertory.architectures.dense_ppo import DensePPO

flappy_bird = ExperimentationConfiguration(
    experimentation_name='flappy_bird',
    environment_name='FlappyBirdRllib',
)

# Ray
flappy_bird.ray_local_mode = False

# Reinforcement Learning
flappy_bird.reinforcement_learning_configuration.training_name = 'base'
flappy_bird.reinforcement_learning_configuration.architecture = DensePPO
flappy_bird.reinforcement_learning_configuration.number_environment_runners = 16
flappy_bird.reinforcement_learning_configuration.number_environment_per_environment_runners = 1
flappy_bird.reinforcement_learning_configuration.number_gpus_per_learner = 1
flappy_bird.reinforcement_learning_configuration.train_batch_size = 40_000
flappy_bird.reinforcement_learning_configuration.minibatch_size = 10_000
flappy_bird.reinforcement_learning_configuration.batch_mode = 'truncate_episodes'
flappy_bird.reinforcement_learning_configuration.number_epochs = 16
flappy_bird.reinforcement_learning_configuration.clip_policy_parameter = 0.1

# Video Episodes
flappy_bird.video_episodes_generation_configuration.number_environment_runners = 5

# Trajectory Dataset Generation
flappy_bird.trajectory_dataset_generation_configuration.number_environment_runners = 10
flappy_bird.trajectory_dataset_generation_configuration.number_iterations = 100
flappy_bird.trajectory_dataset_generation_configuration.minimal_steps_per_iteration_per_environment_runners = 1000

# Rendering Trajectory Dataset Generation
flappy_bird.rendering_trajectory_dataset_generation_configuration.number_environment_runners = 5
flappy_bird.rendering_trajectory_dataset_generation_configuration.number_iterations = 10
flappy_bird.rendering_trajectory_dataset_generation_configuration.minimal_steps_per_iteration_per_environment_runners = 100

# Surrogate Policy Training
# flappy_bird.surrogate_policy_training_configuration.training_name = 'new_architecture_1.0_clusterization_loss_2_cluster_1_repulsion'
# flappy_bird.surrogate_policy_training_configuration.clusterization_loss_coefficient = 1.0
# flappy_bird.surrogate_policy_training_configuration.architecture_configuration = {
#     'shape_layers': [64, 64],
#     'indexes_latent_space_to_clusterize': [1, 3],
# }
# flappy_bird.surrogate_policy_training_configuration.clusterization_function_configuration.update({
#     'number_cluster': 2,
# })
# flappy_bird.surrogate_policy_training_configuration.clusterization_loss_configuration.update({
#     'number_centroids_repulsion': 1,
# })

# flappy_bird.surrogate_policy_training_configuration.training_name = '4_cluster_2_repulsion_0.05_clusterization_loss'
# flappy_bird.surrogate_policy_training_configuration.clusterization_loss_coefficient = 0.05
# flappy_bird.surrogate_policy_training_configuration.clusterization_function_configuration.update({
#     'number_cluster': 4,
# })
# flappy_bird.surrogate_policy_training_configuration.clusterization_loss_configuration.update({
#     'number_centroids_repulsion': 2,
# })

flappy_bird.surrogate_policy_training_configuration.training_name = 'mass_4_cluster_2_repulsion'
flappy_bird.surrogate_policy_training_configuration.clusterization_loss_coefficient = 1
flappy_bird.surrogate_policy_training_configuration.clusterization_function_configuration.update({
    'number_cluster': 4,
})
flappy_bird.surrogate_policy_training_configuration.clusterization_loss_configuration.update({
    'number_centroids_repulsion': 2,
})
flappy_bird.surrogate_policy_training_configuration.number_surrogate_policies_to_train = 4
flappy_bird.surrogate_policy_training_configuration.maximum_training_time_by_policy = timedelta(minutes=30)