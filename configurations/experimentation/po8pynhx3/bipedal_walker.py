from torch.nn import LeakyReLU
from configurations.structure.experimentation_configuration import ExperimentationConfiguration
from rllib_repertory.architectures.dense_ppo import DensePPO

bipedal_walker = ExperimentationConfiguration(
    experimentation_name='bipedal_walker',
    environment_name='BipedalWalkerRllib',
)

# Ray
bipedal_walker.ray_local_mode = False

# Reinforcement Learning
# https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml
bipedal_walker.reinforcement_learning_configuration.training_name = 'base'
bipedal_walker.reinforcement_learning_configuration.architecture = DensePPO
bipedal_walker.reinforcement_learning_configuration.number_environment_runners = 16
bipedal_walker.reinforcement_learning_configuration.number_environment_per_environment_runners = 1
bipedal_walker.reinforcement_learning_configuration.number_gpus_per_learner = 1
bipedal_walker.reinforcement_learning_configuration.train_batch_size = 40_000
bipedal_walker.reinforcement_learning_configuration.minibatch_size = 10_000
bipedal_walker.reinforcement_learning_configuration.batch_mode = 'complete_episodes'
bipedal_walker.reinforcement_learning_configuration.number_epochs = 16
bipedal_walker.reinforcement_learning_configuration.clip_policy_parameter = 0.1

# Video Episodes
bipedal_walker.video_episodes_generation_configuration.number_environment_runners = 5

# Trajectory Dataset Generation
bipedal_walker.trajectory_dataset_generation_configuration.number_environment_runners = 10
bipedal_walker.trajectory_dataset_generation_configuration.number_iterations = 100
bipedal_walker.trajectory_dataset_generation_configuration.minimal_steps_per_iteration_per_environment_runners = 1000

# Rendering Trajectory Dataset Generation
bipedal_walker.rendering_trajectory_dataset_generation_configuration.number_environment_runners = 5
bipedal_walker.rendering_trajectory_dataset_generation_configuration.number_iterations = 10
bipedal_walker.rendering_trajectory_dataset_generation_configuration.minimal_steps_per_iteration_per_environment_runners = 100

# Surrogate Policy Training
bipedal_walker.surrogate_policy_training_configuration.training_name = '4_cluster_1_repulsion'
bipedal_walker.surrogate_policy_training_configuration.clusterization_function_configuration.update({
    'number_cluster': 4,
})
bipedal_walker.surrogate_policy_training_configuration.clusterization_loss_configuration.update({
    'number_centroids_repulsion': 1,
})

# Surrogate Policy Evaluation
bipedal_walker.surrogate_policy_evaluation_configuration.evaluation_duration = 200
bipedal_walker.surrogate_policy_evaluation_configuration.number_environment_runners = 5
bipedal_walker.surrogate_policy_evaluation_configuration.number_gpus_per_environment_runners = 0.1


