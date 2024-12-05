from torch.nn import LeakyReLU

from configurations.structure.experimentation_configuration import ExperimentationConfiguration
from rllib_repertory.architectures.dense_ppo import DensePPO

flappy_bird = ExperimentationConfiguration(
    experimentation_name='flappy_bird',
    environment_name='FlappyBirdRllib',
)

# Reinforcement Learning
flappy_bird.reinforcement_learning_configuration.architecture = DensePPO
flappy_bird.reinforcement_learning_configuration.batch_mode = 'truncate_episodes'
flappy_bird.reinforcement_learning_configuration.architecture = DensePPO
flappy_bird.reinforcement_learning_configuration.architecture_configuration = {
    'configuration_hidden_layers': [128, 128, 128],
    'activation_function': LeakyReLU(),
}
flappy_bird.reinforcement_learning_configuration.train_batch_size = 80_000
flappy_bird.reinforcement_learning_configuration.minibatch_size = 20_000
flappy_bird.reinforcement_learning_configuration.number_environment_runners = 10
flappy_bird.reinforcement_learning_configuration.number_gpus_per_learner = 1

# Video Episodes Generation
flappy_bird.video_episodes_generation_configuration.minimal_number_videos = 10
flappy_bird.video_episodes_generation_configuration.number_environment_runners = 1

# Trajectory Dataset Generation
# flappy_bird.trajectory_dataset_generation_configuration.number_environment_runners = 5
# flappy_bird.trajectory_dataset_generation_configuration.number_gpus_per_environment_runners = 1/flappy_bird.trajectory_dataset_generation_configuration.number_environment_runners
# flappy_bird.trajectory_dataset_generation_configuration.number_iterations = 300
# flappy_bird.trajectory_dataset_generation_configuration.minimal_steps_per_iteration_per_environment_runners = 1_000

flappy_bird.trajectory_dataset_generation_configuration.save_rendering = True
flappy_bird.trajectory_dataset_generation_configuration.number_rendering_to_stack = 15
flappy_bird.trajectory_dataset_generation_configuration.number_environment_runners = 2
flappy_bird.trajectory_dataset_generation_configuration.number_gpus_per_environment_runners = 1/flappy_bird.trajectory_dataset_generation_configuration.number_environment_runners
flappy_bird.trajectory_dataset_generation_configuration.number_iterations = 10
flappy_bird.trajectory_dataset_generation_configuration.minimal_steps_per_iteration_per_environment_runners = 1_000

# Surrogate Policy Training
flappy_bird.surrogate_policy_training_configuration.batch_size = 20_000
flappy_bird.surrogate_policy_training_configuration.mini_chunk_size = 100_000
flappy_bird.surrogate_policy_training_configuration.number_mini_chunks = 2
flappy_bird.surrogate_policy_training_configuration.architecture_configuration = {
    'shape_layers': [64, 64, 64, 64],
    'latent_space_to_clusterize': [False, True, True, False],
}
flappy_bird.surrogate_policy_training_configuration.clusterization_loss_configuration.update({
    'number_cluster': 4,
    'margin_between_clusters': 10.0,
    # 'sliding_centroids': True,
    # 'margin_between_clusters': 2.0,
    # 'margin_intra_cluster': 0.0,
    # 'attraction_loss_coefficient': 1.0,
    # 'repulsion_loss_coefficient': 1.0,
})
