from configurations.structure.experimentation_configuration import ExperimentationConfiguration
from environments.pong_survivor.configurations import classic_two_balls
from rllib.architectures.dense_ppo import DensePPO

pong_survivor_two_balls = ExperimentationConfiguration(
    experimentation_name='pong_survivor_tow_balls',
    environment_name='PongSurvivor',
)
pong_survivor_two_balls.environment_configuration = classic_two_balls

# Ray
pong_survivor_two_balls.ray_local_mode = False

# Reinforcement Learning
pong_survivor_two_balls.reinforcement_learning_configuration.architecture = DensePPO
pong_survivor_two_balls.reinforcement_learning_configuration.train_batch_size = 40_000
pong_survivor_two_balls.reinforcement_learning_configuration.minibatch_size = 10_000

# Video Episodes
pong_survivor_two_balls.video_episodes_generation_configuration.number_environment_runners = 5
pong_survivor_two_balls.video_episodes_generation_configuration.number_gpus_per_environment_runners = 1/pong_survivor_two_balls.video_episodes_generation_configuration.number_environment_runners

# Trajectory Dataset Generation
pong_survivor_two_balls.trajectory_dataset_generation_configuration.number_environment_runners = 10
pong_survivor_two_balls.trajectory_dataset_generation_configuration.number_gpus_per_environment_runners = 1/pong_survivor_two_balls.trajectory_dataset_generation_configuration.number_environment_runners
pong_survivor_two_balls.trajectory_dataset_generation_configuration.number_iterations = 300
pong_survivor_two_balls.trajectory_dataset_generation_configuration.minimal_steps_per_iteration_per_environment_runners = 1_000

# pong_survivor_two_balls.trajectory_dataset_generation_configuration.save_rendering = True
# pong_survivor_two_balls.trajectory_dataset_generation_configuration.number_environment_runners = 5
# pong_survivor_two_balls.trajectory_dataset_generation_configuration.number_gpus_per_environment_runners = 1/pong_survivor_two_balls.trajectory_dataset_generation_configuration.number_environment_runners
# pong_survivor_two_balls.trajectory_dataset_generation_configuration.number_iterations = 10
# pong_survivor_two_balls.trajectory_dataset_generation_configuration.minimal_steps_per_iteration_per_environment_runners = 1_000

# Surrogate Policy Training
pong_survivor_two_balls.surrogate_policy_training_configuration.batch_size = 20_000
pong_survivor_two_balls.surrogate_policy_training_configuration.mini_chunk_size = 100_000
pong_survivor_two_balls.surrogate_policy_training_configuration.number_mini_chunks = 2
pong_survivor_two_balls.surrogate_policy_training_configuration.architecture_configuration = {
    'shape_layers': [128, 64, 32, 16, 32, 64, 128],
    'latent_space_to_clusterize': [False, False, True, True, True, False, False],
}
pong_survivor_two_balls.surrogate_policy_training_configuration.clusterization_loss_configuration.update({
    'number_cluster': 4,
    'margin_between_clusters': 10.0,
    # 'sliding_centroids': True,
    # 'margin_between_clusters': 2.0,
    # 'margin_intra_cluster': 0.0,
    # 'attraction_loss_coefficient': 1.0,
    # 'repulsion_loss_coefficient': 1.0,
})
