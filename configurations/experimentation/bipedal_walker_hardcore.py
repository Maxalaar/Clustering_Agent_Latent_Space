from torch.nn import LeakyReLU

from configurations.structure.experimentation_configuration import ExperimentationConfiguration
from rllib_repertory.architectures.dense_ppo import DensePPO
from rllib_repertory.architectures.transformer_ppo import TransformerPPO

bipedal_walker_hardcore = ExperimentationConfiguration(
    experimentation_name='bipedal_walker_hardcore',
    environment_name='BipedalWalkerRllib',
)
bipedal_walker_hardcore.environment_configuration = {'hardcore': True}

# Ray
bipedal_walker_hardcore.ray_local_mode = False
bipedal_walker_hardcore.reinforcement_learning_configuration.number_gpus_per_learner = 1
bipedal_walker_hardcore.reinforcement_learning_configuration.number_environment_runners = 16
bipedal_walker_hardcore.reinforcement_learning_configuration.number_environment_per_environment_runners = 1

# Reinforcement Learning
# https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml
# https://github.com/ovechkin-dm/ppo-lstm-parallel
bipedal_walker_hardcore.reinforcement_learning_configuration.training_name = 'base'
bipedal_walker_hardcore.reinforcement_learning_configuration.architecture = DensePPO
bipedal_walker_hardcore.reinforcement_learning_configuration.architecture_configuration = {
    'configuration_hidden_layers': [32, 64, 128, 64, 32],
    'activation_function': LeakyReLU(),
}
bipedal_walker_hardcore.reinforcement_learning_configuration.learning_rate = 1e-3
bipedal_walker_hardcore.reinforcement_learning_configuration.use_generalized_advantage_estimator = True
bipedal_walker_hardcore.reinforcement_learning_configuration.lambda_gae = 0.95
bipedal_walker_hardcore.reinforcement_learning_configuration.train_batch_size = 100_000
bipedal_walker_hardcore.reinforcement_learning_configuration.minibatch_size = 100_000
bipedal_walker_hardcore.reinforcement_learning_configuration.number_epochs = 32
bipedal_walker_hardcore.reinforcement_learning_configuration.entropy_coefficient = 0.0
bipedal_walker_hardcore.reinforcement_learning_configuration.clip_policy_parameter = 0.01

# Video Episodes
bipedal_walker_hardcore.video_episodes_generation_configuration.number_environment_runners = 5

# Trajectory Dataset Generation
bipedal_walker_hardcore.trajectory_dataset_generation_configuration.number_environment_runners = 10
bipedal_walker_hardcore.trajectory_dataset_generation_configuration.number_iterations = 100
bipedal_walker_hardcore.trajectory_dataset_generation_configuration.minimal_steps_per_iteration_per_environment_runners = 1000

# Rendering Trajectory Dataset Generation
bipedal_walker_hardcore.rendering_trajectory_dataset_generation_configuration.number_environment_runners = 5
bipedal_walker_hardcore.rendering_trajectory_dataset_generation_configuration.number_iterations = 10
bipedal_walker_hardcore.rendering_trajectory_dataset_generation_configuration.minimal_steps_per_iteration_per_environment_runners = 100

# Surrogate Policy Training
bipedal_walker_hardcore.surrogate_policy_training_configuration.training_name = '4_cluster_2_repulsion'
bipedal_walker_hardcore.surrogate_policy_training_configuration.clusterization_function_configuration.update({
    'number_cluster': 4,
})
bipedal_walker_hardcore.surrogate_policy_training_configuration.clusterization_loss_configuration.update({
    'number_centroids_repulsion': 2,
})

# Surrogate Policy Evaluation
bipedal_walker_hardcore.surrogate_policy_evaluation_configuration.evaluation_duration = 200
bipedal_walker_hardcore.surrogate_policy_evaluation_configuration.number_environment_runners = 5
bipedal_walker_hardcore.surrogate_policy_evaluation_configuration.number_gpus_per_environment_runners = 0.1
