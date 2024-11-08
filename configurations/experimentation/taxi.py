from ray.rllib.examples.connectors.classes.count_based_curiosity import CountBasedCuriosity
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)

from configurations.structure.experimentation_configuration import ExperimentationConfiguration

taxi = ExperimentationConfiguration(
    experimentation_name='taxi',
    environment_name='TaxiRllib',
)
taxi.ray_local_mode = True
# taxi.reinforcement_learning_configuration.learner_connector = CountBasedCuriosity(intrinsic_reward_coeff=1000)

# taxi.reinforcement_learning_configuration = ReinforcementLearningConfiguration()
# taxi.reinforcement_learning_configuration.number_environment_runners = 2
# taxi.reinforcement_learning_configuration.architecture_name = 'dense'
# taxi.reinforcement_learning_configuration.architecture_configuration = {
#     'configuration_hidden_layers': [64, 64],
# }
# taxi.reinforcement_learning_configuration.evaluation_num_environment_runners = 1
# taxi.reinforcement_learning_configuration.checkpoint_frequency = 50

# taxi.reinforcement_learning_configuration.algorithm_name = 'PPO'
# taxi.reinforcement_learning_configuration.number_environment_runners = 2
# taxi.reinforcement_learning_configuration.exploration_configuration = {
#     'type': 'EpsilonGreedy',
#     'initial_epsilon': 1.0,
#     'final_epsilon': 0.05,
#     'epsilon_timesteps': 1_000_000,
# }
#
# # taxi.reinforcement_learning_configuration.train_batch_size = 40_000
# # taxi.reinforcement_learning_configuration.mini_batch_size_per_learner = 10_000
#
# taxi.trajectory_dataset_generation_configuration.number_iterations = 100
# taxi.trajectory_dataset_generation_configuration.minimal_steps_per_iteration = 80_000
# # taxi.trajectory_dataset_generation_configuration.number_environment_runners = 0
# taxi.trajectory_dataset_generation_configuration.number_environment_per_environment_runners = 4
