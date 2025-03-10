from copy import deepcopy
from datetime import timedelta

from configurations.structure.experimentation_configuration import ExperimentationConfiguration
from configurations.structure.surrogate_policy_training_configuration import SurrogatePolicyTrainingConfiguration
from environments.pong_survivor.configurations import classic_two_balls

multi_pong_survivor_two_balls = ExperimentationConfiguration(
    experimentation_name='multi_pong_survivor_two_balls',
    environment_name='PongSurvivor',
)
multi_pong_survivor_two_balls.environment_configuration = classic_two_balls

# Ray
multi_pong_survivor_two_balls.ray_local_mode = False

# Surrogate Policy Trainings
surrogate_policy_training_configurations = []
for number_cluster in [2, 3, 4, 5, 6]:

    surrogate_policy_training_configuration = SurrogatePolicyTrainingConfiguration()
    surrogate_policy_training_configuration.clusterization_loss_configuration.update({
        'number_centroids_repulsion': 1,
    })
    surrogate_policy_training_configuration.number_surrogate_policies_to_train = 3
    surrogate_policy_training_configuration.maximum_training_time_by_policy = timedelta(minutes=60)
    surrogate_policy_training_configuration.clusterization_function_configuration.update({
        'number_cluster': number_cluster,
    })
    surrogate_policy_training_configuration.training_name = str(number_cluster) + '_cluster_1_repulsion'

    surrogate_policy_training_configurations.append(surrogate_policy_training_configuration)

multi_pong_survivor_two_balls.surrogate_policy_training_configuration = surrogate_policy_training_configurations

# Surrogate Policy Evaluation
multi_pong_survivor_two_balls.surrogate_policy_evaluation_configuration.evaluation_duration = 100
multi_pong_survivor_two_balls.surrogate_policy_evaluation_configuration.number_environment_runners = 5
multi_pong_survivor_two_balls.surrogate_policy_evaluation_configuration.number_gpus_per_environment_runners = 0.1