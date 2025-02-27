from datetime import timedelta

from configurations.structure.experimentation_configuration import ExperimentationConfiguration
from configurations.structure.surrogate_policy_training_configuration import SurrogatePolicyTrainingConfiguration

multi_lunar_lander = ExperimentationConfiguration(
    experimentation_name='multi_lunar_lander',
    environment_name='LunarLanderRllib',
)

# Ray
multi_lunar_lander.ray_local_mode = False

# Surrogate Policy Training
surrogate_policy_training_configurations = []
for number_cluster in [2, 3, 4, 5, 6]:

    surrogate_policy_training_configuration = SurrogatePolicyTrainingConfiguration()
    surrogate_policy_training_configuration.clusterization_loss_configuration.update({
        'number_centroids_repulsion': 1,
    })
    surrogate_policy_training_configuration.number_surrogate_policies_to_train = 3
    surrogate_policy_training_configuration.maximum_training_time_by_policy = timedelta(minutes=10)
    surrogate_policy_training_configuration.clusterization_function_configuration.update({
        'number_cluster': number_cluster,
    })
    surrogate_policy_training_configuration.training_name = str(number_cluster) + '_cluster_1_repulsion'

    surrogate_policy_training_configurations.append(surrogate_policy_training_configuration)

multi_lunar_lander.surrogate_policy_training_configuration = surrogate_policy_training_configurations