from datetime import timedelta
from configurations.structure.experimentation_configuration import ExperimentationConfiguration
from configurations.structure.surrogate_policy_training_configuration import SurrogatePolicyTrainingConfiguration

multi_bipedal_walker = ExperimentationConfiguration(
    experimentation_name='multi_bipedal_walker',
    environment_name='BipedalWalkerRllib',
)

# Ray
multi_bipedal_walker.ray_local_mode = False

# Surrogate Policy Training
surrogate_policy_training_configurations = []
for number_cluster in [2, 4, 6]:

    surrogate_policy_training_configuration = SurrogatePolicyTrainingConfiguration()
    surrogate_policy_training_configuration.clusterization_loss_configuration.update({
        'number_centroids_repulsion': 1,
    })
    surrogate_policy_training_configuration.number_surrogate_policies_to_train = 3
    surrogate_policy_training_configuration.maximum_training_time_by_policy = timedelta(minutes=60*2)
    surrogate_policy_training_configuration.clusterization_function_configuration.update({
        'number_cluster': number_cluster,
    })
    surrogate_policy_training_configuration.training_name = str(number_cluster) + '_cluster_1_repulsion'

    surrogate_policy_training_configurations.append(surrogate_policy_training_configuration)

multi_bipedal_walker.surrogate_policy_training_configuration = surrogate_policy_training_configurations