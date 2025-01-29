from ray.rllib.examples.connectors.classes.count_based_curiosity import CountBasedCuriosity
from configurations.structure.experimentation_configuration import ExperimentationConfiguration
from rllib_repertory.architectures.dense_ppo import DensePPO

taxi = ExperimentationConfiguration(
    experimentation_name='taxi',
    environment_name='TaxiRllib',
)
taxi.environment_configuration = {'new_observation_space': False}

# Ray
taxi.ray_local_mode = False

# Reinforcement Learning
taxi.reinforcement_learning_configuration.training_name = 'base'
taxi.reinforcement_learning_configuration.learner_connector = lambda *args, **kwargs: CountBasedCuriosity(intrinsic_reward_coeff=1)
taxi.reinforcement_learning_configuration.entropy_coefficient = 0.1
taxi.reinforcement_learning_configuration.train_batch_size = 10_000
taxi.reinforcement_learning_configuration.minibatch_size = 10_000
taxi.reinforcement_learning_configuration.number_environment_runners = 4

