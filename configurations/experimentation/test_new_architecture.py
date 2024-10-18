from configurations.structure.experimentation_configuration import ExperimentationConfiguration
from environments.pong_survivor.configurations import classic_two_balls

test_new_architecture = ExperimentationConfiguration(
    experimentation_name='test_new_architecture',
    environment_name='AntRllib',    # CartPoleRllib,
)
# test_new_architecture.environment_configuration = classic_two_balls

test_new_architecture.reinforcement_learning_configuration.ray_local_mode = False
test_new_architecture.reinforcement_learning_configuration.architecture_name = 'attention'
# test_new_architecture.reinforcement_learning_configuration.architecture_name = 'transformer'
test_new_architecture.reinforcement_learning_configuration.train_batch_size = 246*32
test_new_architecture.reinforcement_learning_configuration.mini_batch_size_per_learner = 246
